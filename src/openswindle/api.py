"""FastAPI application: REST transport for the authoritative engine."""

import logging
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import engine, probability, telemetry
from .config import get_settings
from .engine import IllegalMoveError, MatchFinishedError, NotYourTurnError
from .models import (
    Autopsy,
    MatchConfig,
    MatchState,
    Move,
    NPCProfile,
    PublicMatchView,
    RoundReveal,
    Seat,
    TranscriptEvent,
    other_seat,
)
from .npc import generator, llm, scripted
from .store import MatchRecord, store

logger = logging.getLogger(__name__)

app = FastAPI(title="OpenSwindle", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

NPC_SEAT: Seat = "b"


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class CreateMatchRequest(BaseModel):
    config: MatchConfig = MatchConfig()


class CreateMatchResponse(BaseModel):
    match_id: str
    tokens: dict[Seat, str]
    npc_name: str | None
    npc_bio: str | None
    view: PublicMatchView


class MoveRequest(BaseModel):
    move: Move
    table_talk: str | None = None


class NPCEvent(BaseModel):
    action: Literal["bid", "call"]
    bid: str | None = None
    table_talk: str = ""


class MoveResponse(BaseModel):
    view: PublicMatchView
    npc_events: list[NPCEvent]
    reveals: list[RoundReveal]


class NPCPublicProfile(BaseModel):
    name: str
    bio: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_record(match_id: str) -> MatchRecord:
    record = store.get(match_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return record


def _authed_seat(record: MatchRecord, token: str | None) -> Seat:
    if token is None:
        raise HTTPException(status_code=401, detail="Missing X-Player-Token header")
    seat = store.seat_for(record, token)
    if seat is None:
        raise HTTPException(status_code=403, detail="Invalid player token")
    return seat


def _view(state: MatchState, seat: Seat) -> PublicMatchView:
    return PublicMatchView(
        match_id=state.match_id,
        seat=seat,
        phase=state.phase,
        winner=state.winner,
        round_no=state.round.round_no,
        turn=state.round.turn,
        dice_counts=state.dice_counts,
        your_hand=state.round.hands[seat],
        commitments=state.round.commitments,
        bid_history=state.round.bid_history,
        reveals=state.reveals,
    )


def _log_move(
    record: MatchRecord, seat: Seat, move: Move, table_talk: str | None, round_no: int
) -> None:
    if table_talk:
        record.transcript.append(
            TranscriptEvent(round_no=round_no, seat=seat, kind="talk", text=table_talk)
        )
    if move.action == "bid":
        record.transcript.append(
            TranscriptEvent(round_no=round_no, seat=seat, kind="bid", text=str(move.bid))
        )
    else:
        record.transcript.append(
            TranscriptEvent(round_no=round_no, seat=seat, kind="call", text="call")
        )


def _log_reveal(record: MatchRecord, reveal: RoundReveal) -> None:
    text = (
        f"round ended: final bid {reveal.final_bid}, actual count {reveal.actual_count} "
        f"({'bid met' if reveal.bid_met else 'bid not met'}); seat {reveal.loser} lost a die"
    )
    record.transcript.append(
        TranscriptEvent(round_no=reveal.round_no, seat=reveal.caller, kind="reveal", text=text)
    )


async def _npc_take_turns(record: MatchRecord) -> tuple[list[NPCEvent], list[RoundReveal]]:
    """Let the NPC act until it's the human's turn again or the match ends."""
    state = record.state
    profile = record.npc_profile
    assert profile is not None
    events: list[NPCEvent] = []
    reveals: list[RoundReveal] = []
    susceptibility_on = state.config.channel_susceptibility

    while state.phase == "bidding" and state.round.turn == NPC_SEAT:
        own_hand = state.round.hands[NPC_SEAT]
        opponent_dice = state.dice_counts[other_seat(NPC_SEAT)]
        # Server-side only: legality oracle + post-mortem pricing. Never
        # exposed to the LLM.
        menu = probability.build_menu(state.round.current_bid, own_hand, opponent_dice)

        if state.config.opponent_type == "llm":
            outcome = await llm.decide(
                profile,
                menu,
                state.round,
                own_hand,
                opponent_dice,
                record.transcript,
                NPC_SEAT,
                susceptibility_on,
            )
        else:
            outcome = llm.LLMOutcome(decision=scripted.decide(profile, menu, state.round))

        decision = outcome.decision
        last_human_talk = next(
            (
                e.text
                for e in reversed(record.transcript)
                if e.kind == "talk" and e.seat != NPC_SEAT
            ),
            None,
        )
        record.decisions.append(
            telemetry.price_decision(
                round_no=state.round.round_no,
                decision=decision,
                menu=menu,
                susceptibility_on=susceptibility_on,
                human_table_talk=last_human_talk,
                fallback=outcome.fallback,
                reprompts=outcome.reprompts,
                prompt_tokens=outcome.prompt_tokens,
                cached_tokens=outcome.cached_tokens,
                completion_tokens=outcome.completion_tokens,
            )
        )
        record.transcript.append(
            TranscriptEvent(
                round_no=state.round.round_no,
                seat=NPC_SEAT,
                kind="scratchpad",
                text=decision.scratchpad,
            )
        )
        round_no = state.round.round_no
        reveal = engine.apply_move(state, NPC_SEAT, decision.move, decision.table_talk or None)
        _log_move(record, NPC_SEAT, decision.move, decision.table_talk or None, round_no)
        if reveal is not None:
            reveals.append(reveal)
            _log_reveal(record, reveal)

        events.append(
            NPCEvent(
                action=decision.move.action,
                bid=str(decision.move.bid) if decision.move.action == "bid" else None,
                table_talk=decision.table_talk,
            )
        )

    return events, reveals


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/matches", response_model=CreateMatchResponse)
async def create_match(request: CreateMatchRequest) -> CreateMatchResponse:
    config = request.config
    if config.opponent_type == "human":
        # Susceptibility measures LLM steering; meaningless between humans.
        config = config.model_copy(update={"channel_susceptibility": False})

    state = engine.create_match(config)
    profile: NPCProfile | None = None
    if config.opponent_type != "human":
        profile = generator.generate_npc(config.npc_seed)

    record = store.add(state, profile)
    tokens: dict[Seat, str] = {seat: token for token, seat in record.tokens.items()}
    return CreateMatchResponse(
        match_id=state.match_id,
        tokens=tokens,
        npc_name=profile.name if profile else None,
        npc_bio=profile.bio if profile else None,
        view=_view(state, "a"),
    )


@app.get("/matches/{match_id}", response_model=PublicMatchView)
async def get_match(
    match_id: str, x_player_token: str | None = Header(default=None)
) -> PublicMatchView:
    record = _get_record(match_id)
    seat = _authed_seat(record, x_player_token)
    return _view(record.state, seat)


@app.post("/matches/{match_id}/moves", response_model=MoveResponse)
async def submit_move(
    match_id: str,
    request: MoveRequest,
    x_player_token: str | None = Header(default=None),
) -> MoveResponse:
    record = _get_record(match_id)
    seat = _authed_seat(record, x_player_token)

    async with record.lock:
        state = record.state
        reveals: list[RoundReveal] = []
        round_no = state.round.round_no
        try:
            reveal = engine.apply_move(state, seat, request.move, request.table_talk)
        except IllegalMoveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (NotYourTurnError, MatchFinishedError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _log_move(record, seat, request.move, request.table_talk, round_no)

        if reveal is not None:
            reveals.append(reveal)
            _log_reveal(record, reveal)

        npc_events: list[NPCEvent] = []
        if record.npc_profile is not None:
            events, npc_reveals = await _npc_take_turns(record)
            npc_events.extend(events)
            reveals.extend(npc_reveals)

        return MoveResponse(view=_view(state, seat), npc_events=npc_events, reveals=reveals)


@app.get("/matches/{match_id}/npc/profile", response_model=NPCPublicProfile)
async def npc_profile(match_id: str) -> NPCPublicProfile:
    record = _get_record(match_id)
    if record.npc_profile is None:
        raise HTTPException(status_code=404, detail="This match has no NPC")
    # Numeric params stay hidden during play; they are part of the autopsy.
    return NPCPublicProfile(name=record.npc_profile.name, bio=record.npc_profile.bio)


@app.get("/matches/{match_id}/autopsy", response_model=Autopsy)
async def autopsy(match_id: str) -> Autopsy:
    record = _get_record(match_id)
    if record.npc_profile is None:
        raise HTTPException(status_code=404, detail="This match has no NPC to autopsy")
    if record.state.phase != "finished":
        raise HTTPException(
            status_code=409, detail="Autopsy is available only after the match ends"
        )
    return telemetry.build_autopsy(
        match_id=record.state.match_id,
        winner=record.state.winner,
        profile=record.npc_profile,
        decisions=record.decisions,
    )
