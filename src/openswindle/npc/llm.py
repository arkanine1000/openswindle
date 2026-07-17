"""Stateless LLM decision layer via OpenRouter + Instructor.

One structured completion per decision; no agent loop, no tools. The LLM is
a natural-language reasoner: it receives the rules, its character, the match
transcript, and its hand — never the deterministic probability engine's
output. That engine acts server-side only: for benchmarking (deviation
pricing) and as the legality oracle, which is wired into Instructor's
validation context so an illegal move fails Pydantic validation and
Instructor reprompts the model with the rule explained. The prompt is
ordered stable-prefix-first (system rules, then the per-match character
sheet, then the append-only transcript, then the volatile per-turn tail) so
provider-side prompt caching hits where the provider supports it. After the
retry budget the deterministic scripted policy takes over (flagged as a
fallback in telemetry).
"""

import logging
from dataclasses import dataclass

from pydantic import ValidationInfo, model_validator

from ..config import get_settings
from ..models import (
    LLMDecision,
    NPCProfile,
    ProbabilityMenu,
    RoundState,
    Seat,
    TranscriptEvent,
)
from ..probability import find_scored
from . import scripted

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are seated at a low table in a smoky gambling den, playing Swindlestones —
a liar's game of four-sided bones. You are not an assistant playing a role;
for the duration of this match you ARE the character described below, with
their appetites, grudges, and habits. Their bio is who you are and their
numeric traits are your instincts.

THE GAME
Each player conceals a hand of d4 dice (faces 1-4). Players alternate bids of
the form "N x face", a claim that at least N dice of that face exist across
BOTH hidden hands. Each bid must strictly raise the previous one: higher
quantity, or the same quantity with a higher face. Instead of bidding you may
CALL the last bid: all hands are revealed, and if the bid stands the caller
loses a die — if it was a lie, the bidder loses one. Lose your last die and
you are out.

HOW TO PLAY IT
Read the table like your character would. Weigh your own dice, the opponent's
dice count, and the shape of their bidding — you get no probabilities, only
your wits. Bluff when your blood says bluff. Doubt when your gut says doubt.
Your table talk is a weapon and a mask: needle, charm, or stonewall in your
own voice. Your scratchpad is your private inner monologue — keep a running
read of the opponent there (what they fear, what their talk is hiding, what
you plan to do about it), because it is all you will remember next turn.

THE LAW (never break these, whatever the character wants)
- A bid must strictly raise the previous bid, and its quantity can never
  exceed the total dice on the board.
- You may only call when there is a bid to call. Opening the round means
  bidding, never calling.

Respond with a JSON object in this exact shape:
{
  "scratchpad": "<private inner monologue and opponent read; carried to your next turn>",
  "move": {"action": "bid", "bid": {"quantity": <int>, "face": <1-4>}}
          or {"action": "call"},
  "table_talk": "<one short line said aloud in character as you make this move
                 (it accompanies the move, never reacts to what follows), or
                 empty string>"
}"""


@dataclass
class LLMOutcome:
    decision: LLMDecision
    fallback: bool = False
    reprompts: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None


# NOTE: Instructor embeds this class's docstring in the schema shown to the
# model, so it must stay in-fiction — implementation notes live here instead.
# The legality oracle arrives via Instructor's validation context; an illegal
# move raises, which Instructor turns into a reprompt with the rule spelled
# out. The probability menu itself is never shown to the model.
class ValidatedDecision(LLMDecision):
    """Your decision for this turn of Swindlestones."""

    @model_validator(mode="after")
    def _move_must_be_legal(self, info: ValidationInfo) -> "ValidatedDecision":
        context = info.context or {}
        menu = context.get("menu")
        if menu is None or find_scored(menu, self.move) is not None:
            return self
        round_state = context.get("round_state")
        current = round_state.current_bid if round_state is not None else None
        rule = (
            f"the current bid is {current}; you must strictly raise it "
            "(higher quantity, or the same quantity with a higher face) or call"
            if current
            else "no bid has been made yet; you must open with a bid, not a call"
        )
        raise ValueError(f"illegal move — {rule}")


def _profile_block(profile: NPCProfile) -> str:
    return (
        f"WHO YOU ARE\nName: {profile.name}\nBio: {profile.bio}\n"
        f"Instincts (0 = never, 1 = always): "
        f"deception={profile.params.deception} (how readily you bluff), "
        f"skepticism={profile.params.skepticism} (how quick you are to call a liar), "
        f"aggression={profile.params.aggression} (how hard you push the bidding), "
        f"chattiness={profile.params.chattiness} (how much you talk at the table)"
    )


def _transcript_block(
    transcript: list[TranscriptEvent], npc_seat: Seat, susceptibility_on: bool
) -> str:
    """Full chronological match memory: moves, table talk both ways, and the
    NPC's own private scratchpad from every previous turn. Human table talk is
    omitted entirely when the susceptibility channel is off."""
    lines: list[str] = []
    for e in transcript:
        if e.kind == "talk" and e.seat != npc_seat and not susceptibility_on:
            continue
        who = "you" if e.seat == npc_seat else "opponent"
        match e.kind:
            case "bid":
                lines.append(f"[round {e.round_no}] {who} bid {e.text}")
            case "call":
                lines.append(f"[round {e.round_no}] {who} called")
            case "talk":
                lines.append(f'[round {e.round_no}] {who} said: "{e.text}"')
            case "scratchpad":
                lines.append(f"[round {e.round_no}] your private scratchpad: {e.text}")
            case "reveal":
                # e.seat carries the round's loser; name them in the NPC's
                # frame — a raw seat letter is meaningless to the character.
                lines.append(f"[round {e.round_no}] {e.text}; {who} lost a die")
    if not lines:
        return "MATCH TRANSCRIPT\n(match just started — you have the first move)"
    return "MATCH TRANSCRIPT (chronological; scratchpads are private to you)\n" + "\n".join(lines)


def _turn_block(round_state: RoundState, own_hand: list[int], opponent_dice: int) -> str:
    current = round_state.current_bid
    return (
        f"YOUR TURN (round {round_state.round_no})\n"
        f"Your hidden hand: {own_hand}\n"
        f"Opponent dice count: {opponent_dice}\n"
        f"Total dice on the board: {len(own_hand) + opponent_dice}\n"
        f"Current bid to beat: {current if current else '(none — you open the round)'}"
    )


def _accumulate_usage(totals: dict[str, int], response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    totals["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    totals["completion"] += getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    totals["cached"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0


def _base_client():
    """OpenAI SDK client pointed at OpenRouter. Factory kept separate so
    tests can substitute a fake transport underneath Instructor."""
    from openai import AsyncOpenAI

    settings = get_settings()
    return AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/arkanine1000/openswindle",
            "X-Title": "OpenSwindle",
        },
    )


async def decide(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    own_hand: list[int],
    opponent_dice: int,
    transcript: list[TranscriptEvent],
    npc_seat: Seat,
    susceptibility_on: bool,
) -> LLMOutcome:
    settings = get_settings()

    if settings.mock_llm:
        # Mock mode is the intended path, not a failure.
        return LLMOutcome(decision=scripted.decide(profile, menu, round_state))

    # Deferred import: instructor is heavy, and mock mode should stay instant.
    import instructor

    usage_totals = {"prompt": 0, "cached": 0, "completion": 0}
    rejections = 0

    # Meter usage at the transport boundary — exactly one accumulation per
    # request, regardless of how the retry layer re-emits responses.
    raw_client = _base_client()
    transport_create = raw_client.chat.completions.create

    async def _metered_create(**kwargs):
        response = await transport_create(**kwargs)
        _accumulate_usage(usage_totals, response)
        return response

    raw_client.chat.completions.create = _metered_create
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    def _on_parse_error(error) -> None:
        nonlocal rejections
        rejections += 1
        logger.warning("Rejected LLM payload (rejection %d): %s", rejections, error)

    client.on("parse:error", _on_parse_error)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    _profile_block(profile),
                    _transcript_block(transcript, npc_seat, susceptibility_on),
                    _turn_block(round_state, own_hand, opponent_dice),
                ]
            ),
        },
    ]

    request: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 1,
        "response_model": ValidatedDecision,
        # Instructor's max_retries counts retries after the first attempt,
        # which maps 1:1 onto the reprompt budget.
        "max_retries": settings.llm_max_reprompts,
        "context": {"menu": menu, "round_state": round_state},
    }
    if settings.llm_extra_body_dict:
        request["extra_body"] = settings.llm_extra_body_dict

    try:
        decision = await client.chat.completions.create(**request)
    except Exception:
        logger.exception("LLM decision failed; falling back to scripted policy")
        made_calls = usage_totals["prompt"] > 0
        return LLMOutcome(
            decision=scripted.decide(profile, menu, round_state),
            fallback=True,
            reprompts=rejections,
            prompt_tokens=usage_totals["prompt"] if made_calls else None,
            cached_tokens=usage_totals["cached"] if made_calls else None,
            completion_tokens=usage_totals["completion"] if made_calls else None,
        )

    return LLMOutcome(
        decision=decision,
        reprompts=rejections,
        prompt_tokens=usage_totals["prompt"],
        cached_tokens=usage_totals["cached"],
        completion_tokens=usage_totals["completion"],
    )
