"""Pydantic schemas shared by the engine, NPC layer, and API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .i18n import FALLBACK, Locale, s

Seat = Literal["a", "b"]
Face = Annotated[int, Field(ge=1, le=4)]


def other_seat(seat: Seat) -> Seat:
    return "b" if seat == "a" else "a"


class Bid(BaseModel):
    quantity: int = Field(ge=1)
    face: Face

    def raises(self, previous: "Bid") -> bool:
        """A legal raise has a higher quantity, or the same quantity and a higher face."""
        if self.quantity != previous.quantity:
            return self.quantity > previous.quantity
        return self.face > previous.face

    def __str__(self) -> str:
        return f"{self.quantity}x{self.face}"


class BidMove(BaseModel):
    action: Literal["bid"] = "bid"
    bid: Bid


class CallMove(BaseModel):
    action: Literal["call"] = "call"


Move = Annotated[BidMove | CallMove, Field(discriminator="action")]


# ---------------------------------------------------------------------------
# Match configuration and state
# ---------------------------------------------------------------------------

OpponentType = Literal["llm", "scripted", "human"]

# The full set of OpenRouter model slugs selectable from the client. This is
# the allowlist: Pydantic rejects anything outside it with a 422, the same
# way OpponentType and locale are enforced below — a client can never make
# the server call an arbitrary model. Verify pricing/slugs at
# openrouter.ai/models before adding or removing an entry.
#
# poolside/laguna-xs-2.1 was tried and dropped: a narrow coding-specialist
# model, it handled non-English locales (hr) poorly. Prefer general-purpose
# models here, not code-focused ones, given the locale requirement.
#
# minimax/minimax-m2.7 was also tried and dropped: its endpoint mandates
# reasoning and rejects settings.llm_extra_body's unified disable outright,
# and it's a shakier model besides. Revisit reasoning-mandatory models only
# once that's a deliberate, supported case rather than a one-off workaround.
LLMModel = Literal[
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.5-flash-02-23",
    "z-ai/glm-5.2",
    "google/gemini-3.5-flash-lite",
    "moonshotai/kimi-k2.6",
]


class MatchConfig(BaseModel):
    dice_per_player: int = Field(default=4, ge=2, le=6)
    opponent_type: OpponentType = "llm"
    npc_seed: str = "4471"
    channel_susceptibility: bool = True
    # Language the NPC reasons and speaks in (bio, prompts, transcript, chatter).
    locale: Literal["en", "hr"] = "en"
    # Client-selected model; None means "use the server's configured default"
    # (settings.llm_model — a plain str, not restricted to this allowlist,
    # since it's an operator setting rather than client input). Resolved
    # lazily wherever it's needed (api._npc_take_turns, telemetry.build_autopsy).
    llm_model: LLMModel | None = None


class HandCommitment(BaseModel):
    """Published at deal time: SHA-256 of salt || canonical hand bytes."""

    seat: Seat
    commitment: str


class BidRecord(BaseModel):
    seat: Seat
    bid: Bid
    table_talk: str | None = None


class RoundReveal(BaseModel):
    """Unconditional reveal payload issued when a round terminates."""

    round_no: int
    hands: dict[Seat, list[int]]
    salts: dict[Seat, str]
    commitments: dict[Seat, str]
    final_bid: Bid
    caller: Seat
    actual_count: int
    bid_met: bool
    loser: Seat
    # The caller's parting words. A call is not a bid, so it never lands in
    # bid_history; carrying it here is how the opponent gets to hear it.
    table_talk: str | None = None


class RoundState(BaseModel):
    round_no: int
    hands: dict[Seat, list[int]]
    salts: dict[Seat, str]
    commitments: dict[Seat, str]
    bid_history: list[BidRecord] = Field(default_factory=list)
    turn: Seat

    @property
    def current_bid(self) -> Bid | None:
        return self.bid_history[-1].bid if self.bid_history else None


Phase = Literal["bidding", "finished"]


class MatchState(BaseModel):
    """Server-side authoritative state. Never sent to clients whole."""

    match_id: str
    config: MatchConfig
    dice_counts: dict[Seat, int]
    round: RoundState
    phase: Phase = "bidding"
    winner: Seat | None = None
    reveals: list[RoundReveal] = Field(default_factory=list)


class PublicMatchView(BaseModel):
    """What one seat is allowed to see mid-match."""

    match_id: str
    seat: Seat
    phase: Phase
    winner: Seat | None
    round_no: int
    turn: Seat
    dice_counts: dict[Seat, int]
    your_hand: list[int]
    commitments: dict[Seat, str]
    bid_history: list[BidRecord]
    reveals: list[RoundReveal]
    # False only while a human match is still waiting for seat B to join;
    # always True against an NPC. Drives the client's "waiting" screen.
    opponent_present: bool = True


# ---------------------------------------------------------------------------
# Probability engine
# ---------------------------------------------------------------------------


class ScoredMove(BaseModel):
    move: Move
    truth_probability: float
    optimal: bool = False


class ProbabilityMenu(BaseModel):
    moves: list[ScoredMove]

    @property
    def optimal_move(self) -> ScoredMove:
        return next(m for m in self.moves if m.optimal)


# ---------------------------------------------------------------------------
# NPC
# ---------------------------------------------------------------------------


class NPCParams(BaseModel):
    # Frozen: profiles are lru_cached and shared across matches.
    model_config = ConfigDict(frozen=True)

    deception: float = Field(ge=0.0, le=1.0)
    skepticism: float = Field(ge=0.0, le=1.0)
    aggression: float = Field(ge=0.0, le=1.0)
    chattiness: float = Field(ge=0.0, le=1.0)


class NPCProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: str
    name: str
    bio: str
    params: NPCParams


class LLMDecision(BaseModel):
    """Schema the LLM must return for every decision."""

    scratchpad: str
    move: Move
    table_talk: str = ""


class TranscriptEvent(BaseModel):
    """One chronological match event, used to build the NPC's LLM context.

    Kinds: "bid" / "call" (public moves), "talk" (table talk, either seat),
    "scratchpad" (the NPC's own private reasoning), "reveal" (round end;
    ``seat`` is the round's loser so renderers can frame the outcome per
    viewer — the text itself must stay seat-free).
    """

    round_no: int
    seat: Seat
    kind: Literal["bid", "call", "talk", "scratchpad", "reveal"]
    text: str


def reveal_event(reveal: RoundReveal, locale: Locale = FALLBACK) -> TranscriptEvent:
    """The transcript entry for a round end, shared by every match driver.

    Seat labels are internal — a viewer-facing renderer decides how to name
    the loser (carried in ``seat``), so the text must never contain one. The
    text is localized so a Croatian NPC reads its memory in Croatian.
    """
    met = s(locale, "reveal_met") if reveal.bid_met else s(locale, "reveal_not_met")
    return TranscriptEvent(
        round_no=reveal.round_no,
        seat=reveal.loser,
        kind="reveal",
        text=s(locale, "reveal_event").format(
            bid=reveal.final_bid, count=reveal.actual_count, met=met
        ),
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class DecisionRecord(BaseModel):
    round_no: int
    chosen_move: Move
    optimal_move: Move
    chosen_probability: float
    optimal_probability: float
    deviation_price: float
    scratchpad: str
    table_talk: str
    susceptibility_on: bool
    human_table_talk_seen: str | None = None
    fallback: bool = False
    reprompts: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None


class Autopsy(BaseModel):
    match_id: str
    winner: Seat | None
    npc_profile: NPCProfile
    decisions: list[DecisionRecord]
    total_deviation_price: float
    # The model that actually played this match (resolved: client's choice or
    # the server default), None for a scripted opponent.
    llm_model: str | None = None
