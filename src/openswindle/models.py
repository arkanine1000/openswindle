"""Pydantic schemas shared by the engine, NPC layer, and API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class MatchConfig(BaseModel):
    dice_per_player: int = Field(default=4, ge=2, le=6)
    opponent_type: OpponentType = "llm"
    npc_seed: str = "4471"
    channel_susceptibility: bool = True


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
    "scratchpad" (the NPC's own private reasoning), "reveal" (round end).
    """

    round_no: int
    seat: Seat
    kind: Literal["bid", "call", "talk", "scratchpad", "reveal"]
    text: str


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
