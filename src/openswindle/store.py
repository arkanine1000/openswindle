"""In-memory match store. Nothing survives a restart."""

import asyncio
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from .config import get_settings
from .models import DecisionRecord, MatchState, NPCProfile, Seat, TranscriptEvent


@dataclass
class MatchRecord:
    state: MatchState
    npc_profile: NPCProfile | None
    tokens: dict[str, Seat]  # player token -> seat
    issued_seats: set[Seat]
    finished_at: float | None = None
    decisions: list[DecisionRecord] = field(default_factory=list)
    # Chronological match log (moves, table talk, NPC scratchpads, reveals).
    # Append-only: it doubles as the LLM's memory and keeps the prompt prefix
    # stable for provider-side caching.
    transcript: list[TranscriptEvent] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MatchStore:
    def __init__(
        self,
        finished_ttl_seconds: int = 3600,
        max_finished_matches: int = 1000,
        now: Callable[[], float] = monotonic,
    ) -> None:
        self._matches: dict[str, MatchRecord] = {}
        self._finished_ttl_seconds = finished_ttl_seconds
        self._max_finished_matches = max_finished_matches
        self._now = now

    def add(self, state: MatchState, npc_profile: NPCProfile | None) -> MatchRecord:
        self.prune()
        tokens: dict[str, Seat] = {secrets.token_urlsafe(16): "a"}
        if state.config.opponent_type == "human":
            tokens[secrets.token_urlsafe(16)] = "b"
        record = MatchRecord(
            state=state,
            npc_profile=npc_profile,
            tokens=tokens,
            issued_seats={"a"},
        )
        self._matches[state.match_id] = record
        return record

    def get(self, match_id: str) -> MatchRecord | None:
        self.prune()
        return self._matches.get(match_id)

    def seat_for(self, record: MatchRecord, token: str) -> Seat | None:
        return record.tokens.get(token)

    def issued_tokens(self, record: MatchRecord) -> dict[Seat, str]:
        return {
            seat: token
            for token, seat in record.tokens.items()
            if seat in record.issued_seats
        }

    def issue_token(self, record: MatchRecord, seat: Seat) -> str | None:
        if seat in record.issued_seats:
            return None
        token = next(
            (token for token, token_seat in record.tokens.items() if token_seat == seat),
            None,
        )
        if token is None:
            return None
        record.issued_seats.add(seat)
        return token

    def mark_finished(self, record: MatchRecord) -> None:
        if record.state.phase != "finished" or record.finished_at is not None:
            return
        record.finished_at = self._now()
        self.prune()

    def prune(self) -> None:
        now = self._now()
        expired = [
            match_id
            for match_id, record in self._matches.items()
            if record.finished_at is not None
            and now - record.finished_at >= self._finished_ttl_seconds
        ]
        for match_id in expired:
            del self._matches[match_id]

        finished = sorted(
            (
                (record.finished_at, match_id)
                for match_id, record in self._matches.items()
                if record.finished_at is not None
            ),
            key=lambda item: item[0] or 0,
        )
        overflow = len(finished) - self._max_finished_matches
        for _, match_id in finished[: max(overflow, 0)]:
            del self._matches[match_id]

settings = get_settings()
store = MatchStore(
    finished_ttl_seconds=settings.finished_match_ttl_seconds,
    max_finished_matches=settings.max_finished_matches,
)
