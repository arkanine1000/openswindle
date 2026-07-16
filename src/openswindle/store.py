"""In-memory match store. Active matches only; nothing survives a restart."""

import asyncio
import secrets
from dataclasses import dataclass, field

from .models import DecisionRecord, MatchState, NPCProfile, Seat, TranscriptEvent


@dataclass
class MatchRecord:
    state: MatchState
    npc_profile: NPCProfile | None
    tokens: dict[str, Seat]  # player token -> seat
    decisions: list[DecisionRecord] = field(default_factory=list)
    # Chronological match log (moves, table talk, NPC scratchpads, reveals).
    # Append-only: it doubles as the LLM's memory and keeps the prompt prefix
    # stable for provider-side caching.
    transcript: list[TranscriptEvent] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MatchStore:
    def __init__(self) -> None:
        self._matches: dict[str, MatchRecord] = {}

    def add(self, state: MatchState, npc_profile: NPCProfile | None) -> MatchRecord:
        tokens: dict[str, Seat] = {secrets.token_urlsafe(16): "a"}
        if state.config.opponent_type == "human":
            tokens[secrets.token_urlsafe(16)] = "b"
        record = MatchRecord(state=state, npc_profile=npc_profile, tokens=tokens)
        self._matches[state.match_id] = record
        return record

    def get(self, match_id: str) -> MatchRecord | None:
        return self._matches.get(match_id)

    def seat_for(self, record: MatchRecord, token: str) -> Seat | None:
        return record.tokens.get(token)


store = MatchStore()
