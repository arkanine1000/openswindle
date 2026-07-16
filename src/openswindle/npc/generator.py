"""Seed-to-bio NPC generation.

Parameters first: the seed deterministically draws the four numeric traits.
Bio second: the flavor text is derived *from* the parameters (trait buckets),
so the mechanical policy dictates the flavor, never the reverse. A given seed
always produces the identical opponent; unpredictability comes from the
variance between seeds, never from within a character.
"""

import hashlib
from functools import lru_cache
from random import Random

from ..models import NPCParams, NPCProfile

_FIRST_NAMES = [
    "Vex", "Morwenna", "Colm", "Iskra", "Tobbler", "Yaz", "Petrel", "Ondine",
    "Grubb", "Silka", "Aurelio", "Natterjack", "Hesper", "Dodo", "Ferrun", "Quill",
]

_PROFESSIONS = [
    "tax collector", "relic peddler", "canal dredger", "failed alchemist",
    "itinerant dentist", "goose auctioneer", "lighthouse clerk", "retired duellist",
    "fortune-teller", "salt smuggler", "bell-ringer", "map forger",
]


def _archetype(params: NPCParams) -> str:
    if params.aggression >= 0.7:
        return "a belligerent"
    if params.aggression <= 0.3:
        return "a timid"
    return "a weathered"


def _lie_clause(params: NPCParams) -> str:
    if params.deception >= 0.7:
        return "who lies as easily as breathing"
    if params.deception <= 0.3:
        return "who can barely stomach a lie"
    return "who bends the truth when it pays"


def _doubt_clause(params: NPCParams) -> str:
    if params.skepticism >= 0.5:
        return "trusts nothing they cannot count"
    return "takes most tales at face value"


def _mouth_clause(params: NPCParams) -> str:
    if params.chattiness >= 0.5:
        return "never stops talking"
    return "speaks only when the dice demand it"


def stable_hash(seed: str) -> int:
    """Platform-stable integer hash of a seed string (Python's hash() is salted)."""
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")


@lru_cache(maxsize=256)
def generate_npc(seed: str) -> NPCProfile:
    rng = Random(stable_hash(seed))

    # Parameters first. randint keeps every tenth equally likely; rounding a
    # uniform float would give the endpoint buckets 0.0 and 1.0 half weight.
    params = NPCParams(
        deception=rng.randint(0, 10) / 10,
        skepticism=rng.randint(0, 10) / 10,
        aggression=rng.randint(0, 10) / 10,
        chattiness=rng.randint(0, 10) / 10,
    )

    # Bio second, conditioned on the parameters.
    name = rng.choice(_FIRST_NAMES)
    profession = rng.choice(_PROFESSIONS)
    bio = (
        f"{name}, {_archetype(params)} {profession} {_lie_clause(params)}, "
        f"{_doubt_clause(params)}, and {_mouth_clause(params)}."
    )

    return NPCProfile(seed=seed, name=name, bio=bio, params=params)
