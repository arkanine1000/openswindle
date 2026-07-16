"""Seed-to-bio NPC generation.

Parameters first: the seed deterministically draws the four numeric traits.
Bio second: the flavor text is derived *from* the parameters (archetype
buckets), so the mechanical policy dictates the flavor, never the reverse.
Each profile also plants 1-2 persistent, mechanically enforced tells chosen
by the same RNG, so a given seed always produces the identical opponent.
"""

import hashlib
from functools import lru_cache
from random import Random

from ..models import NPCParams, NPCProfile, Tell

# Catalog of plantable tells. Each tell_id has an enforcement hook in the
# scripted policy and is stated as a hard behavioral rule in the LLM prompt.
TELL_CATALOG: dict[str, str] = {
    "opens_low": "Always opens a round one quantity below what the maths supports.",
    "never_bluff_face4": (
        "Never bluffs on face 4; a face-4 bid from them is always covered by their hand."
    ),
    "pair_conservative": "Always bids conservatively (the safest legal raise) when holding a pair.",
    "call_shy_early": "Never calls within the first two bids of a round.",
    "escalates_when_quiet": (
        "Raises the quantity by two whenever the opponent said nothing last turn."
    ),
}

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
    if params.deception >= 0.5 and params.aggression >= 0.5:
        return "an unhinged"
    if params.deception >= 0.5:
        return "a smiling"
    if params.aggression >= 0.5:
        return "a belligerent"
    return "a timid"


def _temperament(params: NPCParams) -> str:
    doubt = (
        "who trusts no claim they cannot count themselves"
        if params.skepticism >= 0.5
        else "who takes most tales at face value"
    )
    mouth = (
        "and never stops talking"
        if params.chattiness >= 0.5
        else "and speaks only when the dice demand it"
    )
    return f"{doubt} {mouth}"


def stable_hash(seed: str) -> int:
    """Platform-stable integer hash of a seed string (Python's hash() is salted)."""
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")


@lru_cache(maxsize=256)
def generate_npc(seed: str) -> NPCProfile:
    rng = Random(stable_hash(seed))

    # Parameters first.
    params = NPCParams(
        deception=round(rng.random(), 3),
        skepticism=round(rng.random(), 3),
        aggression=round(rng.random(), 3),
        chattiness=round(rng.random(), 3),
    )

    # Planted tells: 1-2 persistent behaviors.
    tell_ids = rng.sample(sorted(TELL_CATALOG), k=rng.choice([1, 2]))
    tells = [Tell(tell_id=t, description=TELL_CATALOG[t]) for t in tell_ids]

    # Bio second, conditioned on the parameters.
    name = rng.choice(_FIRST_NAMES)
    profession = rng.choice(_PROFESSIONS)
    bio = f"{name}, {_archetype(params)} {profession} {_temperament(params)}."

    return NPCProfile(seed=seed, name=name, bio=bio, params=params, tells=tells)
