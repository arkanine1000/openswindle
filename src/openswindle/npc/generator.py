"""Seed-to-bio NPC generation.

Parameters first: the seed deterministically draws the four numeric traits.
Bio second: the flavor text is derived *from* the parameters (trait buckets),
so the mechanical policy dictates the flavor, never the reverse. A given seed
always produces the identical opponent (same name, same profession, same trait
buckets) in every locale — only the language of the bio changes.

NPCs are gender-neutral: English uses they/them; Croatian avoids person-gender
entirely by describing the character in gender-free present-tense verb phrases,
so no invented name ever has to be assigned a grammatical gender.
"""

import hashlib
from functools import lru_cache
from random import Random

from ..i18n import FALLBACK, Locale
from ..models import NPCParams, NPCProfile

_FIRST_NAMES = [
    "Vex", "Morwenna", "Colm", "Iskra", "Tobbler", "Yaz", "Petrel", "Ondine",
    "Grubb", "Silka", "Aurelio", "Natterjack", "Hesper", "Dodo", "Ferrun", "Quill",
]

# Parallel per-locale profession lists — same index = the same character.
_PROFESSIONS: dict[Locale, list[str]] = {
    "en": [
        "tax collector", "relic peddler", "canal dredger", "failed alchemist",
        "itinerant dentist", "goose auctioneer", "lighthouse clerk", "retired duellist",
        "fortune-teller", "salt smuggler", "bell-ringer", "map forger",
    ],
    "hr": [
        "poreznik", "prodavač relikvija", "kopač kanala", "propali alkemičar",
        "putujući zubar", "prodavač gusaka", "čuvar svjetionika", "umirovljeni dvobojac",
        "gatara", "krijumčar soli", "zvonar", "krivotvoritelj karata",
    ],
}

# Trait-bucket phrases per locale. English keeps its exact original wording;
# Croatian uses gender-free present-tense verb phrases (no adjective agreement).
_ARCHETYPE: dict[Locale, dict[str, str]] = {
    "en": {"belligerent": "a belligerent", "timid": "a timid", "weathered": "a weathered"},
    "hr": {
        "belligerent": "voli svađu za stolom",
        "timid": "igra oprezno",
        "weathered": "malo toga još može iznenaditi",
    },
}
_LIE: dict[Locale, dict[str, str]] = {
    "en": {
        "easy": "who lies as easily as breathing",
        "barely": "who can barely stomach a lie",
        "bends": "who bends the truth when it pays",
    },
    "hr": {
        "easy": "laže lako kao što diše",
        "barely": "jedva podnosi laž",
        "bends": "iskrivljuje istinu kad se isplati",
    },
}
_DOUBT: dict[Locale, dict[str, str]] = {
    "en": {"counts": "trusts nothing they cannot count", "faith": "takes most tales at face value"},
    "hr": {
        "counts": "ne vjeruje ničemu što ne može izbrojati",
        "faith": "vjeruje gotovo svakoj priči",
    },
}
_MOUTH: dict[Locale, dict[str, str]] = {
    "en": {"talks": "never stops talking", "quiet": "speaks only when the dice demand it"},
    "hr": {"talks": "nikad ne prestaje govoriti", "quiet": "govori tek kad kocke to zahtijevaju"},
}


def _archetype_key(params: NPCParams) -> str:
    if params.aggression >= 0.7:
        return "belligerent"
    if params.aggression <= 0.3:
        return "timid"
    return "weathered"


def _lie_key(params: NPCParams) -> str:
    if params.deception >= 0.7:
        return "easy"
    if params.deception <= 0.3:
        return "barely"
    return "bends"


def _doubt_key(params: NPCParams) -> str:
    return "counts" if params.skepticism >= 0.5 else "faith"


def _mouth_key(params: NPCParams) -> str:
    return "talks" if params.chattiness >= 0.5 else "quiet"


def _compose_bio(locale: Locale, name: str, prof: str, params: NPCParams) -> str:
    arch = _ARCHETYPE[locale][_archetype_key(params)]
    lie = _LIE[locale][_lie_key(params)]
    doubt = _DOUBT[locale][_doubt_key(params)]
    mouth = _MOUTH[locale][_mouth_key(params)]
    # NOTE (localization): the ONE hardcoded locale branch — Croatian
    # restructures the bio (gender-free verb-phrase list) vs English (adjective
    # phrase). Fine for two languages; refactor to a per-locale assembler
    # (dict[Locale, Callable]) before adding a third.
    if locale == "hr":
        # Gender-free: name, role noun, then present-tense verb phrases.
        arch = arch[:1].upper() + arch[1:]
        return f"{name}, {prof}. {arch}, {lie}, {doubt} i {mouth}."
    return f"{name}, {arch} {prof} {lie}, {doubt}, and {mouth}."


def stable_hash(seed: str) -> int:
    """Platform-stable integer hash of a seed string (Python's hash() is salted)."""
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")


@lru_cache(maxsize=256)
def generate_npc(seed: str, locale: Locale = FALLBACK) -> NPCProfile:
    if locale not in _PROFESSIONS:
        locale = FALLBACK
    rng = Random(stable_hash(seed))

    # Parameters first. randint keeps every tenth equally likely; rounding a
    # uniform float would give the endpoint buckets 0.0 and 1.0 half weight.
    params = NPCParams(
        deception=rng.randint(0, 10) / 10,
        skepticism=rng.randint(0, 10) / 10,
        aggression=rng.randint(0, 10) / 10,
        chattiness=rng.randint(0, 10) / 10,
    )

    # Bio second, conditioned on the parameters. Draw name + profession by index
    # so the same seed is the same character in every locale.
    name = rng.choice(_FIRST_NAMES)
    prof_index = rng.randrange(len(_PROFESSIONS[FALLBACK]))
    profession = _PROFESSIONS[locale][prof_index]
    bio = _compose_bio(locale, name, profession, params)

    return NPCProfile(seed=seed, name=name, bio=bio, params=params)
