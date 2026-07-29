"""Deterministic parameter-driven NPC policy.

Serves three roles: the "scripted" opponent type, mock mode for the LLM
layer, and the fallback when the LLM exhausts its reprompts. All stochastic
choices are seeded from (npc seed, round number, bid count) so a character
behaves identically in identical situations — variance exists between
characters, never within one.

Scratchpads written here can enter the LLM's transcript via the fallback
path, so they must never contain probability-engine data.
"""

from random import Random

from ..i18n import FALLBACK, Locale
from ..models import (
    CallMove,
    LLMDecision,
    Move,
    NPCParams,
    NPCProfile,
    ProbabilityMenu,
    RoundState,
)
from .generator import stable_hash

# Croatian is a DRAFT pending native review. Lines stay gender-neutral toward
# the opponent (2nd-person present, imperatives) — no he/she assumed.
_BID_CHATTER: dict[Locale, list[str]] = {
    "en": [
        "You blink too much when you bid.",
        "The dice like me tonight.",
        "I have counted worse hands than yours.",
        "My old master lost a finger to a bid like that.",
        "Go on then, surprise me.",
        "Careful. The bones remember greed.",
    ],
    "hr": [
        "Previše trepćeš kad se okladiš.",
        "Kocke me večeras vole.",
        "Ima i gorih ruku od tvoje.",
        "Hajde, iznenadi me.",
        "Oprezno. Kocke pamte pohlepu.",
        "Znam čitati lice poput tvoga.",
    ],
}

_CALL_CHATTER: dict[Locale, list[str]] = {
    "en": [
        "Liar. Show me.",
        "Bold. Foolish, but bold.",
        "No. That story ends here.",
        "Turn them over. Slowly.",
    ],
    "hr": [
        "Lažeš. Pokaži.",
        "Odvažno. Glupo, ali odvažno.",
        "Ne. Tu priča završava.",
        "Okreni ih. Polako.",
    ],
}


def _turn_rng(profile: NPCProfile, round_state: RoundState) -> Random:
    key = f"{profile.seed}:{round_state.round_no}:{len(round_state.bid_history)}"
    return Random(stable_hash(key))


def decide(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    locale: Locale = FALLBACK,
) -> LLMDecision:
    rng = _turn_rng(profile, round_state)
    p = profile.params
    move = _choose_move(p, menu, rng)

    chatter = _CALL_CHATTER if move.action == "call" else _BID_CHATTER
    pool = chatter.get(locale, chatter[FALLBACK])
    table_talk = rng.choice(pool) if rng.random() < p.chattiness else ""
    scratch = (
        f"[scripted] instincts: deception={p.deception} "
        f"skepticism={p.skepticism} aggression={p.aggression}"
    )
    return LLMDecision(scratchpad=scratch, move=move, table_talk=table_talk)


def _choose_move(p: NPCParams, menu: ProbabilityMenu, rng: Random) -> Move:
    bids = sorted(
        (m for m in menu.moves if m.move.action == "bid"),
        key=lambda m: -m.truth_probability,
    )
    call = next((m for m in menu.moves if m.move.action == "call"), None)

    # Call decision: skepticism lowers the confidence bar for calling out a lie.
    if call is not None:
        if not bids:
            return CallMove()  # No legal raise remains; forced call.
        threshold = 0.9 - 0.5 * p.skepticism
        if call.truth_probability >= threshold:
            return CallMove()

    # Bluff roll: deception sets how often a low-truth claim is made, and
    # aggression sets how outrageous that claim gets.
    if rng.random() < p.deception:
        bluffs = [m for m in bids if m.truth_probability < 0.5]
        if bluffs:
            target = 0.35 - 0.3 * p.aggression
            return min(bluffs, key=lambda m: abs(m.truth_probability - target)).move

    # Honest raise: aggression trades safety for velocity by targeting a
    # lower confidence level (0.95 timid -> 0.50 reckless).
    target = 0.95 - 0.45 * p.aggression
    return min(bids, key=lambda m: abs(m.truth_probability - target)).move
