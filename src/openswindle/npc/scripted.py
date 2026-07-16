"""Deterministic parameter-driven NPC policy.

Serves three roles: the "scripted" opponent type, mock mode for the LLM
layer, and the fallback when the LLM exhausts its reprompts. All stochastic
choices are seeded from (npc seed, round number, bid count) so a character
behaves identically in identical situations — variance exists between
characters, never within one.
"""

from random import Random

from ..models import (
    Bid,
    CallMove,
    LLMDecision,
    NPCProfile,
    ProbabilityMenu,
    RoundState,
    ScoredMove,
)
from .generator import stable_hash

_CHATTER = [
    "You blink too much when you bid.",
    "The dice like me tonight.",
    "I have counted worse hands than yours.",
    "Bold. Foolish, but bold.",
    "My old master lost a finger to a bid like that.",
    "Go on then, surprise me.",
]


def _turn_rng(profile: NPCProfile, round_state: RoundState) -> Random:
    key = f"{profile.seed}:{round_state.round_no}:{len(round_state.bid_history)}"
    return Random(stable_hash(key))


def _has_tell(profile: NPCProfile, tell_id: str) -> bool:
    return any(t.tell_id == tell_id for t in profile.tells)


def _bids(menu: ProbabilityMenu) -> list[ScoredMove]:
    return [m for m in menu.moves if m.move.action == "bid"]


def _call(menu: ProbabilityMenu) -> ScoredMove | None:
    return next((m for m in menu.moves if m.move.action == "call"), None)


def decide(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    own_hand: list[int],
    opponent_dice: int,
) -> LLMDecision:
    rng = _turn_rng(profile, round_state)
    p = profile.params
    move = _choose_move(profile, menu, round_state, own_hand, opponent_dice, rng)

    table_talk = rng.choice(_CHATTER) if rng.random() < p.chattiness else ""
    scratch = f"[scripted] skepticism={p.skepticism} aggression={p.aggression}"
    return LLMDecision(scratchpad=scratch, move=move, table_talk=table_talk)


def _choose_move(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    own_hand: list[int],
    opponent_dice: int,
    rng: Random,
):
    p = profile.params
    bids = sorted(_bids(menu), key=lambda m: -m.truth_probability)
    call = _call(menu)

    # Call decision: skepticism lowers the confidence bar for calling.
    if call is not None:
        call_shy = _has_tell(profile, "call_shy_early") and len(round_state.bid_history) <= 2
        threshold = 0.9 - 0.5 * p.skepticism
        if not call_shy and call.truth_probability >= threshold:
            return CallMove()
        if not bids:
            return CallMove()  # No legal raise remains; forced call.

    # Tell: opening bid pegged one quantity below what the maths supports.
    current_bid = round_state.current_bid
    if current_bid is None and _has_tell(profile, "opens_low"):
        face = max(set(own_hand), key=own_hand.count)
        supported = own_hand.count(face) + opponent_dice // 4
        quantity = max(1, supported - 1)
        target = Bid(quantity=quantity, face=face)
        for m in bids:
            if m.move.bid == target:  # type: ignore[union-attr]
                return m.move

    # Tell: pairs force the safest legal raise.
    holds_pair = any(own_hand.count(f) >= 2 for f in set(own_hand))
    if holds_pair and _has_tell(profile, "pair_conservative"):
        return bids[0].move

    # Bluff roll: deception sets how often a low-truth bid is chosen.
    if rng.random() < p.deception:
        bluffs = [m for m in bids if m.truth_probability < 0.5]
        if _has_tell(profile, "never_bluff_face4"):
            bluffs = [m for m in bluffs if m.move.bid.face != 4]  # type: ignore[union-attr]
        if bluffs:
            # Aggression pushes the bluff further down the plausibility list.
            idx = min(int(p.aggression * 3), len(bluffs) - 1)
            return bluffs[idx].move

    # Honest raise: aggression trades a little safety for velocity.
    idx = min(int(p.aggression * 2), len(bids) - 1)
    choice = bids[idx]

    # Tell: raise quantity by two after a silent opponent turn.
    if _has_tell(profile, "escalates_when_quiet") and round_state.bid_history:
        last = round_state.bid_history[-1]
        if not last.table_talk and current_bid is not None:
            escalated = Bid(quantity=current_bid.quantity + 2, face=current_bid.face)
            for m in bids:
                if m.move.bid == escalated:  # type: ignore[union-attr]
                    return m.move

    return choice.move
