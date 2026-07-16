"""Long-horizon simulations: engine invariants and parameter-driven behavior.

The human seat plays random legal moves; the NPC plays its scripted policy.
Every simulation is fully seeded, so these tests are deterministic despite
exercising thousands of decisions.
"""

import random
from collections import Counter

from openswindle import engine, fairness, probability
from openswindle.models import MatchConfig
from openswindle.npc import generator, scripted


def play_match(match_seed: int, npc_seed: str, dice: int, on_npc_decision=None):
    """Drive a full match, asserting engine invariants throughout."""
    state = engine.create_match(
        MatchConfig(dice_per_player=dice, opponent_type="scripted", npc_seed=npc_seed)
    )
    profile = generator.generate_npc(npc_seed)
    rnd = random.Random(match_seed)
    moves = 0

    while state.phase == "bidding":
        moves += 1
        assert moves < 5000, "match failed to terminate"
        seat = state.round.turn
        opponent = "b" if seat == "a" else "a"
        own = state.round.hands[seat]
        menu = probability.build_menu(
            state.round.current_bid, own, state.dice_counts[opponent]
        )
        total_before = sum(state.dice_counts.values())
        round_before = state.round.round_no

        if seat == "a":
            move = rnd.choice(menu.moves).move
            talk = rnd.choice(["you're bluffing", "", ""]) or None
            reveal = engine.apply_move(state, "a", move, talk)
        else:
            decision = scripted.decide(profile, menu, state.round)
            assert probability.find_scored(menu, decision.move) is not None, (
                f"scripted policy chose an illegal move: {decision.move}"
            )
            assert scripted.decide(profile, menu, state.round) == decision, (
                "scripted policy is not deterministic"
            )
            if on_npc_decision is not None:
                on_npc_decision(state, menu, decision)
            reveal = engine.apply_move(state, "b", decision.move)

        if reveal is not None:
            assert sum(state.dice_counts.values()) == total_before - 1
            for s in ("a", "b"):
                assert fairness.verify_commitment(
                    reveal.salts[s], reveal.hands[s], reveal.commitments[s]
                )
            if state.phase == "bidding":
                assert state.round.round_no == round_before + 1
                assert state.round.turn == reveal.loser

    assert state.winner in ("a", "b")
    assert 0 in state.dice_counts.values()
    return state


def test_engine_invariants_over_many_matches():
    for i in range(24):
        play_match(match_seed=i, npc_seed=f"seed {i}", dice=2 + i % 5)


def _seeds_where(predicate, count: int, universe: int = 400) -> list[str]:
    seeds = [
        f"seed {i}"
        for i in range(universe)
        if predicate(generator.generate_npc(f"seed {i}").params)
    ]
    assert len(seeds) >= count, "predicate too rare in seed universe"
    return seeds[:count]


def _behavior_rates(npc_seed: str, matches: int = 5) -> tuple[float, float]:
    """(call rate when calling is possible, bluff rate when bluffs are available)."""
    calls = call_ops = bluffs = bluff_ops = 0

    def tally(state, menu, decision):
        nonlocal calls, call_ops, bluffs, bluff_ops
        if any(m.move.action == "call" for m in menu.moves):
            call_ops += 1
            if decision.move.action == "call":
                calls += 1
        if decision.move.action == "bid":
            bid_moves = [m for m in menu.moves if m.move.action == "bid"]
            if any(m.truth_probability < 0.5 for m in bid_moves):
                bluff_ops += 1
                chosen = probability.find_scored(menu, decision.move)
                if chosen.truth_probability < 0.5:
                    bluffs += 1

    for match_seed in range(matches):
        play_match(match_seed, npc_seed, dice=4, on_npc_decision=tally)
    return calls / max(call_ops, 1), bluffs / max(bluff_ops, 1)


def test_skepticism_drives_call_rate():
    skeptics = _seeds_where(lambda p: p.skepticism >= 0.8, 4)
    trusting = _seeds_where(lambda p: p.skepticism <= 0.2, 4)
    skeptic_rate = sum(_behavior_rates(s)[0] for s in skeptics) / len(skeptics)
    trusting_rate = sum(_behavior_rates(s)[0] for s in trusting) / len(trusting)
    assert skeptic_rate > trusting_rate, (
        f"skeptics called {skeptic_rate:.2f} vs trusting {trusting_rate:.2f}"
    )


def test_deception_drives_bluff_rate():
    liars = _seeds_where(lambda p: p.deception >= 0.8, 4)
    honest = _seeds_where(lambda p: p.deception <= 0.2, 4)
    liar_rate = sum(_behavior_rates(s)[1] for s in liars) / len(liars)
    honest_rate = sum(_behavior_rates(s)[1] for s in honest) / len(honest)
    assert liar_rate > honest_rate, (
        f"liars bluffed {liar_rate:.2f} vs honest {honest_rate:.2f}"
    )


def test_deal_distribution_is_uniform():
    """Mutual-entropy dealing produces uniform d4 faces (SHA-256 mod 4 is unbiased)."""
    counts: Counter[int] = Counter()
    n_hands = 2000
    for i in range(n_hands):
        salt_a = i.to_bytes(4, "big") * 8
        salt_b = (i * 7 + 3).to_bytes(4, "big") * 8
        counts.update(fairness.deal_hand(salt_a, salt_b, i % 11, "a", 6))
    total = n_hands * 6
    for face in (1, 2, 3, 4):
        share = counts[face] / total
        assert 0.235 < share < 0.265, f"face {face} share {share:.4f} off uniform"
