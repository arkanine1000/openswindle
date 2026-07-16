from itertools import product

from openswindle import probability
from openswindle.models import Bid, CallMove


def brute_force_tail(n: int, k: int) -> float:
    """P(at least k of n d4 dice show a fixed face), by full enumeration."""
    hits = sum(
        1 for roll in product(range(1, 5), repeat=n) if roll.count(1) >= k
    )
    return hits / 4**n


def test_binomial_tail_matches_enumeration():
    for n in range(0, 7):
        for k in range(0, n + 2):
            assert abs(probability.binomial_tail(n, k) - brute_force_tail(n, k)) < 1e-12


def test_bid_truth_probability_uses_own_hand():
    # Holding 2 threes, bidding 2x3 is certain.
    assert probability.bid_truth_probability(Bid(quantity=2, face=3), [3, 3, 1], 4) == 1.0
    # Needing more than all opponent dice is impossible.
    assert probability.bid_truth_probability(Bid(quantity=6, face=2), [1, 1, 1], 2) == 0.0


def test_legal_raises_strictness():
    current = Bid(quantity=2, face=3)
    raises = probability.legal_raises(current, total_dice=4)
    assert Bid(quantity=2, face=4) in raises
    assert Bid(quantity=3, face=1) in raises
    assert Bid(quantity=2, face=3) not in raises
    assert Bid(quantity=2, face=2) not in raises
    assert Bid(quantity=1, face=4) not in raises
    assert all(b.quantity <= 4 for b in raises)


def test_menu_opening_has_no_call():
    menu = probability.build_menu(None, [1, 2, 3, 4], 4)
    assert all(m.move.action == "bid" for m in menu.moves)
    assert sum(1 for m in menu.moves if m.optimal) == 1


def test_menu_call_probability_complements_bid():
    current = Bid(quantity=3, face=2)
    own = [2, 1, 4]
    menu = probability.build_menu(current, own, 3)
    call = next(m for m in menu.moves if m.move.action == "call")
    assert abs(
        call.truth_probability
        + probability.bid_truth_probability(current, own, 3)
        - 1.0
    ) < 1e-12


def test_find_scored_detects_illegal():
    menu = probability.build_menu(Bid(quantity=2, face=2), [1, 1], 2)
    assert probability.find_scored(menu, CallMove()) is not None
    from openswindle.models import BidMove

    assert probability.find_scored(menu, BidMove(bid=Bid(quantity=1, face=1))) is None
