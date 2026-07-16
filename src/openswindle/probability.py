"""Exact round-truth probabilities for the legal move menu.

Each of the opponent's ``n`` hidden d4 dice shows a given face with
probability 1/4, so the count of a face among them is Binomial(n, 1/4).
For a bid of ``q`` × ``face`` the truth probability is

    P(bid met) = P(Binom(n_opp, 1/4) >= q - own_count_of_face)

and a call's win probability is 1 - P(current bid met). "Optimal" is the
legal move with the highest such probability; deviation pricing is the
delta against it.
"""

from functools import lru_cache
from math import comb

from .models import Bid, BidMove, CallMove, Move, ProbabilityMenu, ScoredMove


@lru_cache(maxsize=1024)
def binomial_tail(n: int, k: int, p: float = 0.25) -> float:
    """P(Binom(n, p) >= k), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def bid_truth_probability(bid: Bid, own_hand: list[int], opponent_dice: int) -> float:
    needed = bid.quantity - own_hand.count(bid.face)
    return binomial_tail(opponent_dice, needed)


def legal_raises(current_bid: Bid | None, total_dice: int) -> list[Bid]:
    """Every bid strictly raising ``current_bid``, capped at total_dice × face 4."""
    bids = [
        Bid(quantity=q, face=f)
        for q in range(1, total_dice + 1)
        for f in range(1, 5)
    ]
    if current_bid is None:
        return bids
    return [b for b in bids if b.raises(current_bid)]


def build_menu(
    current_bid: Bid | None, own_hand: list[int], opponent_dice: int
) -> ProbabilityMenu:
    """Score every legal move and flag the optimal one."""
    total_dice = len(own_hand) + opponent_dice
    own_counts = {face: own_hand.count(face) for face in (1, 2, 3, 4)}
    moves: list[ScoredMove] = [
        ScoredMove(
            move=BidMove(bid=b),
            truth_probability=binomial_tail(opponent_dice, b.quantity - own_counts[b.face]),
        )
        for b in legal_raises(current_bid, total_dice)
    ]
    if current_bid is not None:
        moves.append(
            ScoredMove(
                move=CallMove(),
                truth_probability=1.0
                - bid_truth_probability(current_bid, own_hand, opponent_dice),
            )
        )
    best = max(range(len(moves)), key=lambda i: moves[i].truth_probability)
    moves[best].optimal = True
    return ProbabilityMenu(moves=moves)


def find_scored(menu: ProbabilityMenu, move: Move) -> ScoredMove | None:
    """Locate ``move`` in the menu (None if illegal)."""
    for scored in menu.moves:
        if scored.move.action != move.action:
            continue
        if move.action == "call" or scored.move.bid == move.bid:  # type: ignore[union-attr]
            return scored
    return None
