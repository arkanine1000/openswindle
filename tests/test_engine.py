import pytest

from openswindle import engine, fairness
from openswindle.models import Bid, BidMove, CallMove, MatchConfig


def make_match(dice_per_player: int = 3) -> engine.MatchState:
    return engine.create_match(
        MatchConfig(dice_per_player=dice_per_player, opponent_type="scripted")
    )


def test_create_match_deals_committed_hands():
    state = make_match(4)
    assert state.dice_counts == {"a": 4, "b": 4}
    for seat in ("a", "b"):
        hand = state.round.hands[seat]
        assert len(hand) == 4
        assert fairness.verify_commitment(
            state.round.salts[seat], hand, state.round.commitments[seat]
        )


def test_turn_order_enforced():
    state = make_match()
    with pytest.raises(engine.NotYourTurnError):
        engine.apply_move(state, "b", BidMove(bid=Bid(quantity=1, face=1)))


def test_bid_must_raise():
    state = make_match()
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=2, face=3)))
    with pytest.raises(engine.IllegalMoveError):
        engine.apply_move(state, "b", BidMove(bid=Bid(quantity=2, face=3)))
    with pytest.raises(engine.IllegalMoveError):
        engine.apply_move(state, "b", BidMove(bid=Bid(quantity=1, face=4)))
    # Same quantity, higher face is legal.
    engine.apply_move(state, "b", BidMove(bid=Bid(quantity=2, face=4)))


def test_bid_cannot_exceed_board():
    state = make_match(2)
    with pytest.raises(engine.IllegalMoveError):
        engine.apply_move(state, "a", BidMove(bid=Bid(quantity=5, face=1)))


def test_cannot_call_without_bid():
    state = make_match()
    with pytest.raises(engine.IllegalMoveError):
        engine.apply_move(state, "a", CallMove())


def test_call_adjudication_and_die_loss():
    state = make_match(3)
    face = state.round.hands["a"][0]
    total = engine.total_face_count(state, face)

    # A bid of exactly the true count is met: the caller ("b") must lose.
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=total, face=face)))
    reveal = engine.apply_move(state, "b", CallMove())
    assert reveal is not None
    assert reveal.bid_met
    assert reveal.loser == "b"
    assert state.dice_counts["b"] == 2
    # Loser opens the next round.
    assert state.round.turn == "b"
    assert state.round.round_no == 2


def test_call_table_talk_carried_into_reveal():
    state = make_match(3)
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=1, face=1)))
    reveal = engine.apply_move(state, "b", CallMove(), table_talk="You're bluffing.")
    assert reveal is not None
    # The call is not a bid, so bid_history never holds it — the reveal is the
    # only place the caller's words survive for the opponent to read.
    assert reveal.table_talk == "You're bluffing."


def test_call_without_table_talk_leaves_reveal_talk_none():
    state = make_match(3)
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=1, face=1)))
    reveal = engine.apply_move(state, "b", CallMove())
    assert reveal is not None
    assert reveal.table_talk is None


def test_overbid_punished_on_call():
    state = make_match(2)
    total_dice = sum(state.dice_counts.values())
    face = 1
    true_count = engine.total_face_count(state, face)
    # Bid more than the truth (guaranteed possible unless all dice show face).
    if true_count == total_dice:
        face = 2
        true_count = engine.total_face_count(state, face)
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=true_count + 1, face=face)))
    reveal = engine.apply_move(state, "b", CallMove())
    assert reveal is not None
    assert not reveal.bid_met
    assert reveal.loser == "a"


def test_match_ends_at_zero_dice():
    state = make_match(2)
    # Grind seat "a" down deterministically: "a" always bids one over the
    # truth, "b" always calls.
    while state.phase == "bidding":
        opener = state.round.turn
        other = "b" if opener == "a" else "a"
        face = 1
        count = engine.total_face_count(state, face)
        if count == sum(state.dice_counts.values()):
            face = 2
            count = engine.total_face_count(state, face)
        engine.apply_move(state, opener, BidMove(bid=Bid(quantity=count + 1, face=face)))
        engine.apply_move(state, other, CallMove())
    assert state.winner is not None
    assert 0 in state.dice_counts.values()
    with pytest.raises(engine.MatchFinishedError):
        engine.apply_move(state, "a", CallMove())


def test_reveal_verifies_against_commitments():
    state = make_match(3)
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=1, face=1)))
    reveal = engine.apply_move(state, "b", CallMove())
    assert reveal is not None
    for seat in ("a", "b"):
        assert fairness.verify_commitment(
            reveal.salts[seat], reveal.hands[seat], reveal.commitments[seat]
        )
