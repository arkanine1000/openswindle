"""Authoritative match engine: dealing, bid validation, call adjudication."""

import uuid

from . import fairness
from .models import (
    Bid,
    BidRecord,
    MatchConfig,
    MatchState,
    Move,
    RoundReveal,
    RoundState,
    Seat,
    other_seat,
)


class EngineError(Exception):
    """Base class for rule violations."""


class NotYourTurnError(EngineError):
    pass


class IllegalMoveError(EngineError):
    pass


class MatchFinishedError(EngineError):
    pass


def _deal_round(dice_counts: dict[Seat, int], round_no: int, opener: Seat) -> RoundState:
    salts = {seat: fairness.draw_salt() for seat in ("a", "b")}
    hands: dict[Seat, list[int]] = {}
    commitments: dict[Seat, str] = {}
    for seat in ("a", "b"):
        hands[seat] = fairness.deal_hand(
            salts["a"], salts["b"], round_no, seat, dice_counts[seat]
        )
        commitments[seat] = fairness.commit_hand(salts[seat], hands[seat])
    return RoundState(
        round_no=round_no,
        hands=hands,
        salts={seat: salt.hex() for seat, salt in salts.items()},
        commitments=commitments,
        turn=opener,
    )


def create_match(config: MatchConfig) -> MatchState:
    dice_counts: dict[Seat, int] = {"a": config.dice_per_player, "b": config.dice_per_player}
    return MatchState(
        match_id=uuid.uuid4().hex,
        config=config,
        dice_counts=dice_counts,
        round=_deal_round(dice_counts, round_no=1, opener="a"),
    )


def total_face_count(state: MatchState, face: int) -> int:
    return sum(hand.count(face) for hand in state.round.hands.values())


def apply_move(
    state: MatchState, seat: Seat, move: Move, table_talk: str | None = None
) -> RoundReveal | None:
    """Mutate ``state`` with ``seat``'s move.

    Returns the RoundReveal if the move was a call (round terminated),
    otherwise None. Raises EngineError subclasses on rule violations.
    """
    if state.phase == "finished":
        raise MatchFinishedError("Match is already finished")
    if state.round.turn != seat:
        raise NotYourTurnError(f"It is seat {state.round.turn}'s turn")

    if move.action == "bid":
        return _apply_bid(state, seat, move.bid, table_talk)
    return _apply_call(state, seat, table_talk)


def _apply_bid(
    state: MatchState, seat: Seat, bid: Bid, table_talk: str | None
) -> None:
    current = state.round.current_bid
    total_dice = sum(state.dice_counts.values())
    if bid.quantity > total_dice:
        raise IllegalMoveError(
            f"Bid quantity {bid.quantity} exceeds total dice on the board ({total_dice})"
        )
    if current is not None and not bid.raises(current):
        raise IllegalMoveError(
            f"Bid {bid} does not raise the current bid {current}: "
            "raise the quantity, or keep it and raise the face"
        )
    state.round.bid_history.append(BidRecord(seat=seat, bid=bid, table_talk=table_talk))
    state.round.turn = other_seat(seat)
    return None


def _apply_call(
    state: MatchState, caller: Seat, table_talk: str | None = None
) -> RoundReveal:
    final_bid = state.round.current_bid
    if final_bid is None:
        raise IllegalMoveError("Cannot call before any bid has been made")

    actual = total_face_count(state, final_bid.face)
    bid_met = actual >= final_bid.quantity
    # If the bid stands, the caller loses; otherwise the bidder loses.
    bidder = state.round.bid_history[-1].seat
    loser = caller if bid_met else bidder

    reveal = RoundReveal(
        round_no=state.round.round_no,
        hands=state.round.hands,
        salts=state.round.salts,
        commitments=state.round.commitments,
        final_bid=final_bid,
        caller=caller,
        actual_count=actual,
        bid_met=bid_met,
        loser=loser,
        table_talk=table_talk or None,
    )
    state.reveals.append(reveal)
    state.dice_counts[loser] -= 1

    if state.dice_counts[loser] == 0:
        state.phase = "finished"
        state.winner = other_seat(loser)
    else:
        state.round = _deal_round(
            state.dice_counts, round_no=state.round.round_no + 1, opener=loser
        )
    return reveal


def abort_match(state: MatchState) -> RoundReveal:
    """Software abort: reveal the current round unconditionally and end the match."""
    current = state.round.current_bid or Bid(quantity=1, face=1)
    reveal = RoundReveal(
        round_no=state.round.round_no,
        hands=state.round.hands,
        salts=state.round.salts,
        commitments=state.round.commitments,
        final_bid=current,
        caller=state.round.turn,
        actual_count=total_face_count(state, current.face),
        bid_met=total_face_count(state, current.face) >= current.quantity,
        loser=state.round.turn,
    )
    state.reveals.append(reveal)
    state.phase = "finished"
    return reveal
