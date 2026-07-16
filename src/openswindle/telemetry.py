"""Deviation pricing and scratchpad history for the post-match autopsy."""

from .models import (
    Autopsy,
    DecisionRecord,
    LLMDecision,
    NPCProfile,
    ProbabilityMenu,
    Seat,
)
from .probability import find_scored


def price_decision(
    round_no: int,
    decision: LLMDecision,
    menu: ProbabilityMenu,
    susceptibility_on: bool,
    human_table_talk: str | None,
    fallback: bool = False,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> DecisionRecord:
    """Price a decision: the win-probability delta vs the optimal move."""
    optimal = menu.optimal_move
    chosen = find_scored(menu, decision.move)
    if chosen is None:  # Defensive: callers validate legality before pricing.
        raise ValueError(f"Cannot price an illegal move: {decision.move}")
    return DecisionRecord(
        round_no=round_no,
        chosen_move=decision.move,
        optimal_move=optimal.move,
        chosen_probability=chosen.truth_probability,
        optimal_probability=optimal.truth_probability,
        deviation_price=optimal.truth_probability - chosen.truth_probability,
        scratchpad=decision.scratchpad,
        table_talk=decision.table_talk,
        susceptibility_on=susceptibility_on,
        human_table_talk_seen=human_table_talk if susceptibility_on else None,
        fallback=fallback,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )


def build_autopsy(
    match_id: str,
    winner: Seat | None,
    profile: NPCProfile,
    decisions: list[DecisionRecord],
) -> Autopsy:
    return Autopsy(
        match_id=match_id,
        winner=winner,
        npc_profile=profile,
        decisions=decisions,
        total_deviation_price=sum(d.deviation_price for d in decisions),
    )
