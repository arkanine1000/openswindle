"""Stateless LLM decision layer via OpenRouter + Instructor.

One structured completion per decision; no agent loop, no tools. The LLM is
a natural-language reasoner: it receives the rules, its character, the match
transcript, and its hand — never the deterministic probability engine's
output. That engine acts server-side only: for benchmarking (deviation
pricing) and as the legality oracle, which is wired into Instructor's
validation context so an illegal move fails Pydantic validation and
Instructor reprompts the model with the rule explained. The prompt is
ordered stable-prefix-first (system rules, then the per-match character
sheet, then the append-only transcript, then the volatile per-turn tail) so
provider-side prompt caching hits where the provider supports it. After the
retry budget the deterministic scripted policy takes over (flagged as a
fallback in telemetry).
"""

import logging
from dataclasses import dataclass

from pydantic import ValidationInfo, model_validator

from ..config import get_settings
from ..i18n import FALLBACK, Locale, s, system_prompt
from ..models import (
    LLMDecision,
    NPCProfile,
    ProbabilityMenu,
    RoundState,
    Seat,
    TranscriptEvent,
)
from ..probability import find_scored
from . import scripted

logger = logging.getLogger(__name__)
# The system prompt (and all model-facing scaffolding) is localized in
# ``i18n.py`` — see system_prompt(locale) and s(locale, key).


@dataclass
class LLMOutcome:
    decision: LLMDecision
    fallback: bool = False
    reprompts: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None


# NOTE: Instructor embeds this class's docstring in the schema shown to the
# model, so it must stay in-fiction — implementation notes live here instead.
# The legality oracle arrives via Instructor's validation context; an illegal
# move raises, which Instructor turns into a reprompt with the rule spelled
# out. The probability menu itself is never shown to the model.
class ValidatedDecision(LLMDecision):
    """Your decision for this turn of Swindlestones."""

    @model_validator(mode="after")
    def _move_must_be_legal(self, info: ValidationInfo) -> "ValidatedDecision":
        context = info.context or {}
        menu = context.get("menu")
        if menu is None or find_scored(menu, self.move) is not None:
            return self
        locale: Locale = context.get("locale", FALLBACK)
        round_state = context.get("round_state")
        current = round_state.current_bid if round_state is not None else None
        rule = (
            s(locale, "reprompt_have_bid").format(bid=current)
            if current
            else s(locale, "reprompt_open")
        )
        raise ValueError(s(locale, "reprompt_prefix").format(rule=rule))


def _profile_block(profile: NPCProfile, locale: Locale = FALLBACK) -> str:
    p = profile.params
    return (
        s(locale, "who_you_are").format(name=profile.name, bio=profile.bio)
        + s(locale, "instincts")
        + s(locale, "trait_deception").format(v=p.deception)
        + s(locale, "trait_skepticism").format(v=p.skepticism)
        + s(locale, "trait_aggression").format(v=p.aggression)
        + s(locale, "trait_chattiness").format(v=p.chattiness)
    )


def _transcript_block(
    transcript: list[TranscriptEvent],
    npc_seat: Seat,
    susceptibility_on: bool,
    locale: Locale = FALLBACK,
) -> str:
    """Full chronological match memory: moves, table talk both ways, and the
    NPC's own private scratchpad from every previous turn. Human table talk is
    omitted entirely when the susceptibility channel is off."""
    who_you = s(locale, "who_you")
    who_opp = s(locale, "who_opponent")
    lines: list[str] = []
    for e in transcript:
        if e.kind == "talk" and e.seat != npc_seat and not susceptibility_on:
            continue
        who = who_you if e.seat == npc_seat else who_opp
        match e.kind:
            case "bid":
                lines.append(s(locale, "line_bid").format(r=e.round_no, who=who, text=e.text))
            case "call":
                lines.append(s(locale, "line_call").format(r=e.round_no, who=who))
            case "talk":
                lines.append(s(locale, "line_talk").format(r=e.round_no, who=who, text=e.text))
            case "scratchpad":
                lines.append(s(locale, "line_scratchpad").format(r=e.round_no, text=e.text))
            case "reveal":
                # e.seat carries the round's loser; named in the NPC's frame —
                # a raw seat letter is meaningless to the character.
                lines.append(s(locale, "line_reveal").format(r=e.round_no, text=e.text, who=who))
    if not lines:
        return s(locale, "transcript_empty")
    return s(locale, "transcript_header") + "\n" + "\n".join(lines)


def _hand_counts(own_hand: list[int]) -> str:
    # Reuse the bid syntax ("N x face") the model already reads in THE LAW, so
    # tallying its own duplicates doesn't ask it to hold a second notation in
    # its head while it's also reasoning about the bid and the opponent.
    counts = [(face, own_hand.count(face)) for face in range(1, 5) if face in own_hand]
    return ", ".join(f"{n} x {face}" for face, n in counts)


def _turn_block(
    round_state: RoundState, own_hand: list[int], opponent_dice: int, locale: Locale = FALLBACK
) -> str:
    current = round_state.current_bid
    # Attribute the standing bid to the opponent and state the stance, right in
    # the volatile tail the model leans on hardest — otherwise it can narrate
    # its own new bid as if the opponent had claimed it.
    stance = (
        s(locale, "stance_have_bid").format(bid=current)
        if current is not None
        else s(locale, "stance_open")
    )
    header = s(locale, "turn_header").format(
        r=round_state.round_no,
        hand=own_hand,
        counts=_hand_counts(own_hand),
        opp=opponent_dice,
        total=len(own_hand) + opponent_dice,
    )
    return header + stance


def _accumulate_usage(totals: dict[str, int], response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    totals["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    totals["completion"] += getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    totals["cached"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0


def _base_client():
    """OpenAI SDK client pointed at OpenRouter. Factory kept separate so
    tests can substitute a fake transport underneath Instructor."""
    from openai import AsyncOpenAI

    settings = get_settings()
    return AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/arkanine1000/openswindle",
            "X-Title": "OpenSwindle",
        },
    )


async def decide(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    own_hand: list[int],
    opponent_dice: int,
    transcript: list[TranscriptEvent],
    npc_seat: Seat,
    susceptibility_on: bool,
    locale: Locale = FALLBACK,
) -> LLMOutcome:
    settings = get_settings()

    if settings.mock_llm:
        # Mock mode is the intended path, not a failure.
        return LLMOutcome(decision=scripted.decide(profile, menu, round_state, locale))

    # Deferred import: instructor is heavy, and mock mode should stay instant.
    import instructor

    usage_totals = {"prompt": 0, "cached": 0, "completion": 0}
    rejections = 0

    # Meter usage at the transport boundary — exactly one accumulation per
    # request, regardless of how the retry layer re-emits responses.
    raw_client = _base_client()
    transport_create = raw_client.chat.completions.create

    async def _metered_create(**kwargs):
        response = await transport_create(**kwargs)
        _accumulate_usage(usage_totals, response)
        return response

    raw_client.chat.completions.create = _metered_create
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    def _on_parse_error(error) -> None:
        nonlocal rejections
        rejections += 1
        logger.warning("Rejected LLM payload (rejection %d): %s", rejections, error)

    client.on("parse:error", _on_parse_error)

    messages = [
        {"role": "system", "content": system_prompt(locale)},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    _profile_block(profile, locale),
                    _transcript_block(transcript, npc_seat, susceptibility_on, locale),
                    _turn_block(round_state, own_hand, opponent_dice, locale),
                ]
            ),
        },
    ]

    request: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 1,
        "response_model": ValidatedDecision,
        # Instructor's max_retries counts retries after the first attempt,
        # which maps 1:1 onto the reprompt budget.
        "max_retries": settings.llm_max_reprompts,
        # locale reaches the validator so reprompts are in the match's language.
        "context": {"menu": menu, "round_state": round_state, "locale": locale},
    }
    if settings.llm_extra_body_dict:
        request["extra_body"] = settings.llm_extra_body_dict

    try:
        decision = await client.chat.completions.create(**request)
    except Exception:
        logger.exception("LLM decision failed; falling back to scripted policy")
        made_calls = usage_totals["prompt"] > 0
        return LLMOutcome(
            decision=scripted.decide(profile, menu, round_state, locale),
            fallback=True,
            reprompts=rejections,
            prompt_tokens=usage_totals["prompt"] if made_calls else None,
            cached_tokens=usage_totals["cached"] if made_calls else None,
            completion_tokens=usage_totals["completion"] if made_calls else None,
        )

    return LLMOutcome(
        decision=decision,
        reprompts=rejections,
        prompt_tokens=usage_totals["prompt"],
        cached_tokens=usage_totals["cached"],
        completion_tokens=usage_totals["completion"],
    )
