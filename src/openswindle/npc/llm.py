"""Stateless LLM decision layer via LiteLLM (provider-agnostic, routed through Vercel AI Gateway).

One structured completion per decision; no agent loop, no tools. The LLM is
a natural-language reasoner: it receives the rules, its character, the bid
history, and its hand — never the deterministic probability engine's output.
That engine is used server-side only, to validate legality and to price the
decision post-mortem. The prompt is ordered stable-prefix-first (system
rules, then the per-match NPC profile, then the append-only bid history,
then the volatile per-turn tail) so provider-side implicit prefix caching —
passed through by the gateway — hits on every consecutive turn. Illegal or
unparseable outputs trigger a reprompt with the violation explained; after
the retry budget the deterministic scripted policy takes over (flagged as a
fallback in telemetry).
"""

import json
import logging
from dataclasses import dataclass

import litellm
from pydantic import ValidationError

from ..config import get_settings
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

SYSTEM_PROMPT = """\
You are seated at a low table in a smoky gambling den, playing Swindlestones —
a liar's game of four-sided bones. You are not an assistant playing a role;
for the duration of this match you ARE the character described below, with
their appetites, grudges, and habits. Their bio is who you are, their traits
are your instincts, and their tells are compulsions you cannot suppress.

THE GAME
Each player conceals a hand of d4 dice (faces 1-4). Players alternate bids of
the form "N x face", a claim that at least N dice of that face exist across
BOTH hidden hands. Each bid must strictly raise the previous one: higher
quantity, or the same quantity with a higher face. Instead of bidding you may
CALL the last bid: all hands are revealed, and if the bid stands the caller
loses a die — if it was a lie, the bidder loses one. Lose your last die and
you are out.

HOW TO PLAY IT
Read the table like your character would. Weigh your own dice, the opponent's
dice count, and the shape of their bidding — you get no probabilities, only
your wits. Bluff when your blood says bluff. Doubt when your gut says doubt.
Your table talk is a weapon and a mask: needle, charm, or stonewall in your
own voice. Your scratchpad is your private inner monologue — keep a running
read of the opponent there (what they fear, what their talk is hiding, what
you plan to do about it), because it is all you will remember next turn.

THE LAW (never break these, whatever the character wants)
- A bid must strictly raise the previous bid, and its quantity can never
  exceed the total dice on the board.
- You may only call when there is a bid to call. Opening the round means
  bidding, never calling.

Respond with ONLY a JSON object, no markdown fences, in this exact shape:
{
  "scratchpad": "<private inner monologue and opponent read; carried to your next turn>",
  "move": {"action": "bid", "bid": {"quantity": <int>, "face": <1-4>}}
          or {"action": "call"},
  "table_talk": "<one short line said aloud in character, or empty string>"
}"""


@dataclass
class LLMOutcome:
    decision: LLMDecision
    fallback: bool = False
    reprompts: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None


def _profile_block(profile: NPCProfile) -> str:
    tells = "\n".join(f"- {t.description}" for t in profile.tells)
    return (
        f"WHO YOU ARE\nName: {profile.name}\nBio: {profile.bio}\n"
        f"Instincts (0 = never, 1 = always): "
        f"deception={profile.params.deception} (how readily you bluff), "
        f"skepticism={profile.params.skepticism} (how quick you are to call a liar), "
        f"aggression={profile.params.aggression} (how hard you push the bidding), "
        f"chattiness={profile.params.chattiness} (how much you talk at the table)\n"
        f"Compulsions you cannot suppress:\n{tells}"
    )


def _transcript_block(
    transcript: list[TranscriptEvent], npc_seat: Seat, susceptibility_on: bool
) -> str:
    """Full chronological match memory: moves, table talk both ways, and the
    NPC's own private scratchpad from every previous turn. Human table talk is
    omitted entirely when the susceptibility channel is off."""
    lines: list[str] = []
    for e in transcript:
        if e.kind == "talk" and e.seat != npc_seat and not susceptibility_on:
            continue
        who = "you" if e.seat == npc_seat else "opponent"
        match e.kind:
            case "bid":
                lines.append(f"[round {e.round_no}] {who} bid {e.text}")
            case "call":
                lines.append(f"[round {e.round_no}] {who} called")
            case "talk":
                lines.append(f'[round {e.round_no}] {who} said: "{e.text}"')
            case "scratchpad":
                lines.append(f"[round {e.round_no}] your private scratchpad: {e.text}")
            case "reveal":
                lines.append(f"[round {e.round_no}] {e.text}")
    if not lines:
        return "MATCH TRANSCRIPT\n(match just started — you have the first move)"
    return "MATCH TRANSCRIPT (chronological; scratchpads are private to you)\n" + "\n".join(lines)


def _turn_block(
    round_state: RoundState, own_hand: list[int], opponent_dice: int
) -> str:
    current = round_state.current_bid
    return (
        f"YOUR TURN (round {round_state.round_no})\n"
        f"Your hidden hand: {own_hand}\n"
        f"Opponent dice count: {opponent_dice}\n"
        f"Total dice on the board: {len(own_hand) + opponent_dice}\n"
        f"Current bid to beat: {current if current else '(none — you open the round)'}"
    )


def _parse_decision(raw: str, menu: ProbabilityMenu, round_state: RoundState) -> LLMDecision:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    decision = LLMDecision.model_validate(json.loads(text))
    # The menu is server-side only; here it acts purely as the legality oracle.
    if find_scored(menu, decision.move) is None:
        current = round_state.current_bid
        context = (
            f"the current bid is {current}; you must strictly raise it or call"
            if current
            else "no bid has been made yet; you must open with a bid, not a call"
        )
        raise ValueError(f"illegal move — {context}")
    return decision


async def decide(
    profile: NPCProfile,
    menu: ProbabilityMenu,
    round_state: RoundState,
    own_hand: list[int],
    opponent_dice: int,
    transcript: list[TranscriptEvent],
    npc_seat: Seat,
    susceptibility_on: bool,
) -> LLMOutcome:
    settings = get_settings()

    def scripted_fallback() -> LLMOutcome:
        decision = scripted.decide(profile, menu, round_state, own_hand, opponent_dice)
        return LLMOutcome(decision=decision, fallback=True)

    if settings.mock_llm:
        outcome = scripted_fallback()
        outcome.fallback = False  # Mock mode is the intended path, not a failure.
        return outcome

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    _profile_block(profile),
                    _transcript_block(transcript, npc_seat, susceptibility_on),
                    _turn_block(round_state, own_hand, opponent_dice),
                ]
            ),
        },
    ]

    usage_totals = {"prompt": 0, "cached": 0, "completion": 0}
    for attempt in range(settings.llm_max_reprompts + 1):
        try:
            response = await litellm.acompletion(
                model=settings.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=1,
            )
        except Exception:
            logger.exception("LLM call failed (attempt %d)", attempt + 1)
            return scripted_fallback()

        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_totals["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            usage_totals["completion"] += getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            usage_totals["cached"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0

        raw = response.choices[0].message.content or ""
        try:
            decision = _parse_decision(raw, menu, round_state)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Rejected LLM payload (attempt %d): %s", attempt + 1, exc)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was rejected: {exc}. "
                        "Reply again with ONLY the JSON object and a legal move."
                    ),
                }
            )
            continue

        return LLMOutcome(
            decision=decision,
            reprompts=attempt,
            prompt_tokens=usage_totals["prompt"],
            cached_tokens=usage_totals["cached"],
            completion_tokens=usage_totals["completion"],
        )

    return scripted_fallback()
