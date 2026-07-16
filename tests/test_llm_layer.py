"""LLM layer tests against a fake provider: reprompts, fallback, payload hygiene."""

import json
from types import SimpleNamespace

import litellm
import pytest

from openswindle import engine, probability
from openswindle.config import Settings
from openswindle.models import Bid, BidMove, MatchConfig, TranscriptEvent
from openswindle.npc import generator, llm

GOOD = json.dumps(
    {"scratchpad": "they are hiding something", "move": {"action": "call"}, "table_talk": "Liar."}
)
ILLEGAL = json.dumps(
    {
        "scratchpad": "x",
        "move": {"action": "bid", "bid": {"quantity": 1, "face": 1}},  # does not raise 2x3
        "table_talk": "",
    }
)
GARBAGE = "the dice are a metaphor, actually"

HUMAN_TALK = "trust me, there are four threes on this table"


def _response(content: str, prompt: int = 100, cached: int = 0, completion: int = 20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


@pytest.fixture
def game(monkeypatch):
    monkeypatch.setattr(
        llm, "get_settings", lambda: Settings(mock_llm=False, llm_max_reprompts=2)
    )
    state = engine.create_match(MatchConfig(dice_per_player=3, opponent_type="llm"))
    engine.apply_move(state, "a", BidMove(bid=Bid(quantity=2, face=3)), HUMAN_TALK)
    profile = generator.generate_npc("seed 4471")
    own = state.round.hands["b"]
    menu = probability.build_menu(state.round.current_bid, own, state.dice_counts["a"])
    transcript = [
        TranscriptEvent(round_no=1, seat="a", kind="talk", text=HUMAN_TALK),
        TranscriptEvent(round_no=1, seat="a", kind="bid", text="2x3"),
    ]
    return state, profile, own, menu, transcript


async def _decide(game, susceptibility: bool = True):
    state, profile, own, menu, transcript = game
    return await llm.decide(
        profile, menu, state.round, own, state.dice_counts["a"], transcript, "b", susceptibility
    )


def _fake_provider(monkeypatch, responses: list):
    captured: list[dict] = []

    async def fake(**kwargs):
        captured.append(kwargs)
        return responses[min(len(captured) - 1, len(responses) - 1)]

    monkeypatch.setattr(litellm, "acompletion", fake)
    return captured


async def test_reprompts_until_legal_and_accumulates_usage(monkeypatch, game):
    captured = _fake_provider(
        monkeypatch, [_response(GARBAGE), _response(ILLEGAL), _response(GOOD)]
    )
    outcome = await _decide(game)
    assert len(captured) == 3
    assert outcome.reprompts == 2
    assert not outcome.fallback
    assert outcome.decision.move.action == "call"
    assert outcome.prompt_tokens == 300  # all three attempts priced
    # The reprompt must explain the violation in rule terms.
    rejection = captured[2]["messages"][-1]["content"]
    assert "2x3" in rejection


async def test_falls_back_to_scripted_after_budget(monkeypatch, game):
    _fake_provider(monkeypatch, [_response(GARBAGE)])
    outcome = await _decide(game)
    state, profile, own, menu, transcript = game
    assert outcome.fallback
    assert outcome.reprompts == 3
    assert outcome.prompt_tokens == 300  # burned tokens preserved on fallback
    assert probability.find_scored(menu, outcome.decision.move) is not None


async def test_transport_failure_falls_back_cleanly(monkeypatch, game):
    async def explode(**kwargs):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(litellm, "acompletion", explode)
    outcome = await _decide(game)
    assert outcome.fallback
    assert outcome.reprompts == 0
    assert outcome.prompt_tokens is None  # no tokens were ever billed
    state, profile, own, menu, transcript = game
    assert probability.find_scored(menu, outcome.decision.move) is not None


async def test_payload_never_leaks_probability_engine(monkeypatch, game):
    captured = _fake_provider(monkeypatch, [_response(GOOD)])
    await _decide(game)
    state, profile, own, menu, transcript = game
    text = "\n".join(m["content"] for m in captured[0]["messages"])
    assert "optimal" not in text.lower()
    assert "menu" not in text.lower()
    for scored in menu.moves:
        assert f"{scored.truth_probability:.3f}" not in text
    # But the character sheet and game state must be present.
    assert profile.bio in text
    assert str(own) in text


async def test_susceptibility_toggle_filters_human_talk(monkeypatch, game):
    captured = _fake_provider(monkeypatch, [_response(GOOD), _response(GOOD)])
    await _decide(game, susceptibility=True)
    on_text = "\n".join(m["content"] for m in captured[0]["messages"])
    assert HUMAN_TALK in on_text

    captured.clear()
    await _decide(game, susceptibility=False)
    off_text = "\n".join(m["content"] for m in captured[0]["messages"])
    assert HUMAN_TALK not in off_text
    assert "bid 2x3" in off_text  # the move itself is still public knowledge


async def test_mock_mode_never_touches_the_provider(monkeypatch, game):
    state, profile, own, menu, transcript = game
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(mock_llm=True))

    async def explode(**kwargs):
        raise AssertionError("provider must not be called in mock mode")

    monkeypatch.setattr(litellm, "acompletion", explode)
    outcome = await _decide(game)
    assert not outcome.fallback
    assert outcome.prompt_tokens is None
    assert probability.find_scored(menu, outcome.decision.move) is not None
