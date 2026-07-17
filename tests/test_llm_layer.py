"""LLM layer tests: real Instructor over a faked OpenAI transport.

Covers reprompt-until-legal (via the validation-context legality oracle),
fallback behavior, payload hygiene, and the susceptibility toggle.
"""

import json

import pytest
from openai.types.chat import ChatCompletion

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


def _completion(content: str, prompt: int = 100, completion: int = 20) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


def _fake_transport(monkeypatch, responses: list):
    """AsyncOpenAI client whose create() replays canned responses (or raises)."""
    from openai import AsyncOpenAI

    calls: list[dict] = []
    client = AsyncOpenAI(api_key="test-key", base_url="http://localhost:1")

    async def fake_create(**kwargs):
        calls.append(kwargs)
        result = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(result, Exception):
            raise result
        # Fresh object per call: Instructor mutates response.usage in place to
        # aggregate across retries, as real HTTP responses are never shared.
        return result.model_copy(deep=True)

    client.chat.completions.create = fake_create
    monkeypatch.setattr(llm, "_base_client", lambda: client)
    return calls


@pytest.fixture
def game(monkeypatch):
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: Settings(mock_llm=False, llm_max_reprompts=2, llm_model="test/model"),
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


async def test_reprompts_until_legal_and_accumulates_usage(monkeypatch, game):
    calls = _fake_transport(
        monkeypatch, [_completion(GARBAGE), _completion(ILLEGAL), _completion(GOOD)]
    )
    outcome = await _decide(game)
    assert len(calls) == 3
    assert outcome.reprompts == 2
    assert not outcome.fallback
    assert outcome.decision.move.action == "call"
    assert outcome.prompt_tokens == 300  # all three attempts accumulated
    # The reprompt for the illegal move must explain the rule in game terms.
    retry_text = json.dumps(calls[2]["messages"])
    assert "2x3" in retry_text


async def test_falls_back_to_scripted_after_budget(monkeypatch, game):
    calls = _fake_transport(monkeypatch, [_completion(GARBAGE)])
    outcome = await _decide(game)
    state, profile, own, menu, transcript = game
    assert outcome.fallback
    assert len(calls) == 3  # llm_max_reprompts=2 -> 3 attempts
    assert outcome.reprompts == 3
    assert outcome.prompt_tokens == 300  # burned tokens preserved on fallback
    assert probability.find_scored(menu, outcome.decision.move) is not None


async def test_transport_failure_falls_back_cleanly(monkeypatch, game):
    calls = _fake_transport(monkeypatch, [RuntimeError("openrouter down")])
    outcome = await _decide(game)
    state, profile, own, menu, transcript = game
    assert outcome.fallback
    assert len(calls) == 1  # non-validation errors are not retried
    assert outcome.prompt_tokens is None  # no tokens were ever billed
    assert probability.find_scored(menu, outcome.decision.move) is not None


async def test_payload_never_leaks_probability_engine(monkeypatch, game):
    calls = _fake_transport(monkeypatch, [_completion(GOOD)])
    await _decide(game)
    state, profile, own, menu, transcript = game
    text = json.dumps(calls[0]["messages"])
    assert "optimal" not in text.lower()
    assert "menu" not in text.lower()
    for scored in menu.moves:
        assert f"{scored.truth_probability:.3f}" not in text
    # But the character sheet and game state must be present.
    assert profile.name in text
    assert str(own[0]) in text


async def test_susceptibility_toggle_filters_human_talk(monkeypatch, game):
    calls = _fake_transport(monkeypatch, [_completion(GOOD), _completion(GOOD)])
    await _decide(game, susceptibility=True)
    on_text = json.dumps(calls[0]["messages"])
    assert HUMAN_TALK in on_text

    calls.clear()
    await _decide(game, susceptibility=False)
    off_text = json.dumps(calls[0]["messages"])
    assert HUMAN_TALK not in off_text
    assert "bid 2x3" in off_text  # the move itself is still public knowledge


async def test_extra_body_is_forwarded(monkeypatch, game):
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: Settings(
            mock_llm=False,
            llm_max_reprompts=2,
            llm_model="test/model",
            llm_extra_body='{"reasoning": {"effort": "none"}}',
        ),
    )
    calls = _fake_transport(monkeypatch, [_completion(GOOD)])
    await _decide(game)
    assert calls[0]["extra_body"] == {"reasoning": {"effort": "none"}}


async def test_mock_mode_never_touches_the_provider(monkeypatch, game):
    state, profile, own, menu, transcript = game
    monkeypatch.setattr(llm, "get_settings", lambda: Settings(mock_llm=True))

    def explode():
        raise AssertionError("client must not be constructed in mock mode")

    monkeypatch.setattr(llm, "_base_client", explode)
    outcome = await _decide(game)
    assert not outcome.fallback
    assert outcome.prompt_tokens is None
    assert probability.find_scored(menu, outcome.decision.move) is not None
