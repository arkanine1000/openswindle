from openswindle import engine, probability
from openswindle.models import MatchConfig
from openswindle.npc import generator, scripted


def test_generation_is_deterministic():
    a = generator.generate_npc("seed 4471")
    b = generator.generate_npc("seed 4471")
    assert a == b


def test_generation_varies_between_seeds():
    profiles = {generator.generate_npc(str(i)).bio for i in range(20)}
    assert len(profiles) > 1


def test_params_bound_and_tells_planted():
    for i in range(50):
        profile = generator.generate_npc(f"seed {i}")
        p = profile.params
        for value in (p.deception, p.skepticism, p.aggression, p.chattiness):
            assert 0.0 <= value <= 1.0
        assert 1 <= len(profile.tells) <= 2
        assert all(t.tell_id in generator.TELL_CATALOG for t in profile.tells)


def test_bio_derives_from_params():
    profile = generator.generate_npc("seed 4471")
    # The archetype adjective must match the parameter quadrant.
    p = profile.params
    if p.deception >= 0.5 and p.aggression >= 0.5:
        assert "unhinged" in profile.bio
    elif p.deception >= 0.5:
        assert "smiling" in profile.bio
    elif p.aggression >= 0.5:
        assert "belligerent" in profile.bio
    else:
        assert "timid" in profile.bio


def _decision_for(seed: str):
    state = engine.create_match(MatchConfig(dice_per_player=4, opponent_type="scripted"))
    profile = generator.generate_npc(seed)
    own = state.round.hands["b"]
    menu = probability.build_menu(None, own, state.dice_counts["a"])
    return scripted.decide(profile, menu, state.round, own, state.dice_counts["a"])


def test_scripted_policy_is_consistent_within_character():
    # Same character in the same situation always acts identically.
    state = engine.create_match(MatchConfig(dice_per_player=4, opponent_type="scripted"))
    profile = generator.generate_npc("seed 7")
    own = state.round.hands["b"]
    menu = probability.build_menu(None, own, state.dice_counts["a"])
    first = scripted.decide(profile, menu, state.round, own, state.dice_counts["a"])
    second = scripted.decide(profile, menu, state.round, own, state.dice_counts["a"])
    assert first == second


def test_scripted_policy_always_legal():
    for i in range(30):
        decision = _decision_for(f"seed {i}")
        # Opening move can never be a call.
        assert decision.move.action == "bid"
