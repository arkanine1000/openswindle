from openswindle import engine, probability
from openswindle.models import MatchConfig
from openswindle.npc import generator, scripted


def test_generation_is_deterministic():
    assert generator.generate_npc("seed 4471") == generator.generate_npc("seed 4471")


def test_generation_varies_between_seeds():
    bios = {generator.generate_npc(str(i)).bio for i in range(20)}
    assert len(bios) > 1


def test_params_bound_and_one_decimal():
    for i in range(50):
        p = generator.generate_npc(f"seed {i}").params
        for value in (p.deception, p.skepticism, p.aggression, p.chattiness):
            assert 0.0 <= value <= 1.0
            assert value == round(value, 1)


def test_bio_derives_from_params():
    for i in range(30):
        profile = generator.generate_npc(f"seed {i}")
        p, bio = profile.params, profile.bio
        if p.aggression >= 0.7:
            assert "belligerent" in bio
        elif p.aggression <= 0.3:
            assert "timid" in bio
        else:
            assert "weathered" in bio
        if p.deception >= 0.7:
            assert "lies as easily as breathing" in bio
        elif p.deception <= 0.3:
            assert "barely stomach a lie" in bio
        else:
            assert "bends the truth" in bio
        expected_doubt = "trusts nothing" if p.skepticism >= 0.5 else "face value"
        assert expected_doubt in bio
        expected_mouth = "never stops talking" if p.chattiness >= 0.5 else "dice demand it"
        assert expected_mouth in bio


def _opening_decision(seed: str):
    state = engine.create_match(MatchConfig(dice_per_player=4, opponent_type="scripted"))
    profile = generator.generate_npc(seed)
    menu = probability.build_menu(None, state.round.hands["b"], state.dice_counts["a"])
    return profile, menu, state


def test_scripted_policy_is_consistent_within_character():
    profile, menu, state = _opening_decision("seed 7")
    assert scripted.decide(profile, menu, state.round) == scripted.decide(
        profile, menu, state.round
    )


def test_scripted_opening_is_always_a_legal_bid():
    for i in range(30):
        profile, menu, state = _opening_decision(f"seed {i}")
        decision = scripted.decide(profile, menu, state.round)
        assert decision.move.action == "bid"
        assert probability.find_scored(menu, decision.move) is not None
