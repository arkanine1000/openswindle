from openswindle import engine
from openswindle.models import MatchConfig
from openswindle.store import MatchStore


def _finished_record(store: MatchStore, now: list[float], seed: str = "seed") -> str:
    state = engine.create_match(MatchConfig(opponent_type="scripted", npc_seed=seed))
    record = store.add(state, npc_profile=None)
    engine.abort_match(state)
    store.mark_finished(record)
    now[0] += 1
    return state.match_id


def test_finished_matches_expire_by_ttl():
    now = [100.0]
    store = MatchStore(finished_ttl_seconds=5, max_finished_matches=100, now=lambda: now[0])
    match_id = _finished_record(store, now)

    assert store.get(match_id) is not None
    now[0] += 5
    assert store.get(match_id) is None


def test_finished_matches_prune_by_max_count_without_touching_active_matches():
    now = [100.0]
    store = MatchStore(finished_ttl_seconds=1000, max_finished_matches=2, now=lambda: now[0])
    active = engine.create_match(MatchConfig(opponent_type="scripted"))
    store.add(active, npc_profile=None)

    first = _finished_record(store, now, "seed 1")
    second = _finished_record(store, now, "seed 2")
    third = _finished_record(store, now, "seed 3")

    assert store.get(active.match_id) is not None
    assert store.get(first) is None
    assert store.get(second) is not None
    assert store.get(third) is not None
