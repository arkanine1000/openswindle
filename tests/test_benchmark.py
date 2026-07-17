"""Benchmark runner tests (offline: scripted opponent and mock LLM)."""

import dataclasses

from openswindle import benchmark


async def test_scripted_benchmark_is_reproducible():
    options = benchmark.BenchmarkOptions(
        matches=2, dice_per_player=3, opponent_type="scripted", run_seed="bench-test-1"
    )
    first = await benchmark.run(options)
    second = await benchmark.run(options)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)

    assert len(first.modes) == 1  # susceptibility channel doesn't exist for scripted
    mode = first.modes[0]
    assert not mode.susceptibility_on
    assert mode.matches == 2
    assert mode.decisions > 0
    assert mode.total_deviation_price >= 0
    assert 0.0 <= mode.optimal_move_rate <= 1.0
    assert first.susceptibility_price_delta is None
    assert first.model is None


async def test_mock_llm_benchmark_runs_paired_modes():
    options = benchmark.BenchmarkOptions(
        matches=1,
        dice_per_player=2,
        opponent_type="llm",
        susceptibility="both",
        run_seed="bench-test-2",
    )
    report = await benchmark.run(options)
    assert report.mock_llm  # conftest forces mock mode
    assert report.model is None
    assert [m.susceptibility_on for m in report.modes] == [True, False]
    assert all(m.fallbacks == 0 for m in report.modes)
    assert report.susceptibility_price_delta is not None
    # Mock decisions ignore table talk, so paired deals must price identically.
    assert report.susceptibility_price_delta == 0.0


async def test_different_run_seeds_change_the_deals():
    base = benchmark.BenchmarkOptions(
        matches=1, dice_per_player=3, opponent_type="scripted", run_seed="bench-test-3"
    )
    other = dataclasses.replace(base, run_seed="bench-test-4")
    first = await benchmark.run(base)
    second = await benchmark.run(other)
    assert dataclasses.asdict(first.modes[0]) != dataclasses.asdict(second.modes[0])


def test_salt_patch_restores_the_original():
    from openswindle import fairness

    original = fairness.draw_salt
    with benchmark.deterministic_salts("bench-test-5"):
        assert fairness.draw_salt is not original
    assert fairness.draw_salt is original


async def test_benchmark_transcript_includes_seat_free_reveals(monkeypatch):
    """Parity with the API path: benchmark NPCs must see round-end reveals."""
    seen: dict = {}
    original = benchmark.llm.decide

    async def spy(profile, menu, round_state, own, opponent_dice, transcript, seat, sus):
        seen["transcript"] = transcript
        return await original(
            profile, menu, round_state, own, opponent_dice, transcript, seat, sus
        )

    monkeypatch.setattr(benchmark.llm, "decide", spy)
    options = benchmark.BenchmarkOptions(
        matches=1,
        dice_per_player=2,
        opponent_type="llm",
        susceptibility="off",
        run_seed="bench-parity-1",
    )
    await benchmark.run(options)

    reveals = [e for e in seen["transcript"] if e.kind == "reveal"]
    assert reveals, "benchmark transcripts must include round-end reveals"
    assert all("seat" not in e.text for e in reveals)
