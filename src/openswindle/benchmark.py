"""Reproducible benchmark runner over the NPC decision layer.

Plays seeded matches with a scripted probe in the human seat and aggregates
the telemetry the API records per match: deviation pricing against the
optimal move, discipline (fallbacks, reprompts), and token usage. The probe
emits claim-bearing table talk — half plain truths about its own hand, half
confident nonsense — so the susceptibility channel has something to bite on;
running with --susceptibility both plays the same salted deals with the
channel on and off and reports the deviation-price delta between them.

With a fixed --run-seed, salts, deals, probe behavior, and NPC seeds are all
deterministic: identical runs produce identical reports for deterministic
opponents (scripted or mock). Live LLM opponents introduce their own
variance on top of the fixed deals.

Usage:
    uv run openswindle-benchmark --matches 5 --opponent scripted
    uv run openswindle-benchmark --matches 10 --opponent llm --susceptibility both
"""

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from random import Random

from . import engine, fairness, probability, telemetry
from .config import get_settings
from .models import (
    DecisionRecord,
    MatchConfig,
    Move,
    Seat,
    TranscriptEvent,
    other_seat,
)
from .npc import generator, llm, scripted
from .npc.generator import stable_hash

PROBE_SEAT: Seat = "a"
NPC_SEAT: Seat = "b"


@dataclass
class BenchmarkOptions:
    matches: int = 5
    dice_per_player: int = 4
    opponent_type: str = "scripted"  # "scripted" | "llm"
    susceptibility: str = "auto"  # "on" | "off" | "both" | "auto"
    npc_seed_base: str = "bench"
    probe_seed: str = "probe 1"
    run_seed: str = "openswindle-bench-1"
    verbose: bool = False


@dataclass
class ModeReport:
    susceptibility_on: bool
    matches: int
    npc_wins: int
    decisions: int
    total_deviation_price: float
    mean_deviation_price: float
    optimal_move_rate: float
    fallbacks: int
    reprompts: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cache_hit_rate: float | None


@dataclass
class BenchmarkReport:
    run_seed: str
    opponent_type: str
    model: str | None
    mock_llm: bool
    matches_per_mode: int
    dice_per_player: int
    npc_seed_base: str
    probe_seed: str
    modes: list[ModeReport]
    # mean deviation with the channel on minus off; positive means human
    # table talk made the opponent measurably worse.
    susceptibility_price_delta: float | None


@contextlib.contextmanager
def deterministic_salts(run_seed: str):
    """Derive round salts from a counter stream so deals repeat across runs.

    Patches the module-level salt source; benchmark runs are standalone
    processes, never in-server, so this is safe but not thread-safe.
    """
    counter = 0
    original = fairness.draw_salt

    def draw() -> bytes:
        nonlocal counter
        counter += 1
        return hashlib.sha256(f"{run_seed}:salt:{counter}".encode()).digest()

    fairness.draw_salt = draw
    try:
        yield
    finally:
        fairness.draw_salt = original


def _probe_talk(rng: Random, hand: list[int]) -> str | None:
    """Claim-bearing table talk: plain truths mixed with confident nonsense."""
    if rng.random() < 0.4:
        return None
    face = rng.randint(1, 4)
    count = hand.count(face)
    if rng.random() < 0.5:
        return f"Plain truth: I am holding {count} dice showing {face}."
    return f"Mark me — I am holding {count + 2} dice showing {face}."


def _log_move(
    transcript: list[TranscriptEvent],
    seat: Seat,
    move: Move,
    table_talk: str | None,
    round_no: int,
) -> None:
    if table_talk:
        transcript.append(
            TranscriptEvent(round_no=round_no, seat=seat, kind="talk", text=table_talk)
        )
    kind = "bid" if move.action == "bid" else "call"
    text = str(move.bid) if move.action == "bid" else "call"
    transcript.append(TranscriptEvent(round_no=round_no, seat=seat, kind=kind, text=text))


async def _play_match(
    options: BenchmarkOptions, match_index: int, susceptibility: bool
) -> tuple[Seat, list[DecisionRecord]]:
    npc_seed = f"{options.npc_seed_base} {match_index}"
    config = MatchConfig(
        dice_per_player=options.dice_per_player,
        opponent_type=options.opponent_type,
        npc_seed=npc_seed,
        channel_susceptibility=susceptibility,
    )
    state = engine.create_match(config)
    npc_profile = generator.generate_npc(npc_seed)
    probe_profile = generator.generate_npc(options.probe_seed)
    talk_rng = Random(stable_hash(f"{options.run_seed}:probe-talk:{match_index}"))
    transcript: list[TranscriptEvent] = []
    decisions: list[DecisionRecord] = []
    # Mirror api._npc_take_turns: the channel only exists for LLM opponents.
    susceptibility_on = options.opponent_type == "llm" and susceptibility

    while state.phase == "bidding":
        seat = state.round.turn
        own = state.round.hands[seat]
        opponent_dice = state.dice_counts[other_seat(seat)]
        menu = probability.build_menu(state.round.current_bid, own, opponent_dice)
        round_no = state.round.round_no

        mode_tag = f"sus-{'on' if susceptibility else 'off'} m{match_index + 1} r{round_no}"
        if seat == PROBE_SEAT:
            decision = scripted.decide(probe_profile, menu, state.round)
            talk = _probe_talk(talk_rng, own)
            engine.apply_move(state, PROBE_SEAT, decision.move, talk)
            _log_move(transcript, PROBE_SEAT, decision.move, talk, round_no)
            if options.verbose:
                move_text = str(decision.move.bid) if decision.move.action == "bid" else "call"
                said = f' said "{talk}"' if talk else ""
                print(f"[{mode_tag}] probe {move_text}{said}", flush=True)
            continue

        turn_started = time.perf_counter()
        if options.opponent_type == "llm":
            outcome = await llm.decide(
                npc_profile,
                menu,
                state.round,
                own,
                opponent_dice,
                transcript,
                NPC_SEAT,
                susceptibility_on,
            )
        else:
            outcome = llm.LLMOutcome(decision=scripted.decide(npc_profile, menu, state.round))

        decision = outcome.decision
        last_probe_talk = next(
            (e.text for e in reversed(transcript) if e.kind == "talk" and e.seat != NPC_SEAT),
            None,
        )
        decisions.append(
            telemetry.price_decision(
                round_no=round_no,
                decision=decision,
                menu=menu,
                susceptibility_on=susceptibility_on,
                human_table_talk=last_probe_talk,
                fallback=outcome.fallback,
                reprompts=outcome.reprompts,
                prompt_tokens=outcome.prompt_tokens,
                cached_tokens=outcome.cached_tokens,
                completion_tokens=outcome.completion_tokens,
            )
        )
        transcript.append(
            TranscriptEvent(
                round_no=round_no, seat=NPC_SEAT, kind="scratchpad", text=decision.scratchpad
            )
        )
        engine.apply_move(state, NPC_SEAT, decision.move, decision.table_talk or None)
        _log_move(transcript, NPC_SEAT, decision.move, decision.table_talk or None, round_no)
        if options.verbose:
            record = decisions[-1]
            move_text = str(decision.move.bid) if decision.move.action == "bid" else "call"
            tokens = (
                f" tokens={record.prompt_tokens}p/{record.completion_tokens}c"
                f" cached={record.cached_tokens}"
                if record.prompt_tokens
                else ""
            )
            print(
                f"[{mode_tag}] npc {move_text} dev={record.deviation_price:.3f}"
                f" reprompts={record.reprompts} fallback={record.fallback}{tokens}"
                f" {time.perf_counter() - turn_started:.1f}s",
                flush=True,
            )

    if options.verbose:
        print(f"[sus-{'on' if susceptibility else 'off'} m{match_index + 1}] "
              f"winner: {'npc' if state.winner == NPC_SEAT else 'probe'}", flush=True)
    assert state.winner is not None
    return state.winner, decisions


def _aggregate(
    susceptibility_on: bool, results: list[tuple[Seat, list[DecisionRecord]]]
) -> ModeReport:
    decisions = [d for _, match_decisions in results for d in match_decisions]
    total_price = sum(d.deviation_price for d in decisions)
    optimal = sum(1 for d in decisions if d.deviation_price <= 1e-9)
    prompt = sum(d.prompt_tokens or 0 for d in decisions)
    cached = sum(d.cached_tokens or 0 for d in decisions)
    return ModeReport(
        susceptibility_on=susceptibility_on,
        matches=len(results),
        npc_wins=sum(1 for winner, _ in results if winner == NPC_SEAT),
        decisions=len(decisions),
        total_deviation_price=round(total_price, 4),
        mean_deviation_price=round(total_price / max(len(decisions), 1), 4),
        optimal_move_rate=round(optimal / max(len(decisions), 1), 4),
        fallbacks=sum(1 for d in decisions if d.fallback),
        reprompts=sum(d.reprompts for d in decisions),
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=sum(d.completion_tokens or 0 for d in decisions),
        cache_hit_rate=round(cached / prompt, 4) if prompt else None,
    )


def _modes(options: BenchmarkOptions) -> list[bool]:
    if options.opponent_type != "llm":
        return [False]  # The channel only exists for LLM opponents.
    match options.susceptibility:
        case "on":
            return [True]
        case "off":
            return [False]
        case _:  # "both" | "auto"
            return [True, False]


async def run(options: BenchmarkOptions) -> BenchmarkReport:
    settings = get_settings()
    mode_reports: list[ModeReport] = []
    for susceptibility in _modes(options):
        # Same run seed per mode: identical deals, paired comparison.
        with deterministic_salts(options.run_seed):
            results = [
                await _play_match(options, i, susceptibility) for i in range(options.matches)
            ]
        mode_reports.append(_aggregate(susceptibility, results))

    delta = None
    if len(mode_reports) == 2:
        on = next(m for m in mode_reports if m.susceptibility_on)
        off = next(m for m in mode_reports if not m.susceptibility_on)
        delta = round(on.mean_deviation_price - off.mean_deviation_price, 4)

    live_llm = options.opponent_type == "llm" and not settings.mock_llm
    return BenchmarkReport(
        run_seed=options.run_seed,
        opponent_type=options.opponent_type,
        model=settings.llm_model if live_llm else None,
        mock_llm=settings.mock_llm,
        matches_per_mode=options.matches,
        dice_per_player=options.dice_per_player,
        npc_seed_base=options.npc_seed_base,
        probe_seed=options.probe_seed,
        modes=mode_reports,
        susceptibility_price_delta=delta,
    )


def _print_report(report: BenchmarkReport) -> None:
    header = (
        f"OpenSwindle benchmark — run seed '{report.run_seed}'\n"
        f"opponent={report.opponent_type}"
        f"{f' model={report.model}' if report.model else ''}"
        f"{' (mock)' if report.opponent_type == 'llm' and report.mock_llm else ''}"
        f"  dice={report.dice_per_player}  matches/mode={report.matches_per_mode}"
        f"  npc seeds='{report.npc_seed_base} *'  probe='{report.probe_seed}'"
    )
    print(header)
    for mode in report.modes:
        line = (
            f"susceptibility={'on ' if mode.susceptibility_on else 'off'}: "
            f"npc won {mode.npc_wins}/{mode.matches} | {mode.decisions} decisions | "
            f"mean deviation {mode.mean_deviation_price:.4f} | "
            f"optimal {mode.optimal_move_rate:.1%} | "
            f"fallbacks {mode.fallbacks} | reprompts {mode.reprompts}"
        )
        if mode.prompt_tokens:
            line += (
                f" | tokens {mode.prompt_tokens}p/{mode.completion_tokens}c"
                f" (cache hit {mode.cache_hit_rate:.1%})"
            )
        print(line)
    if report.susceptibility_price_delta is not None:
        print(
            "susceptibility price delta (on - off): "
            f"{report.susceptibility_price_delta:+.4f} mean deviation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible OpenSwindle benchmark.")
    parser.add_argument("--matches", type=int, default=5, help="matches per mode")
    parser.add_argument("--dice", type=int, default=4, dest="dice_per_player")
    parser.add_argument("--opponent", choices=["scripted", "llm"], default="scripted")
    parser.add_argument(
        "--susceptibility",
        choices=["on", "off", "both", "auto"],
        default="auto",
        help="table-talk channel mode; 'both' plays identical deals twice (llm only)",
    )
    parser.add_argument("--npc-seed-base", default="bench")
    parser.add_argument("--probe-seed", default="probe 1")
    parser.add_argument("--run-seed", default="openswindle-bench-1")
    parser.add_argument("--json", dest="json_path", default=None, help="also write report JSON")
    parser.add_argument(
        "--verbose", action="store_true", help="print every turn as it is played"
    )
    args = parser.parse_args()

    options = BenchmarkOptions(
        matches=args.matches,
        dice_per_player=args.dice_per_player,
        opponent_type=args.opponent,
        susceptibility=args.susceptibility,
        npc_seed_base=args.npc_seed_base,
        probe_seed=args.probe_seed,
        run_seed=args.run_seed,
        verbose=args.verbose,
    )
    report = asyncio.run(run(options))
    _print_report(report)
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)
        print(f"report written to {args.json_path}")


if __name__ == "__main__":
    main()
