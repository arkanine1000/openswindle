# OpenSwindle

[![CI](https://github.com/arkanine1000/openswindle/actions/workflows/ci.yml/badge.svg)](https://github.com/arkanine1000/openswindle/actions/workflows/ci.yml)

An open-source (MIT) backend for **Swindlestones**, the d4 bluffing game from inkle's
adaptation of *Sorcery!* — a close cousin of Liar's Dice. This repo is the authoritative
game server: it deals hands, validates every move, adjudicates calls, proves its own
fairness cryptographically, generates reproducible NPC opponents, and benchmarks how
well (and how honestly) an LLM plays.

A React frontend is developed separately and pointed at this API.

## The game

- Two players, each with a hidden pool of 4-sided dice (faces 1–4). Pools are
  symmetrical and configurable from 2 to 6 dice each (4–12 total).
- Players alternate **bids** of the form `N × face` — a claim that at least N dice of
  that face exist across *both* hidden hands. Every bid must strictly raise the last:
  higher quantity, or same quantity with a higher face. A bid's quantity can never
  exceed the total dice on the board.
- Instead of bidding, a player may **call**. Hands are revealed: if the bid is met or
  exceeded, the caller loses a die; if not, the bidder loses one. The only way a round
  ends is a call.
- The loser of a round opens the next. Lose your last die and the match is over.

## Quickstart

```bash
uv sync
cp .env.example .env          # add your gateway API key, or enable mock mode
uv run uvicorn openswindle.api:app --reload
```

Run the tests (fully offline — no API key needed):

```bash
uv run pytest
```

## Architecture

| Module | Responsibility |
|---|---|
| `engine.py` | Authoritative match FSM: dealing, bid legality, call adjudication, win detection |
| `fairness.py` | SHA-256 commit-reveal and mutual-entropy dealing |
| `probability.py` | Exact binomial truth probabilities for every legal move (server-side only) |
| `npc/generator.py` | Deterministic seed → params → bio → planted tells |
| `npc/scripted.py` | Parameter-driven deterministic policy (mock mode and LLM fallback) |
| `npc/llm.py` | Stateless per-turn LLM decisions via LiteLLM |
| `telemetry.py` | Deviation pricing and the post-match autopsy |
| `api.py` | REST transport (FastAPI); in-memory match store |

Neither the web client nor the LLM holds authoritative state: each receives its own
hand and the public board state every turn, and the server validates everything.

## Cryptographic fairness

At every deal the server publishes a per-hand commitment before any bidding starts:

```
commitment = SHA-256(salt || sorted_hand_bytes)
```

Each die is derived from **both** hands' salts (mutual entropy), so no single salt
controls the outcome:

```
die_i = (SHA-256(salt_a || salt_b || round_no || seat || i) mod 4) + 1
```

Salts are 32 random bytes, drawn fresh **per hand per round** — salting per hand
blocks dictionary attacks against the small hand space. When any round terminates
(call or abort), both hands and both salts are revealed unconditionally. Clients can
audit every round with `fairness.verify_commitment(salt_hex, hand, commitment)` or by
recomputing the two hashes above themselves.

## NPC opponents (seed-to-bio)

Opponents are generated from a shareable seed string (`"seed 4471"`): the seed first
draws four numeric traits — **deception** (bluff frequency), **skepticism** (call
threshold), **aggression** (raise velocity), **chattiness** — and the biography is then
derived *from* those parameters, so the mechanical policy dictates the flavor, never
the reverse. All per-turn randomness is derived from the seed and game position, never
fresh entropy: a character is exactly reproducible for benchmarking, and its *style*
(how bold, how suspicious, how talkative) is learnable across a match without any
scripted triggers that would make it exploitable.

## LLM opponents

LLM NPCs make one stateless structured call per decision through
[LiteLLM](https://docs.litellm.ai) via a gateway-compatible model string. The default
lives in `OPENSWINDLE_LLM_MODEL`. The response schema is enforced:
`scratchpad` (private reasoning), `move`, and `table_talk` (in-character dialogue).
Provider JSON mode is used where supported and dropped automatically where a gateway
rejects it — the prompt contract and parser enforce the schema either way. Illegal or
malformed outputs are rejected and reprompted with the violation explained; after the
retry budget the deterministic scripted policy takes over (flagged in telemetry).

**The LLM is a natural-language reasoner, not a calculator.** It receives the rules,
its character, the full match transcript (moves, table talk, its own past
scratchpads), and its hand — never the probability engine's output. The deterministic
layer is used exclusively server-side, for legality validation and post-mortem
analysis.

Prompts are ordered stable-prefix-first (rules → character → append-only transcript →
current turn) so provider-side implicit prompt caching hits on every consecutive turn.

## Benchmarking & telemetry

- **Channel susceptibility**: whether the human's `table_talk` is included in the LLM
  payload is a toggle (on by default, forced off for human-vs-human and scripted
  opponents). It measures how easily the model is steered by confident nonsense
  attached to true claims.
- **Deviation pricing**: every NPC decision is compared against the mathematically
  optimal move (highest exact truth probability) and the win-probability delta is
  logged.
- **Post-match autopsy** (`GET /matches/{id}/autopsy`, only after the match ends): the
  full scratchpad history, the deviation ledger, reprompt counts, token usage
  (including cached tokens), and the NPC's hidden parameters.

### Benchmark runner

`openswindle-benchmark` plays seeded matches against the NPC layer and aggregates the
telemetry into a report. A scripted probe drives the human seat and mixes plain truths
about its hand with confident nonsense, giving the susceptibility channel something to
bite on; `--susceptibility both` replays identical deals with the channel on and off
and reports the deviation-price delta. A fixed `--run-seed` makes deals, probe
behavior, and NPC seeds fully deterministic.

```bash
uv run openswindle-benchmark --matches 10 --opponent llm --susceptibility both
uv run openswindle-benchmark --matches 5 --opponent scripted --json report.json
```

## API

| Endpoint | Description |
|---|---|
| `POST /matches` | Create a match; returns the creator's seat token and the initial view |
| `POST /matches/{id}/join` | Claim seat `b` in a human-vs-human match |
| `GET /matches/{id}` | Public view for your seat (header `X-Player-Token`) |
| `POST /matches/{id}/moves` | Submit a move (+ optional `table_talk`); NPC replies in the same response |
| `POST /matches/{id}/abort` | End the match and reveal the current round (header `X-Player-Token`) |
| `GET /matches/{id}/npc/profile` | NPC name and bio (params stay hidden until the autopsy) |
| `GET /matches/{id}/autopsy` | Post-match scratchpads, deviation ledger, and NPC reveal |
| `GET /healthz` | Liveness probe |

Example:

```bash
curl -s -X POST localhost:8000/matches \
  -H 'Content-Type: application/json' \
  -d '{"config": {"dice_per_player": 4, "opponent_type": "llm", "npc_seed": "seed 4471"}}'

curl -s -X POST localhost:8000/matches/<id>/moves \
  -H 'Content-Type: application/json' -H 'X-Player-Token: <token>' \
  -d '{"move": {"action": "bid", "bid": {"quantity": 2, "face": 3}}, "table_talk": "I never lie."}'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VERCEL_AI_GATEWAY_API_KEY` | — | Gateway credential for LLM opponents |
| `OPENSWINDLE_LLM_MODEL` | configured default | Any LiteLLM model string |
| `OPENSWINDLE_MOCK_LLM` | `false` | Use the scripted policy instead of an LLM (offline dev/tests) |
| `OPENSWINDLE_LLM_EXTRA_BODY` | — | JSON merged into every LLM request (e.g. disable provider thinking mode) |
| `OPENSWINDLE_JSON_MODE_UNSUPPORTED_MODELS` | configured default | Models to skip provider JSON mode for up front |
| `OPENSWINDLE_CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins (comma-separated) |
| `OPENSWINDLE_FINISHED_MATCH_TTL_SECONDS` | `3600` | Finished-match retention time in memory |
| `OPENSWINDLE_MAX_FINISHED_MATCHES` | `1000` | Maximum finished matches retained before pruning oldest |

Match state is held in memory; matches do not survive a server restart. Finished matches
are pruned by TTL and by the maximum retained-match cap.

## Attribution

Swindlestones comes from [inkle](https://www.inklestudios.com/)'s *Sorcery!*
video-game series, where it appears as a dice-bluffing minigame of the studio's own
design. The adaptation is itself built on the *Sorcery!* gamebooks (1983–85) by
**Steve Jackson**, co-creator of Fighting Fantasy and co-founder of Games Workshop.

OpenSwindle is an independent, unofficial re-implementation of the game's *rules*,
made with affection for both. It is not affiliated with, sponsored by, or endorsed by
inkle or Steve Jackson, and it uses no assets, text, or code from their works. If you
enjoy this game, play [Sorcery!](https://www.inklestudios.com/sorcery/) — it's
wonderful.

## License

[MIT](LICENSE)
