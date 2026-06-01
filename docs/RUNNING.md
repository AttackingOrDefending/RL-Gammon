# Running RL-Gammon

This is the operational guide for the project: how to **train**, **evaluate**, **play the strongest
agent**, run the **world-class feature demos**, and test against **GNU Backgammon** — all on **WSL
Debian**, with copy-pasteable commands and an explanation of every important flag.

> **TL;DR — play the strongest agent now:**
> ```bash
> wsl -d Debian -e bash -lc "cd /mnt/c/Users/panti/PycharmProjects/RL-Gammon && PYTHONPATH=\$PWD python3 -m scripts.play_strong --games 50 --depth 2"
> ```
> See [Recommended strongest setup](#recommended-strongest-setup) for the full rationale and faster
> alternatives.

---

## 1. Environment

The code runs under **WSL Debian** (the Windows Python install lacks `pyspiel`/OpenSpiel, which the
game engine needs). Every command in this doc is meant to be run from the repository root inside WSL.
The canonical wrapper, runnable from a Windows shell, is:

```bash
wsl -d Debian -e bash -lc "cd /mnt/c/Users/panti/PycharmProjects/RL-Gammon && <COMMAND>"
```

Below, `<COMMAND>` is given on its own; wrap it as above when invoking from Windows.

### PYTHONPATH

Scripts are run as modules under the `scripts.*` package (e.g. `python3 -m scripts.play_strong`). For
the `scripts.*` and `rlgammon.*` packages to import, the **repository root must be on `PYTHONPATH`**:

```bash
PYTHONPATH=/mnt/c/Users/panti/PycharmProjects/RL-Gammon python3 -m scripts.<name> ...
# or, from the repo root:
PYTHONPATH=$PWD python3 -m scripts.<name> ...
```

The demos and `eval_vs_random` import only `rlgammon.*` and run from the repo root without the extra
`PYTHONPATH`; setting it always is harmless and recommended.

### GPU

Anything that touches torch (MuZero training/eval, calibrated training) accepts `--device cuda` to use
an NVIDIA GPU, or `--device cpu` to force CPU. Most scripts **auto-detect** CUDA and use it when
available; pass `--device cpu` to override. The TD value net and the strong agent run on CPU and are
not GPU-bound.

### Tests, lint and type-checking

These also run under WSL Debian (`ruff`, `mypy`, `pytest` are installed there):

```bash
python3 -m ruff check --config ruff.toml .      # lint (style, magic numbers, docstrings, ...)
python3 -m mypy .                               # strict static type-check
python3 -m pytest -q                            # the test suite
```

To check a single file/dir (faster, e.g. when iterating):
`python3 -m ruff check --config ruff.toml rlgammon/agents/strong_agent.py` and
`python3 -m mypy rlgammon/agents scripts`.

---

## 2. Training the TD value agent

The TD value network (`TDGammonNet`) is the backbone of both the strong agent and the cube layer. Two
trainers exist.

### 2a. Scalar TD agent — `scripts/train_td.py`

Self-play TD(λ) on a single scalar equity target. Strong checker play, but the win/gammon components
are **not** individually grounded (so it is *not* suitable for cube decisions).

```bash
PYTHONPATH=$PWD python3 -m scripts.train_td --episodes 20000 --eval-every 1000 --eval-games 200 \
    --hidden 128 --lr 0.1 --lamda 0.7 --seed 0 --out td-backgammon
```

| Flag | Meaning |
| --- | --- |
| `--episodes` | number of self-play training episodes (games) |
| `--eval-every` | evaluate vs random every N episodes |
| `--eval-games` | games played per evaluation |
| `--hidden` | hidden-layer width of the value net |
| `--lr` | learning rate |
| `--lamda` | TD(λ) trace-decay parameter (how much distant states count) |
| `--seed` | random seed |
| `--out` | base file name for the saved model (written under `rlgammon/agents/saved_agents/`) |

**Expected result:** reaches ≈100% win-rate vs random within a few thousand episodes.

### 2b. Calibrated TD agent — `scripts/train_td_calibrated.py`

Same self-play, but trains the full **win / gammon / backgammon probability vector** against the
gnubg/XG-style per-outcome target. This is the model you want for **cube and gammon awareness**: its
`o0..o4` components are individually calibrated, so the cube-equity layer reads real probabilities
instead of a gammonless fallback.

```bash
PYTHONPATH=$PWD python3 -m scripts.train_td_calibrated --episodes 20000 --eval-every 1000 \
    --eval-games 200 --hidden 128 --lr 0.1 --lamda 0.7 --seed 0 --device cuda --out td-calibrated
```

Flags as in 2a, plus `--device {cpu,cuda}` (default: benchmark and pick the faster). **Expected
result:** ≈100% vs random *and* calibrated probabilities (verify with `scripts/cube_benchmark.py`,
section 6). The shipped checkpoint
`rlgammon/agents/saved_agents/td-calibrated-077c912f-...-(1500).pt` is the output of this trainer and
is what the strong agent and the cube layer load by default.

---

## 3. Training Stochastic MuZero — `scripts/train_muzero_long.py`

Resumable, long-running self-play training of the Stochastic MuZero network with a batched Gumbel
actor. **This needs many hours to days of GPU time** to become strong.

```bash
PYTHONPATH=$PWD python3 -m scripts.train_muzero_long --device cuda --max-seconds 86400 \
    --parallel 32 --sims 50 --considered 16 --train-steps-per-game 8 \
    --eval-every 200 --eval-games 40 --eval-sims 50 --checkpoint-minutes 15 --self-play batched
```

| Flag | Meaning |
| --- | --- |
| `--device {cpu,cuda}` | torch device (default: cuda if available) |
| `--max-seconds` | wall-clock training budget in seconds |
| `--parallel` | number of parallel self-play games (K) per actor step |
| `--sims` | self-play search simulations per move |
| `--considered` | Gumbel "considered" root actions m |
| `--self-play {batched,single}` | self-play actor: `batched` Gumbel feature actor (default, fast) or `single`-game baseline actor |
| `--train-steps-per-game` | learner gradient steps per completed game (once warm) |
| `--eval-every` / `--eval-games` / `--eval-sims` | evaluate vs random every N games, with this many games and this many search sims |
| `--checkpoint-minutes` | checkpoint the latest network every N minutes |
| `--resume <path>` | resume from a checkpoint |
| `--mcts {gumbel,baseline}` | search algorithm used during training |

There are many additional optimiser/network knobs (`--batch-size --lr --weight-decay
--value-loss-weight --unroll-steps --td-steps --discount --replay-capacity --state-channels --hidden
--codebook-size --support-size --seed`); the defaults are sensible — change them only deliberately.

**Resume:** checkpoints are written to `scripts/muzero_checkpoints/`; the rolling latest is
`scripts/muzero_checkpoints/latest.pt`. Continue a run with:

```bash
PYTHONPATH=$PWD python3 -m scripts.train_muzero_long --device cuda --max-seconds 86400 \
    --resume scripts/muzero_checkpoints/latest.pt
```

> `--self-play batched` is the fast feature path (searches across the K games are batched on the GPU);
> `--self-play single` is the proven one-game-at-a-time baseline, useful for A/B comparison.

---

## 4. Evaluating agents

### 4a. Any agent vs random — `scripts/eval_vs_random.py`

```bash
# TD value agent:
PYTHONPATH=$PWD python3 -m scripts.eval_vs_random --agent td \
    --model td-calibrated-077c912f-18c5-4c02-98a7-8f64254922be-\(1500\).pt --games 200

# MuZero network:
PYTHONPATH=$PWD python3 -m scripts.eval_vs_random --agent muzero --model scripts/muzero_checkpoints/latest.pt \
    --games 200 --sims 50 --mcts gumbel --device cuda
```

| Flag | Meaning |
| --- | --- |
| `--agent {td,muzero}` | which agent to evaluate |
| `--model` | TD: file name within `rlgammon/agents/saved_agents/`; MuZero: path to a state dict |
| `--games` | number of evaluation games (colours alternate, so the win-rate is unbiased) |
| `--sims` | MuZero search simulations per move (ignored for `td`) |
| `--mcts {gumbel,baseline}` | MuZero search: batched Gumbel (default) or single-tree pUCT baseline |
| `--device {cpu,cuda}` | torch device for MuZero inference |
| `--seed` | evaluation RNG seed |
| `--state-channels --hidden --codebook-size` | MuZero architecture (must match the checkpoint) |

Output: `agent=... games=N win_rate=... avg_points=...`. `avg_points` is the mean signed return in
`{-3,-2,-1,+1,+2,+3}`.

### 4b. MuZero long-training checkpoint vs random — `scripts/eval_muzero_checkpoint.py`

Convenience wrapper that reads the architecture straight from a `train_muzero_long` checkpoint:

```bash
PYTHONPATH=$PWD python3 -m scripts.eval_muzero_checkpoint --checkpoint scripts/muzero_checkpoints/latest.pt \
    --games 100 --sims 50 --device cuda --mcts gumbel
```

`--checkpoint` (required), `--games`, `--sims`, `--device`, `--mcts {gumbel,baseline}`, `--seed`.

---

## 5. The strong agent — `scripts/play_strong.py`

The strongest **near-term** configuration, assembled in `rlgammon/agents/strong_agent.py`:

* **value:** the calibrated `TDGammonNet`;
* **leaf evaluator:** a phase-aware `CompositeEvaluator` — RACE/BEAROFF leaves are scored by the
  **exact** analytic bear-off specialist, CONTACT leaves by the value net;
* **search:** `StarMinimax` expectiminimax (2-ply by default, star2 chance-node pruning) — the agent
  "thinks deeper at test time" than the 1-ply greedy training policy;
* **optional rollouts:** wrap the leaf evaluator in a variance-reduced truncated `RolloutEvaluator`
  (stronger, much slower);
* **optional cube:** doubling-cube decisions delegated to a `TDAgent` on the same calibrated net.

```bash
PYTHONPATH=$PWD python3 -m scripts.play_strong --games 50 --depth 2
```

| Flag | What it does |
| --- | --- |
| `--model` | saved-model file name within `rlgammon/agents/saved_agents/` (default: the shipped calibrated checkpoint) |
| `--depth` | expectiminimax search depth in decision plies. `1` = greedy (fast); `2` = 2-ply look-ahead over the opponent's reply (stronger, much slower) |
| `--rollouts` | replace the static net at search leaves with a truncated-rollout evaluator — strongest, but dramatically slower (use with `--depth 1`) |
| `--cube` | build the agent with the doubling-cube decision methods enabled (`should_double`/`should_take`/`cube_action`) |
| `--games` | number of evaluation games vs random (keep **modest** — see the performance note) |
| `--seed` | evaluation RNG seed |

The script prints the strong agent's win-rate/avg-points, the **1-ply greedy baseline** on the same
net and games for contrast, and the lift between them.

> **Performance note — 2-ply is slow.** The search has no candidate-move pruning, so a single 2-ply
> move over backgammon's full per-roll action space (tens to >1000 legal moves, each expanding a
> 21-outcome chance node and the opponent's replies) costs on the order of **~2 minutes per move on
> CPU** from the opening. A full 2-ply game therefore takes well over an hour, so 30–50 games at
> `--depth 2` is not interactive. **For a quick, strong-but-fast run use `--depth 1`** (≈100% vs
> random in seconds); reserve `--depth 2`/`--rollouts` for a small number of games or single-position
> analysis.

### Using the strong agent from Python

```python
from rlgammon.agents.strong_agent import build_strong_agent, StrongAgentConfig
from rlgammon.cube.cube_types import GameMode, MatchContext

agent = build_strong_agent(config=StrongAgentConfig(max_depth=2, use_cube=True,
                                                     match_ctx=MatchContext(GameMode.MONEY)))
action = agent.choose_move(state.legal_actions(), state)   # checker play (search-driven)
double = agent.should_double(state)                        # cube decision (needs use_cube=True)
```

`StrongAgentConfig` also exposes `use_star2`, `use_rollouts`, `rollout_trials`, `rollout_max_depth`
and `rollout_seed`.

---

## 6. World-class feature demos

Each is a runnable, seeded report (`python3 -m scripts.<name>`).

| Script | What it shows |
| --- | --- |
| `scripts/rollout_demo.py` | Static net equity vs truncated-rollout equity; that control-variate variance reduction lowers the standard error at equal trials; and a position where the rollout-guided move differs from the 1-ply greedy move. |
| `scripts/endgame_demo.py` | That the exact analytic bear-off/race specialist is sharper than the value net on disengaged positions. Flags: `--positions --rollouts --seed`. |
| `scripts/cube_benchmark.py` | Probability calibration of the net's components, that cube decisions are match-score-dependent (incl. Crawford/post-Crawford), and which probability path the cube layer used. Flags: `--model --fresh --games --seed`. |
| `scripts/cube_selfplay.py` | Trains/loads a TD net and plays cube-aware self-play matches; compares cube-on vs cube-off. Flags: `--episodes --games --match-length --hidden --lr --lamda --seed --use-cube --eval-baseline --load --save`. |
| `scripts/muzero_ab_demo.py` | A/B demo contrasting the MuZero search variants on a few decision nodes. Flags: `--moves --seed`. |

Example:

```bash
PYTHONPATH=$PWD python3 -m scripts.cube_benchmark --model td-calibrated-077c912f-18c5-4c02-98a7-8f64254922be-\(1500\).pt --games 50
```

---

## 7. Doubling cube and match play

The cube/match layer (`rlgammon/cube/`) is a pure analytic layer over the cubeless game and the value
net. Cube decisions are exposed by the `TDAgent` (`should_double`/`should_take`/`cube_action`) and by
the strong agent when built with `use_cube=True`.

* **Cube on/off:** enable the strong agent's cube methods with `--cube` (CLI) or
  `StrongAgentConfig(use_cube=True)`; checker play is unaffected by the cube flag.
* **Match length / score:** pass a `MatchContext(mode=GameMode.MATCH, match_length=..., my_score=...,
  opp_score=...)`. Money play is `GameMode.MONEY` (match length/score ignored). The cube layer derives
  the away-counts and the Crawford / post-Crawford state automatically. `scripts/cube_selfplay.py`
  exposes `--match-length` (0 = money).
* **MET (match-equity table):** the cube functions default to the **Woolsey-Heinrich** table
  (`rlgammon.cube.met.WOOLSEY_HEINRICH`); pass a different `MET` to `should_double`/`take_decision` to
  override.

> **Calibration caveat — use the calibrated model for the cube.** Cube equity depends on the win /
> gammon / backgammon probabilities. A *scalar*-trained net (`train_td.py`) does not ground those
> components, so its `cube_probs` fall back to a gammonless `[p,0,0,0,0]` vector and the cube
> decisions are weak. **Always use the calibrated checkpoint (section 2b) for cube/match play.**
> `scripts/cube_benchmark.py` reports, per position, whether the raw calibrated vector or the
> gammonless fallback was used.

---

## 8. Testing against GNU Backgammon (gnubg)

The bridge runs an HTTP server **inside** gnubg (`bridge.py`) that the Python side talks to over
`localhost:8001`. See `rlgammon/environment/gnubg/README.md` for the full protocol and caveats.

**1. Start the bridge** (leave it running in the background):

```bash
/usr/games/gnubg -t -q -p rlgammon/environment/gnubg/bridge.py > /tmp/gnubg_bridge.log 2>&1 &
```

**2. Run a test** from a normal Python process (the bridge must already be up):

```python
from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.testing.gnu_testing import GNUTesting

results = GNUTesting(episodes_in_test=1).test(TDAgent(
    pre_made_model_file_name="td-calibrated-077c912f-18c5-4c02-98a7-8f64254922be-(1500).pt"))
print(results)
```

A ready-made end-to-end example is `rlgammon/environment/gnubg/example_gnu_test.py`.

> **Caveats** (from the gnubg README): the bridge must be started *before* any `GNUTesting` call and
> stay running; **port 8001 is hard-coded** on both sides; gnubg embeds its own Python, so `bridge.py`
> runs there; and **do not** clean up with `pkill -f bridge.py` (it would also kill the launching
> shell) — use the documented shutdown instead.

---

## Recommended strongest setup

For the **strongest checker play that is still tractable**, run the strong agent at **2-ply** with the
calibrated net (exact endgames via the composite evaluator, expectiminimax over the opponent's reply).
Keep the game count small because 2-ply is ~minutes/move on CPU:

```bash
wsl -d Debian -e bash -lc "cd /mnt/c/Users/panti/PycharmProjects/RL-Gammon && PYTHONPATH=\$PWD python3 -m scripts.play_strong --games 4 --depth 2 --cube"
```

* For a **fast** strong agent (≈100% vs random in seconds), drop to `--depth 1`.
* For the **absolute strongest** single-position analysis (slowest), add `--rollouts` (best paired
  with `--depth 1`): the truncated, variance-reduced rollout of the calibrated net is the most
  accurate evaluator available.
* For **cube/match play**, keep the calibrated model (the default) and add `--cube`; set the match
  context in Python via `StrongAgentConfig(match_ctx=...)` for match (vs money) decisions.
