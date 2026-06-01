# ICGA tournament tooling

Utilities for playing our backgammon bot in the **ICGA Computer Olympiad** via the
Discord Computer Olympiad Interface (DCOI). The referee speaks the *ICGA move format*;
our bot speaks OpenSpiel / `pyspiel` *internal action integers*. These tools convert
between the two automatically and call the bot.

This replaces the old manual converters (now archived in [`old/`](old/)), which
required typing every dice roll and move into an interactive prompt one at a time and
had a few latent bugs (e.g. the `(n)` repeat notation was silently mangled).

## Files

| File | What it is |
|------|------------|
| [`icga_format.py`](icga_format.py) | Core, dependency-light conversion library (ICGA ⇄ internal). No torch. |
| [`bot.py`](bot.py) | Loads a trained agent and plays a full turn (handles doubles). |
| [`auto_play.py`](auto_play.py) | Offline CLI driver — **batch** and **interactive** modes. |
| [`dcoi_player.py`](dcoi_player.py) | **Fully autonomous** Discord player (plugs into the DCOI framework). |
| [`run_dcoi_bot.py`](run_dcoi_bot.py) | Launcher for the autonomous Discord bot. |
| [`example_game.txt`](example_game.txt) | A sample ICGA transcript for the batch demo. |
| `old/` | The previous, manual utilities (kept for reference). |

Run everything **from the project root** so `rlgammon` and `utils` are importable, and
under the WSL/Linux Python that has `open_spiel` + `torch` installed.

## The ICGA move format (quick reference)

* Dice roll: `"a-b"` e.g. `5-6`.
* Two checkers: `"P<p1>-<d1>-P<p2>-<d2>"` e.g. `P1-6-P12-5`.
* One checker: `"P<p>-<d>"` e.g. `P19-6`.
* Doubles: `"<die>-P<p1>-P<p2>-..."` (1–4 positions) e.g. `2-P1-P12-P17-P19`.
* No legal move: `"pass"`.
* Bars/bear-off are player-relative: **player 0**'s bar is `P0` (bears off toward `P25`),
  **player 1**'s bar is `P25` (bears off toward `P0`).

## Batch mode

Feed the whole transcript-so-far (all dice rolls and moves for both players, in play
order). The tool converts each token to the internal action, replays it, and — if it is
the bot's turn at the end — prints the bot's recommended move in ICGA format.

```bash
# from a file
python3 -m icga.auto_play --file icga/example_game.txt

# or inline (comma/space separated)
python3 -m icga.auto_play --moves "4-3, P12-3-P12-4, 3-6, P24-6-P6-3, 3-3"

# convert only, don't call the bot
python3 -m icga.auto_play --no-bot --moves "4-3, P12-3-P12-4"
```

The transcript parser is tolerant: blank lines, `#` comments (whole-line or inline) and
referee-style lines such as `2. player 0 : P12-3-P12-4` are all accepted.

Typical loop during a game: keep appending the latest dice/opponent move to your
transcript, re-run, play the move the tool prints, repeat.

## Interactive mode

A REPL that prompts for each roll / opponent move and replies with the bot's move when
it is the bot's turn.

```bash
python3 -m icga.auto_play --interactive --side 0   # bot plays player 0
```

`--side 0` = bot is the first player (bar `P0`); `--side 1` = second player (bar `P25`).
Omit `--side` to be asked once at the start. Enter `q` at any prompt to quit.

## Options

| Flag | Meaning |
|------|---------|
| `--file PATH` / `--moves "…"` | transcript source (batch). Also reads stdin if piped. |
| `--interactive` | run the REPL instead of batch. |
| `--side {0,1}` | which player the bot controls (interactive). |
| `--engine {td,search}` | move engine. `td` = 1-ply greedy (default, fast); `search` = expectimax. |
| `--model PATH` | model checkpoint (default: most-trained `good_models` net). |
| `--ply N` | search depth when `--engine search`. |
| `--no-bot` | batch: only convert, do not call the bot. |

## Fully autonomous Discord play (DCOI)

[`dcoi_player.py`](dcoi_player.py) + [`run_dcoi_bot.py`](run_dcoi_bot.py) make the bot
play games on Discord end-to-end with **no human in the loop**, via the
[Discord Computer Olympiad Interface](https://github.com/CohenSolalQuentin/Discord_Computer_Olympiad_Interface)
vendored in [`../discord_interface`](../discord_interface).

`TDGammonAI` subclasses the framework's `BasicPlayer`. On each of our turns the framework
calls `my_plays(game_history, …)`; we replay the ICGA `game_history` onto a pyspiel state
(`icga_format`), let the bot choose, and return the **referee engine's own** matching
legal-move string (so it always passes the framework's `assert move in valid_actions()`).
A safety fallback returns a legal move on any error, so the bot never crashes or forfeits.

Setup (one-time):

1. Install framework deps (it targets Python 3.10) on top of our own (`torch`, `open_spiel`):
   ```bash
   pip install -U discord.py numpy aiofiles pexpect
   ```
2. Fill in [`../discord_interface/parameters.conf`](../discord_interface/parameters.conf):
   `OWNER_ID`, `PLAYER_BOT_DISCORD_TOKEN`, and `BETA_TEST_MODE`. (Creating the Discord bot
   + token is covered by the framework's "automatic play" quick-start PDF.)

Run it (the launcher `cd`s into `discord_interface/` and fixes `sys.path` itself):

```bash
python -m icga.run_dcoi_bot                     # default: td engine, default model
python -m icga.run_dcoi_bot --engine search --ply 1
python -m icga.run_dcoi_bot --player-number 2   # second bot (PLAYER_BOT_2_DISCORD_TOKEN)
python -m icga.run_dcoi_bot --model fresh        # untrained net (smoke-test the plumbing)
```

> **Model compatibility.** The bot needs a `.pt` checkpoint runnable by the *current*
> `rlgammon` agent/model API. If you load an incompatible checkpoint the bot fails loudly
> at startup with a clear message (rather than playing badly), so point `--model` /
> `ICGA_MODEL` at a compatible model. `--model fresh` builds an untrained net, which is
> useful for exercising the Discord/conversion plumbing before a trained model is ready.

Because it needs a live Discord account/token, the network layer can't be unit-tested
here; instead `test/test_dcoi_player.py` drives the **real referee engine headlessly** and
plays full games to completion, exercising the entire pipeline minus the socket.

## How the conversion works

OpenSpiel encodes a (up to) two-checker move as one integer. We decode it exactly
(mirroring `SpielMoveToCheckerMoves`) into `(position, die)` checker moves, so the
conversion is unambiguous and never depends on parsing the human-readable board string.

* **Internal → ICGA** (bot's move out): decode the action(s), map positions to ICGA
  points, preserve OpenSpiel's application order (which is legal by construction and
  keeps chained hops / mandatory bar-entry-first correct). A doubles turn spans two
  OpenSpiel actions and is combined into one `<die>-P…-P…` token.
* **ICGA → internal** (incoming move in): decode every legal action into its ICGA
  checker moves and match the input token's moves as a multiset (with backtracking),
  so any legal ordering is found. Robust to the two equivalent integer encodings
  OpenSpiel sometimes has for the same physical move.

## Validation

`test/test_icga_conversion.py` covers this code:

* unit tests of the pure conversion functions;
* a **round-trip** over real games (internal → ICGA → internal) asserting identical
  resulting boards and observation tensors;
* a **referee-acceptance** test that replays our generated ICGA moves through the actual
  `discord_interface` referee engine and checks every move is in its legal-move list —
  the three completed tournament games in `old/` replay cleanly to the correct winner.

`test/test_dcoi_player.py` additionally plays full **autonomous** games headlessly against
the referee engine (bot vs random and bot vs itself, both sides), asserting every bot move
is engine-legal and games complete — the closest possible test to live Discord play.
