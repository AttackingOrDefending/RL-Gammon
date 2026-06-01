"""Automated ICGA driver.

Takes a list of moves in ICGA format (dice rolls, checker moves and passes for both
players, in play order), converts each one to the internal OpenSpiel action, replays
them on a fresh game state and asks the bot for the next move.

Two modes:

* **batch** (default): feed the whole transcript and the tool prints the internal
  action list plus, if it is the bot's turn at the end, the bot's recommended move
  in ICGA format::

      python3 -m icga.auto_play --file icga/example_game.txt
      python3 -m icga.auto_play --moves "3-1, P19-3-P19-1, 3-2"

* **interactive**: a REPL that prompts for each dice roll / opponent move and replies
  with the bot's move when it is the bot's turn::

      python3 -m icga.auto_play --interactive --side 0

Run from the project root so that ``rlgammon`` and ``utils`` are importable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pyspiel  # type: ignore[import-not-found]

from icga.icga_format import (
    WHITE,
    apply_icga_move,
    dice_to_chance_action,
    is_dice_token,
    parse_dice,
    turn_to_icga,
)

if TYPE_CHECKING:
    from icga.bot import ICGABot

GAME_NAME = "backgammon(scoring_type=full_scoring)"


def tokenize(raw: str) -> list[str]:
    """Split a raw transcript into ICGA tokens.

    Tolerant of several layouts: one token per line, comma / space / semicolon
    separated, ``#`` comments, and referee-style lines such as
    ``"2. player 0 : P19-3-P19-1"`` (the part after the last ``:`` is kept).
    """
    tokens: list[str] = []
    for raw_line in raw.replace(";", "\n").splitlines():
        line = raw_line.split("#", 1)[0].strip()  # drop full-line and inline comments
        if not line:
            continue
        if ":" in line:  # "1. chance : 3-1" / "player 0 : P24-2-P6-3"
            line = line.split(":")[-1].strip()
        tokens.extend(
            part for part in line.replace(",", " ").split()
            if part.lower() == "pass" or "P" in part.upper() or is_dice_token(part)
        )
    return tokens


class GameDriver:
    """Holds a pyspiel state and replays ICGA tokens / queries the bot."""

    def __init__(self, engine: str = "td", model: str | None = None, ply: int = 1) -> None:
        """Create a fresh game state and remember how to build the bot lazily.

        :param engine: move-selection engine forwarded to the bot (``"td"`` or ``"search"``).
        :param model: path to a ``.pt`` model, or ``None`` for the default checkpoint.
        :param ply: search depth forwarded to the bot when it is constructed.
        """
        self.game = pyspiel.load_game(GAME_NAME)
        self.state = self.game.new_initial_state()
        self.dice: tuple[int, int] | None = None  # roll currently in effect
        self.internal: list[int] = []             # internal actions applied so far
        self._engine = engine
        self._model = model
        self._ply = ply
        self._bot: ICGABot | None = None

    @property
    def bot(self) -> ICGABot:
        """Lazily construct the bot (loading torch + the model) on first use."""
        if self._bot is None:
            from icga.bot import DEFAULT_MODEL, ICGABot  # noqa: PLC0415
            self._bot = ICGABot(self._model or DEFAULT_MODEL, engine=self._engine, search_ply=self._ply)
        return self._bot

    def feed_token(self, token: str) -> None:
        """Convert one ICGA token to internal action(s) and apply it."""
        if self.state.is_terminal():
            msg = f"Game already over; cannot apply {token!r}"
            raise ValueError(msg)

        if self.state.is_chance_node():
            if not is_dice_token(token):
                msg = f"Expected a dice roll (e.g. '3-1') but got {token!r}"
                raise ValueError(msg)
            dice = parse_dice(token)
            action = dice_to_chance_action(self.state, dice)
            self.state.apply_action(action)
            self.dice = dice
            self.internal.append(action)
        else:
            if is_dice_token(token):
                msg = f"Expected a checker move but got dice {token!r}"
                raise ValueError(msg)
            if self.dice is None:
                msg = "No dice rolled yet; transcript must start with a roll"
                raise ValueError(msg)
            applied = apply_icga_move(self.state, token, self.dice)
            self.internal.extend(applied)

    def bot_play(self) -> tuple[str, float, int]:
        """Let the bot play the current turn.

        Returns ``(icga_token, side_evaluation, player)`` where ``side_evaluation`` is
        the value from the moving player's own perspective (positive = good for them).
        """
        if self.state.is_chance_node() or self.state.is_terminal():
            msg = "It is not a decision node; the bot cannot move now"
            raise ValueError(msg)
        if self.dice is None:
            msg = "No dice rolled yet; cannot play"
            raise ValueError(msg)
        player = self.state.current_player()
        actions, white_eval = self.bot.play_turn(self.state)
        self.internal.extend(actions)
        side_eval = white_eval if player == WHITE else -white_eval
        return turn_to_icga(actions, self.dice, player), side_eval, player

    def status(self) -> str:
        """Return a one-line human-readable summary of the current game state."""
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = "player 0 (X)" if returns[0] > 0 else "player 1 (O)"
            return f"GAME OVER — {winner} wins (returns {returns})"
        if self.state.is_chance_node():
            return "waiting for a DICE ROLL"
        return f"decision node — player {self.state.current_player()} to move"


# --------------------------------------------------------------------------- #
# Modes                                                                       #
# --------------------------------------------------------------------------- #
def run_batch(driver: GameDriver, tokens: list[str], play_bot: bool = True) -> None:
    """Replay all tokens, then (optionally) print the bot's move."""
    try:
        for token in tokens:
            driver.feed_token(token)
    except ValueError as exc:
        # feed_token's messages already name the offending token; add running context.
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print(f"Internal actions so far: {driver.internal}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Converted {len(tokens)} ICGA tokens -> {len(driver.internal)} internal actions.")
    print(f"Internal moves = {driver.internal}")
    print(driver.state)
    print(driver.status())

    if not play_bot:
        return
    if driver.state.is_terminal() or driver.state.is_chance_node():
        return

    token, side_eval, player = driver.bot_play()
    print("\n================ BOT MOVE ================")
    print(f"Player       : {player}")
    print(f"ICGA move    : {token}")
    print(f"Evaluation   : {side_eval:.4f} (player {player}'s perspective; positive = good)")
    print(f"Internal now : {driver.internal}")
    print(driver.state)
    print(driver.status())


def run_interactive(driver: GameDriver, bot_side: int | None) -> None:
    """REPL: prompt for rolls / opponent moves, reply with the bot's move."""
    if bot_side is None:
        ans = input("Which side does the bot play? [0 = first/bar P0, 1 = second/bar P25]: ").strip()
        bot_side = int(ans) if ans in {"0", "1"} else 0

    print(f"\nBot plays player {bot_side}. Enter rolls as 'a-b', moves in ICGA format, "
          "or 'q' to quit.\n")
    print(driver.state)

    while True:
        if driver.state.is_terminal():
            print("\n" + driver.status())
            return

        if driver.state.is_chance_node():
            raw = input("Dice roll (a-b): ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                return
            try:
                driver.feed_token(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {exc}")
                continue
            print(f"  rolled {driver.dice}  (internal {driver.internal[-1]})")
            continue

        # Decision node.
        if driver.state.current_player() == bot_side:
            token, side_eval, _player = driver.bot_play()
            print(f"\n>>> BOT MOVE: {token}   (eval {side_eval:.4f}, internal {driver.internal})\n")
            print(driver.state)
        else:
            raw = input("Opponent move (ICGA, or 'pass'): ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                return
            try:
                driver.feed_token(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] {exc}")
                continue
            print(f"  applied (internal {driver.internal})")
            print(driver.state)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the ICGA auto-play driver."""
    parser = argparse.ArgumentParser(description="Automate ICGA <-> internal conversion and call the bot.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file", help="path to a transcript file of ICGA tokens")
    src.add_argument("--moves", help="inline ICGA tokens (comma/space separated)")
    parser.add_argument("--interactive", action="store_true", help="run the interactive REPL instead of batch")
    parser.add_argument("--side", type=int, choices=[0, 1], default=None,
                        help="which player the bot controls (interactive mode)")
    parser.add_argument("--engine", choices=["td", "search"], default="td",
                        help="move-selection engine (default: td / 1-ply greedy)")
    parser.add_argument("--model", default=None, help="path to a .pt model (default: most-trained good_models)")
    parser.add_argument("--ply", type=int, default=1, help="search depth when --engine search")
    parser.add_argument("--no-bot", action="store_true", help="batch: only convert, do not call the bot")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the driver in interactive or batch mode."""
    args = build_arg_parser().parse_args(argv)
    driver = GameDriver(engine=args.engine, model=args.model, ply=args.ply)

    if args.interactive:
        run_interactive(driver, args.side)
        return

    if args.file:
        with Path(args.file).open(encoding="utf-8") as fh:
            raw = fh.read()
    elif args.moves:
        raw = args.moves
    else:
        if sys.stdin.isatty():
            build_arg_parser().error("provide --file, --moves, --interactive, or pipe a transcript on stdin")
        raw = sys.stdin.read()

    tokens = tokenize(raw)
    if not tokens:
        build_arg_parser().error("no ICGA tokens found in the input")
    run_batch(driver, tokens, play_bot=not args.no_bot)


if __name__ == "__main__":
    main()
