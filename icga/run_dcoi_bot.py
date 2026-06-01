"""Launch the autonomous TD-Gammon bot on the Discord Computer Olympiad Interface.

This wires :class:`icga.dcoi_player.TDGammonAI` into the DCOI framework's
``bot_starting`` launcher and runs it as a Discord bot that plays backgammon games
end-to-end with no human intervention.

Prerequisites (one-time):
  * Install the framework deps (it targets Python 3.10):
        pip install -U discord.py numpy aiofiles pexpect
    plus our own deps (torch, open_spiel) so the bot can think.
  * Fill in ``discord_interface/parameters.conf``:
        OWNER_ID                      = <your discord user id>
        PLAYER_BOT_DISCORD_TOKEN      = <the bot token>
        BETA_TEST_MODE                = True/False  (which guild id is used)
    (See the DCOI "automatic play" quick-start PDF for creating the bot + token.)

Usage (from anywhere; this script cd's into discord_interface/ itself):
        python -m icga.run_dcoi_bot
        python -m icga.run_dcoi_bot --engine search --ply 1
        python -m icga.run_dcoi_bot --player-number 2   # run a second bot

Options are passed to the player via environment variables, which survive the
framework's between-games re-initialisation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DCOI_DIR = PROJECT_ROOT / "discord_interface"


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options, set up the framework's CWD/path, and launch the Discord bot."""
    parser = argparse.ArgumentParser(description="Run the autonomous TD-Gammon DCOI Discord bot.")
    parser.add_argument("--engine", choices=["td", "search"], default=None,
                        help="move engine (default: td / 1-ply greedy).")
    parser.add_argument("--model", default=None, help="path to a .pt model checkpoint.")
    parser.add_argument("--ply", type=int, default=None, help="search depth when --engine search.")
    parser.add_argument("--player-number", type=int, default=1,
                        help="1 uses PLAYER_BOT_DISCORD_TOKEN, 2 uses PLAYER_BOT_2_DISCORD_TOKEN, etc.")
    args = parser.parse_args(argv)

    # Pass per-run options to TDGammonAI via the environment (survives reset()).
    if args.engine is not None:
        os.environ["ICGA_ENGINE"] = args.engine
    if args.model is not None:
        os.environ["ICGA_MODEL"] = args.model
    if args.ply is not None:
        os.environ["ICGA_PLY"] = str(args.ply)

    if not DCOI_DIR.is_dir():
        parser.error(f"discord_interface not found at {DCOI_DIR}")

    # The framework reads parameters.conf and writes logs relative to the CWD, and
    # imports `discord_interface.*` (so its parent — the project root — must be importable).
    sys.path.insert(0, str(PROJECT_ROOT))
    (DCOI_DIR / "log" / "error_handling").mkdir(parents=True, exist_ok=True)
    os.chdir(DCOI_DIR)

    # Imported here, after sys.path is set, so a missing discord.py gives a clear error.
    from discord_interface.player.model.bot_launcher import numbered_bot_starting  # noqa: PLC0415
    from icga.dcoi_player import TDGammonAI  # noqa: PLC0415

    print(f"[run_dcoi_bot] starting TD-Gammon DCOI bot (player {args.player_number}) "
          f"from {DCOI_DIR}")
    numbered_bot_starting(TDGammonAI, player_number=args.player_number)  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
