"""Integration test for the autonomous DCOI player (icga.dcoi_player).

Drives the real DCOI BackgammonDiscord referee engine headlessly (no Discord): our
TDGammonAI plays full games against a random legal player and against itself, and we
assert every move it produces is accepted by the engine and the games complete.

Uses a fresh (untrained) net via ICGA_MODEL=fresh, so it validates the *pipeline*
(history reconstruction -> bot -> engine-legal move) independently of whether a
trained checkpoint compatible with the current rlgammon value model is available.

Skipped automatically where the DCOI framework deps (discord.py) or pyspiel are
absent (e.g. CI installing only requirements.txt).
"""
import asyncio
from math import inf
import os
import random
from typing import ClassVar

import pytest
import torch as th

os.environ.setdefault("ICGA_MODEL", "fresh")  # untrained net -> no model file needed

# Skip the whole module cleanly when the DCOI framework deps or pyspiel are absent.
pytest.importorskip("pyspiel")
pytest.importorskip("discord")
try:
    from discord_interface.utils.mytime import Time
    from icga.dcoi_player import TDGammonAI
except Exception as exc:  # noqa: BLE001 - framework not importable in this env
    pytest.skip(f"DCOI framework not importable: {exc}", allow_module_level=True)

# Default per-game safety cap on the number of plies before a game is abandoned.
MAX_PLIES = 1500
# Sentinel current-player value the referee returns when it is its own turn to roll dice.
REFEREE_TURN = -1


class CountingAI(TDGammonAI):
    """A TDGammonAI that counts how often the safety fallback fires (0 means clean play)."""

    fallbacks: ClassVar[int] = 0

    def my_plays(self, game_history: list[str], time_left: float = inf,
                 opponent_time_left: float = inf) -> str:
        """Choose a move, counting (and using) the fallback whenever reconstruction fails."""
        del time_left, opponent_time_left  # the pipeline test runs untimed (fast path)
        move = None
        try:
            move = self._choose(game_history)
        except Exception:  # noqa: BLE001 - the fallback must absorb any failure
            move = None
        if move is None:
            type(self).fallbacks += 1
            return self._fallback()
        return move


async def _play(bot_side: int | str, seed: int, max_plies: int = MAX_PLIES) -> CountingAI:
    """Play one headless game to completion and return the player that played it."""
    random.seed(seed)
    # Seed torch too: the fresh (untrained) net is built from torch's global RNG, so without
    # this the bot's move choice -- and hence whether the fallback fires -- depends on whatever
    # RNG state earlier tests leave behind, making the suite-level run flaky.
    th.manual_seed(seed)
    player = CountingAI()
    player.update_game("backgammon")
    player.start()
    game = player.game
    plies = 0
    while not game.ended() and plies < max_plies:
        plies += 1
        cp = game.get_current_player()
        is_bot_turn = cp != REFEREE_TURN and bot_side in ("self", cp)
        if is_bot_turn:
            # plays() asserts the move is legal (move in valid_actions) and applies it.
            await player.plays(Time(minute=30), Time(minute=30))
        else:
            # The referee (dice) or the random opponent plays a random legal action.
            action = random.choice(list(game.valid_actions()))
            player.opponent_plays(action)  # type: ignore[no-untyped-call]
    return player


@pytest.mark.parametrize("bot_side", [0, 1, "self"])
def test_autonomous_player_completes_legal_games(bot_side: int | str) -> None:
    """The bot plays only engine-legal moves and games run to completion."""
    CountingAI.fallbacks = 0
    player = asyncio.run(_play(bot_side, seed=0))
    game = player.game
    assert game.ended(), "game did not complete"
    assert game.winner is not None
    assert CountingAI.fallbacks == 0, f"bot hit the safety fallback {CountingAI.fallbacks} times"
