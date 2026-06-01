"""The only module that imports `pyspiel`; wraps OpenSpiel behind the game protocols.

The import is guarded so this module (and the whole ``rlgammon.game`` package) can be imported
even where OpenSpiel is unavailable (e.g. native Windows); constructing :class:`OpenSpielGame`
then raises :class:`EngineNotAvailableError`.
"""

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.game_errors.game_errors import EngineNotAvailableError

try:
    import pyspiel  # type: ignore[import-not-found]

    _OPENSPIEL_AVAILABLE = True
except ImportError:
    _OPENSPIEL_AVAILABLE = False

DEFAULT_GAME_STRING = "backgammon(scoring_type=full_scoring)"


def is_openspiel_available() -> bool:
    """Return whether the OpenSpiel (`pyspiel`) backend can be imported in this environment."""
    return _OPENSPIEL_AVAILABLE


class OpenSpielGame:
    """Thin wrapper around a loaded `pyspiel` backgammon game that satisfies the BackgammonGame protocol."""

    def __init__(self, game_string: str = DEFAULT_GAME_STRING) -> None:
        """
        Load the underlying OpenSpiel game.

        :param game_string: the OpenSpiel game string to load
        :raises EngineNotAvailableError: if OpenSpiel (`pyspiel`) is not importable
        """
        if not _OPENSPIEL_AVAILABLE:
            raise EngineNotAvailableError
        self._game = pyspiel.load_game(game_string)

    def new_initial_state(self) -> GameState:
        """Return a fresh initial state (a chance node for the opening roll)."""
        state: GameState = self._game.new_initial_state()
        return state

    def num_distinct_actions(self) -> int:
        """Return the size of the unified action id space (1352 for full-scoring backgammon)."""
        return int(self._game.num_distinct_actions())
