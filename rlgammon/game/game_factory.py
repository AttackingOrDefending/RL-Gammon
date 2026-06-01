"""Factory for constructing a backgammon game backend from a PossibleEngine selector."""

from rlgammon.game.backgammon_protocol import BackgammonGame
from rlgammon.game.game_errors.game_errors import WrongEngineTypeError
from rlgammon.game.game_types import PossibleEngine
from rlgammon.game.mock_game import MockGame
from rlgammon.game.openspiel_adapter import DEFAULT_GAME_STRING, OpenSpielGame


def create_game(engine: PossibleEngine = PossibleEngine.OPEN_SPIEL, *,
                game_string: str = DEFAULT_GAME_STRING) -> BackgammonGame:
    """
    Create a backgammon game backend of the requested engine type.

    :param engine: which backend to construct (OpenSpiel for real play, Mock for tests)
    :param game_string: the OpenSpiel game string (ignored by the mock engine)
    :return: a game satisfying the BackgammonGame protocol
    :raises EngineNotAvailableError: if OpenSpiel is requested but `pyspiel` is not importable
    :raises WrongEngineTypeError: if the engine selector is not a known PossibleEngine
    """
    match engine:
        case PossibleEngine.OPEN_SPIEL:
            return OpenSpielGame(game_string)
        case PossibleEngine.MOCK:
            return MockGame()
        case _:
            raise WrongEngineTypeError
