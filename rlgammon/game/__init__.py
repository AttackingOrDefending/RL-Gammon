"""Game engine boundary: a single, thin abstraction over the backgammon engine.

All OpenSpiel (`pyspiel`) usage is confined to this package, so the rest of the codebase depends
only on the structural protocols defined here and stays unit-testable without OpenSpiel installed.
"""

from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.game.feature_extractor import (
    N_BOARD_FEATURES,
    N_OBS,
    apply_sampled_chance,
    board_features,
    chance_action_probs,
    features_side_to_move,
    sample_chance,
)
from rlgammon.game.game_factory import create_game
from rlgammon.game.game_types import PossibleEngine
from rlgammon.game.openspiel_adapter import DEFAULT_GAME_STRING, OpenSpielGame, is_openspiel_available

__all__ = [
    "DEFAULT_GAME_STRING",
    "N_BOARD_FEATURES",
    "N_OBS",
    "BackgammonGame",
    "GameState",
    "OpenSpielGame",
    "PossibleEngine",
    "apply_sampled_chance",
    "board_features",
    "chance_action_probs",
    "create_game",
    "features_side_to_move",
    "is_openspiel_available",
    "sample_chance",
]
