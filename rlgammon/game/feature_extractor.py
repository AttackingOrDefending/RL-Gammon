"""Feature extraction and chance handling, shared by every consumer of the game engine.

This module centralizes two fixes from the original code:
  * features are always taken from a chosen player's own perspective (the original agent
    always evaluated from WHITE while the side to move alternates);
  * chance events are resolved by sampling a real dice outcome by its probability (the original
    code hardcoded ``apply_action(7)``).
"""

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.game_errors.game_errors import NonChanceNodeError
from rlgammon.rlgammon_types import Features

# Number of board features (the value-network input); the trailing 2 dice entries are dropped.
N_BOARD_FEATURES = 198
# Full observation length: 198 board features + 2 dice.
N_OBS = 200


def board_features(state: GameState, perspective: int) -> Features:
    """
    Return the board features from the given player's perspective.

    :param state: the game state to encode
    :param perspective: the player whose perspective to encode (WHITE=0, BLACK=1)
    :return: the first 198 entries of the observation tensor (board only, dice dropped)
    """
    return state.observation_tensor(perspective)[:N_BOARD_FEATURES]


def features_side_to_move(state: GameState) -> Features:
    """
    Return the board features from the perspective of the player to move.

    :param state: the (decision-node) game state to encode
    :return: the board features from ``state.current_player()``'s perspective
    """
    return board_features(state, state.current_player())


def chance_action_probs(state: GameState) -> tuple[list[int], list[float]]:
    """
    Split the pending chance event into parallel lists of action ids and probabilities.

    :param state: the chance-node game state
    :return: a tuple of (action ids, probabilities)
    :raises NonChanceNodeError: if the state is not a chance node
    """
    if not state.is_chance_node():
        raise NonChanceNodeError
    outcomes = state.chance_outcomes()
    actions = [action for action, _ in outcomes]
    probs = [prob for _, prob in outcomes]
    return actions, probs


def sample_chance(state: GameState, rng: np.random.Generator) -> int:
    """
    Sample a single dice outcome at a chance node according to its probability.

    :param state: the chance-node game state
    :param rng: the random number generator to sample with
    :return: the sampled chance action id
    """
    actions, probs = chance_action_probs(state)
    return int(rng.choice(actions, p=probs))


def apply_sampled_chance(state: GameState, rng: np.random.Generator) -> int:
    """
    Sample a dice outcome at a chance node and apply it to the state in place.

    :param state: the chance-node game state (mutated in place)
    :param rng: the random number generator to sample with
    :return: the applied chance action id
    """
    action = sample_chance(state, rng)
    state.apply_action(action)
    return action
