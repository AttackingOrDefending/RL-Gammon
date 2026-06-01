"""Pluggable leaf evaluators: a value-network evaluator and a uniform-random rollout evaluator."""

import numpy as np

from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.feature_extractor import board_features
from rlgammon.models.base_model import BaseModel

# Default maximum number of plies before a rollout is truncated and scored as a draw.
DEFAULT_MAX_PLIES = 200


class ValueNetEvaluator:
    """Leaf evaluator that scores a state with a value network from a player's perspective."""

    def __init__(self, model: BaseModel) -> None:
        """
        Construct the evaluator around a value network.

        :param model: the (perspective-agnostic) value network mapping board features to equity
        """
        self._model = model

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s equity for the given state via the value network.

        :param state: the game state to evaluate
        :param perspective: the player whose equity to return (WHITE=0, BLACK=1)
        :return: the network's equity estimate for ``perspective`` in points
        """
        return float(self._model(board_features(state, perspective)))


class RolloutEvaluator:
    """Leaf evaluator that estimates equity by playing uniformly-random legal moves to the end."""

    def __init__(self, rng: np.random.Generator, max_plies: int = DEFAULT_MAX_PLIES) -> None:
        """
        Construct the rollout evaluator.

        :param rng: the random number generator used to sample chance outcomes and moves
        :param max_plies: the maximum number of plies before truncating the rollout as a draw
        """
        self._rng = rng
        self._max_plies = max_plies

    def evaluate(self, state: GameState, perspective: int) -> float:
        """
        Return ``perspective``'s equity by rolling out random play to a terminal state.

        Chance nodes are resolved by sampling a dice outcome according to its probability; decision
        nodes pick a uniformly-random legal action. The rollout runs on a clone, so the input state
        is never mutated.

        :param state: the game state to evaluate
        :param perspective: the player whose return to read off (WHITE=0, BLACK=1)
        :return: ``perspective``'s terminal return, or 0.0 if the rollout was truncated
        """
        rollout = state.clone()
        for _ in range(self._max_plies):
            if rollout.is_terminal():
                return rollout.returns()[perspective]
            if rollout.is_chance_node():
                outcomes = rollout.chance_outcomes()
                actions = [action for action, _ in outcomes]
                probs = [prob for _, prob in outcomes]
                rollout.apply_action(int(self._rng.choice(actions, p=probs)))
            else:
                legal = rollout.legal_actions()
                rollout.apply_action(int(self._rng.choice(legal)))
        return 0.0
