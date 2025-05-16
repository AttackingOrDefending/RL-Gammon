"""TODO."""
import numpy as np
from numpy.typing import NDArray
from open_spiel.python.algorithms import mcts
import pyspiel

from rlgammon.models.alpha_zero_model import AlphaZeroModel
from rlgammon.rlgammon_types import ActionPolicyList, Feature


class AlphaZeroEvaluator(mcts.Evaluator):  # type: ignore[misc]
    """TODO."""

    def __init__(self, game: pyspiel.Game, model: AlphaZeroModel) -> None:
        """TODO."""
        self.model = model

    def inference(self, state: pyspiel.BackgammonState):
        """TODO."""
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        value, policy = self.model.inference(obs, mask)
        return value[0, 0], policy[0]  # Unpack batch

    def evaluate(self, state: pyspiel.BackgammonState) -> NDArray[np.float32]:
        """TODO."""
        value, _ = self.inference(state)
        return np.array([value, -value])

    def prior(self, state: pyspiel.BackgammonState) -> ActionPolicyList:
        """TODO."""
        if state.is_chance_node():
            return state.chance_outcomes()  # type: ignore[no-any-return]
        # Returns the probabilities for all actions.
        _, policy = self.inference(state)
        return [(action, policy[action]) for action in state.legal_actions()]
