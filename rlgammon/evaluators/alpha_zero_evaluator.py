"""File implementing an alpha-zero style evaluator for MCTS."""
import numpy as np
from numpy.typing import NDArray
from open_spiel.python.algorithms import mcts
import pyspiel

from rlgammon.models.alpha_zero_model import AlphaZeroModel
from rlgammon.models.model_errors.model_errors import ModelNotProvidedToEvaluatorError
from rlgammon.models.model_types import ActorCriticOutput
from rlgammon.rlgammon_types import WHITE, ActionPolicyList


class AlphaZeroEvaluator(mcts.Evaluator):  # type: ignore[misc]
    """TODO."""

    def __init__(self, model: AlphaZeroModel | None = None) -> None:
        """TODO."""
        self.model = model

    def provide_model(self, model: AlphaZeroModel) -> None:
        """
        Add model to the evaluator.

        :param model: alpha-zero style model to add
        """
        self.model = model

    def inference(self, state: pyspiel.BackgammonState) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        TODO.

        :param state: current state in the pyspiel format
        :return:
        """
        if self.model is None:
            raise ModelNotProvidedToEvaluatorError

        obs = state.observation_tensor(WHITE)[:198]
        mask = state.legal_actions_mask()
        return self.model.inference(obs, mask)

    def evaluate(self, state: pyspiel.BackgammonState) -> NDArray[np.float32]:
        """
        TODO.

        :param state:
        :return:
        """
        value, _ = self.inference(state)
        return np.array([value, -value])

    def prior(self, state: pyspiel.BackgammonState) -> ActionPolicyList:
        """
        TODO.

        :param state:
        :return:
        """
        if state.is_chance_node():
            return state.chance_outcomes()  # type: ignore[no-any-return]
        # Returns the probabilities for all actions.
        _, policy = self.inference(state)
        return [(action, policy[action]) for action in state.legal_actions()]
