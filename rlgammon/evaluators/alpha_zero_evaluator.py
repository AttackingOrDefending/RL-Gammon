"""TODO."""
import numpy as np
from open_spiel.python.algorithms import mcts


class AlphaZeroEvaluator(mcts.Evaluator):
    """TODO."""

    def __init__(self, game, model) -> None:
        """TODO."""
        self._model = model

    def inference(self, state):
        """TODO."""
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        value, policy = self._model.inference(obs, mask)
        return value[0, 0], policy[0]  # Unpack batch

    def evaluate(self, state):
        """TODO."""
        value, _ = self.inference(state)
        return np.array([value, -value])

    def prior(self, state):
        """TODO."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all actions.
            _, policy = self.inference(state)
            return [(action, policy[action]) for action in state.legal_actions()]
