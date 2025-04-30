"""TODO"""

import numpy as np

from rlgammon.agents.td_agent import TDAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.rlgammon_types import Action, ActionSet


class TDAgentGnu(TDAgent):
    def choose_move(self, actions: ActionSet, env: BackgammonEnv) -> Action:
        """
        Chooses a move to make given the current board and dice roll,
        which goes to the state with maximal value, when playing against a GNU agent.

        :param actions: set of all possible actions to choose from.
        :param env: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        # TODO add optimization for keeping track of best actions
        best_action = None
        if actions:
            game = env.game
            values = [0.0] * len(actions)
            state = game.save_state()

            best_value = float("-inf") if self.color == WHITE else float("inf")
            best_action = None
            for i, action in enumerate(actions):
                game.execute_play(self.color, action)
                opponent = game.get_opponent(self.color)
                observation = (
                    game.get_board_features(opponent)
                )
                values[i] = self.model(observation).detach().numpy()
                game.restore_state(state)

            best_action_index = (
                int(np.argmax(values))
                if self.color == WHITE
                else int(np.argmin(values))
            )
            best_action = list(actions)[best_action_index]

        return best_action

"""
best_action = None
best_value = float("-inf") if self.color == WHITE else float("inf")
for action in actions:
    env_copy = copy(env)
    state, reward, done, info = env_copy.step(action)
    if not done:
        q_values = self.model(th.tensor(state, dtype=th.float32))
        reward = q_values.item()
    if self.color == WHITE:
        if reward > best_value:
            best_value = reward
            best_action = action
    elif reward < best_value:
        best_value = reward
        best_action = action
return best_action
"""
