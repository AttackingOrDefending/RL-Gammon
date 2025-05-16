#type: ignore  # noqa: PGH003

"""File implementing agent trained with td-learning and capable of playing against a gnubg agent."""
import numpy as np

from rlgammon.agents.gnu_agent import GNUAgent
from rlgammon.agents.td_agent import TDAgent
from rlgammon.environment.backgammon_env import BackgammonEnv  # type: ignore[attr-defined]
from rlgammon.environment.gnubg.gnubg_backgammon import GnubgInterface, gnubgState  # type: ignore[attr-defined]
from rlgammon.models.model_types import ActivationList, LayerList
from rlgammon.rlgammon_types import BLACK, WHITE, ActionGNU, ActionSetGNU


class TDAgentGnu(TDAgent, GNUAgent):
    """Class implementing agent trained with td-learning and capable of playing against a gnubg agent."""

    def __init__(self, gnubg_interface: GnubgInterface, pre_made_model_file_name: str | None = None, lr: float = 0.01,
                 gamma: float = 0.99, lamda: float = 0.99, seed: int = 123, color: int = WHITE,
                 layer_list: LayerList =  None, activation_list: ActivationList = None, dtype: str = "float32") -> None:
        """
        Construct the td-gnu agent by creating a td agent with the given parameters, and
        storing the provided the gnubg interface to use when playing against gnubg.

        :param gnubg_interface: the interface used to communicate with gnubg
        :param pre_made_model_file_name: file name of a previously trained model, None if a new model is to be used
        :param lr: learning rate
        :param gamma: future reward discount
        :param lamda: trace decay parameters (how much to value distant states)
        :param seed: seed for random number generator of torch and the python random package
        :param color: 0 or 1 representing which player the agent controls
        :param layer_list: list of layers to use
        :param activation_list: list of activation functions to use
        :param dtype: the data type of the model
        """
        super().__init__(pre_made_model_file_name, lr, gamma, lamda, seed, color, layer_list, activation_list, dtype)
        self.gnubg_interface = gnubg_interface

    def roll_dice_gnu(self) -> tuple[int, int] | gnubgState:
        """
        Get dice rolls for gnubg environment.

        :return: dice rolls for gnubg environment
        """
        gnubg = self.gnubg_interface.send_command("roll")
        return self.handle_opponent_move(gnubg)

    def choose_move(self, actions: ActionSetGNU, state: BackgammonEnv) -> int | ActionGNU:
        """
        Chooses a move to make given the current board and dice roll,
        which goes to the state with maximal value, when playing against a GNU agent.

        :param actions: set of all possible actions to choose from.
        :param state: the current environment (and it's associated state)
        :return: the chosen move to make.
        """
        best_action = None
        color = state.current_player()
        opponent_color = WHITE if color == BLACK else BLACK
        if actions:
            game = state.game
            values = [-10.] * len(actions) if color == BLACK else [10.] * len(actions)
            state = game.save_state()

            for i, action in enumerate(actions):
                game.execute_play(color, action)
                observation = game.get_board_features(opponent_color, BLACK)
                values[i] = self.model(observation).detach().numpy()
                game.restore_state(state)

            best_action_index = (
                int(np.argmax(values))
                if color == BLACK
                else int(np.argmin(values))
            )
            best_action = list(actions)[best_action_index]

        return best_action

    def handle_opponent_move(self, gnubg: gnubgState) -> gnubgState:
        """
        React to opponent move in the GNU env.

        :param gnubg: the interface used to communicate with gnubg
        :return:
        """
        # Once I roll the dice, 2 possible situations can happen:
        # 1) I can move (the value gnubg.roll is not None)
        # 2) I cannot move, so my opponent rolls the dice and makes its move, and eventually ask for doubling,
        #    so I have to roll the dice again

        # One way to distinguish between the above cases,
        # is to check the color of the player that performs the last move in gnubg:

        # - if the player's color is the same as the TD Agent, it means I can send the 'move' command
        #   (no other moves have been performed after the 'roll' command) - case 1);
        # - if the player's color is not the same as the TD Agent,
        #   this means that the last move performed after the 'roll' is not of the TD agent - case 2)

        previous_agent = gnubg.agent
        if previous_agent == self.color:  # case 1)
            return gnubg
        # case 2)
        while previous_agent != self.color and gnubg.winner is None:
            # check if my opponent asks for doubling
            # always take when opponent asks for doubling
            gnubg = self.gnubg_interface.send_command("take") \
                if gnubg.double else self.gnubg_interface.send_command("roll")
            previous_agent = gnubg.agent
        return gnubg
