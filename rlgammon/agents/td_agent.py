"""File implementing an agent trained with td-learning."""
import pathlib
from uuid import UUID

import torch as th

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.cube.cube_equity import (
    DEFAULT_CUBE_EFFICIENCY,
    CubeAction,
    TakeAction,
    double_decision,
    take_decision,
)
from rlgammon.cube.cube_errors.cube_errors import CubelessModelError
from rlgammon.cube.cube_types import CubeState, MatchContext
from rlgammon.cube.met import MET, WOOLSEY_HEINRICH
from rlgammon.game import GameState, board_features
from rlgammon.models.model_types import ActivationList, LayerList, ValueHead
from rlgammon.models.value_model import TDGammonNet
from rlgammon.rlgammon_types import WHITE, Features


class TDAgent(TrainableAgent):
    """Class implementing an agent trained with td."""

    def __init__(self, pre_made_model_file_name: str | None = None, lr: float = 0.1,
                 gamma: float = 1.0, lamda: float = 0.7, seed: int = 123, color: int = WHITE,
                 layer_list: LayerList | None = None, activation_list: ActivationList | None = None,
                 dtype: str = "float32", value_head: ValueHead = ValueHead.EQUITY_SIGMOID,
                 hidden: int = 128) -> None:
        """
        Construct a td-agent by loading a model or creating a new TD-Gammon value network.

        Note: ``gamma`` is deprecated and ignored (the update is undiscounted), and the externally
        supplied ``layer_list``/``activation_list`` are ignored in favour of the configured value head.

        :param pre_made_model_file_name: file name of a previously trained model, None to build a new one
        :param lr: learning rate
        :param gamma: deprecated future-reward discount, ignored (the update is undiscounted)
        :param lamda: trace decay parameter (how much to value distant states)
        :param seed: seed for the torch and python random number generators
        :param color: 0 or 1 representing which player the agent controls
        :param layer_list: deprecated list of layers, ignored in favour of the configured value head
        :param activation_list: deprecated list of activations, ignored in favour of the configured value head
        :param dtype: the data type of the model
        :param value_head: which value-network output head to build
        :param hidden: number of units in the hidden layer
        """
        super().__init__(color)
        del gamma, layer_list, activation_list
        self.model: TDGammonNet = self.load(pre_made_model_file_name) if pre_made_model_file_name \
            else TDGammonNet(lr=lr, lamda=lamda, hidden=hidden, value_head=value_head, seed=seed, dtype=dtype)

    def episode_setup(self) -> None:
        """Prepare the agent for a training episode by initializing the model's eligibility traces."""
        self.model.init_eligibility_traces()

    def evaluate_position(self, state: Features, decay: bool = False) -> th.Tensor:
        """
        Evaluate the given position using the agent model.

        Note: ``decay`` is a documented no-op kept for interface compatibility (the value is undiscounted).

        :param state: board features to evaluate
        :param decay: deprecated flag, ignored (the value is undiscounted)
        :return: th tensor storing the value of the provided state
        """
        del decay
        value: th.Tensor = self.model(state)
        return value

    def train(self, p: th.Tensor, p_next: th.Tensor) -> float:
        """
        Update the weights of the model according to the td algorithm.

        :param p: value of current state
        :param p_next: value of the next state or final reward if terminal state
        :return: loss associated with the update
        """
        return self.model.update_weights(p, p_next)

    def choose_move(self, actions: list[int], state: GameState) -> int:
        """
        Choose the move leading to the afterstate with the best WHITE-centric value.

        :param actions: set of all possible actions to choose from
        :param state: the current state of the game
        :return: the chosen move to make
        """
        mover = state.current_player()
        best_action = actions[0]
        best_value = -float("inf") if mover == WHITE else float("inf")
        for action in actions:
            nxt = state.clone()
            nxt.apply_action(action)
            value = nxt.returns()[WHITE] if nxt.is_terminal() else float(self.model(board_features(nxt, WHITE)))
            if (mover == WHITE and value > best_value) or (mover != WHITE and value < best_value):
                best_value = value
                best_action = action
        return int(best_action)

    def position_probs(self, state: GameState, perspective: int | None = None) -> list[float]:
        """
        Return the cubeless probability 5-vector for ``state`` from a player's perspective.

        The vector is the raw EQUITY_SIGMOID head output ``(o0, o1, o2, o3, o4)`` of cumulative
        sigmoids; it is the input every cube-equity function consumes.

        :param state: the (non-terminal, non-chance) game state to evaluate
        :param perspective: the player whose probabilities to compute; defaults to the side to move
        :return: the 5-vector ``(o0, o1, o2, o3, o4)`` from ``perspective``'s view
        :raises CubelessModelError: if the model does not use the EQUITY_SIGMOID head
        """
        if self.model.value_head != ValueHead.EQUITY_SIGMOID:
            raise CubelessModelError
        view = perspective if perspective is not None else state.current_player()
        raw = self.model.raw_outputs(board_features(state, view))
        return [float(component) for component in raw]

    def cube_probs(self, state: GameState, perspective: int | None = None) -> list[float]:
        """
        Return a valid cumulative probability 5-vector for cube decisions from a player's view.

        The raw equity head is supervised only through its scalar combination (the cumulative
        win/loss masses are never grounded individually by the TD target), so the raw 5-vector can
        be non-monotone. This method returns the raw vector when it already forms a valid cumulative
        distribution (``1 >= o0 >= o1 >= o2 >= 0`` and ``o0 >= o3 >= o4 >= 0``), and otherwise falls
        back to the gammonless vector ``[p, 0, 0, 0, 0]`` with ``p = (equity + 1) / 2`` clamped to
        ``[0, 1]`` derived from the well-calibrated combined equity. This keeps the cube decisions
        meaningful for value networks trained on a scalar equity target.

        :param state: the (non-terminal, non-chance) game state to evaluate
        :param perspective: the player whose probabilities to compute; defaults to the side to move
        :return: a valid cumulative 5-vector suitable for the cube-equity functions
        :raises CubelessModelError: if the model does not use the EQUITY_SIGMOID head
        """
        raw = self.position_probs(state, perspective)
        o0, o1, o2, o3, o4 = raw
        win_ordered = 1.0 >= o0 >= o1 >= o2 >= 0.0
        lose_ordered = o0 >= o3 >= o4 >= 0.0
        if win_ordered and lose_ordered:
            return raw
        equity = (2.0 * o0 - 1.0) + o1 + o2 - o3 - o4
        win_probability = min(max((equity + 1.0) / 2.0, 0.0), 1.0)
        return [win_probability, 0.0, 0.0, 0.0, 0.0]

    def should_double(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,
                      met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> bool:
        """
        Return whether the on-roll agent should offer a double in ``state``.

        :param state: the decision-node state with the agent on roll
        :param cube: the current cube state from the agent's (on-roll) perspective
        :param match_ctx: the match context from the agent's perspective
        :param met: an optional match-equity table (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index
        :return: whether the agent should double (a double-take or double-pass action)
        """
        action = self.cube_action(state, cube, match_ctx, met=met, x=x)
        return action in (CubeAction.DOUBLE_TAKE, CubeAction.DOUBLE_PASS)

    def cube_action(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,
                    met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> CubeAction:
        """
        Return the agent's full cube decision (no-double / too-good / double-take / double-pass).

        :param state: the decision-node state with the agent on roll
        :param cube: the current cube state from the agent's (on-roll) perspective
        :param match_ctx: the match context from the agent's perspective
        :param met: an optional match-equity table (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index
        :return: the doubler's cube action
        """
        probs = self.cube_probs(state, state.current_player())
        table = met if met is not None else WOOLSEY_HEINRICH
        return double_decision(probs, cube, match_ctx, met=table, x=x)

    def should_take(self, state: GameState, cube: CubeState, match_ctx: MatchContext, *,
                    met: MET | None = None, x: float = DEFAULT_CUBE_EFFICIENCY) -> bool:
        """
        Return whether the agent (the taker) should take an offered double in ``state``.

        The agent is on roll as the taker after the opponent's double; the probabilities are taken
        from the agent's own perspective and the cube is the pre-double cube from that perspective.

        :param state: the decision-node state with the agent (the taker) on roll
        :param cube: the pre-double cube state from the agent's (taker's) perspective
        :param match_ctx: the match context from the agent's perspective
        :param met: an optional match-equity table (defaults to the Woolsey-Heinrich table)
        :param x: the cube-life index
        :return: whether the agent should take (as opposed to passing the double)
        """
        probs = self.cube_probs(state, state.current_player())
        table = met if met is not None else WOOLSEY_HEINRICH
        return take_decision(probs, cube, match_ctx, met=table, x=x) == TakeAction.TAKE

    def save(self, training_session_id: UUID, session_save_count: int,
             main_filename: str | None = "td-backgammon") -> None:
        """
        Save the td model.

        :param training_session_id: uuid of the training session
        :param session_save_count: number of saved sessions
        :param main_filename: name of the file under which the agent is to be saved
        """
        agent_main_filename = f"{main_filename}-{training_session_id}-({session_save_count}).pt"
        agent_file_path = pathlib.Path(__file__).parent
        agent_file_path = agent_file_path.joinpath("saved_agents/")
        agent_file_path.mkdir(parents=True, exist_ok=True)
        th.save(self.model, agent_file_path.joinpath(agent_main_filename))

    def load(self, agent_main_filename: str) -> TDGammonNet:
        """
        Load the td model.

        :param agent_main_filename: name of the file under which the agent is saved
        :return: the loaded agent model
        """
        agent_file_path = pathlib.Path(__file__).parent
        agent_file_path = agent_file_path.joinpath("saved_agents/")
        model: TDGammonNet = th.load(agent_file_path.joinpath(agent_main_filename), weights_only=False)
        return model

    def get_model(self) -> TDGammonNet:
        """
        Get the model this agent is using.

        :return: the agent model if it has one
        """
        return self.model
