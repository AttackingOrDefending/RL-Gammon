import json

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.buffers.uniform_buffer import UniformBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.exploration.base_exploration import BaseExploration
from rlgammon.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration
from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.rlgammon_types import Input, MoveList
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError, WrongExplorationTypeError, \
    WrongBufferTypeError

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid


# TODO Add Agent to training loop


class StepTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()

    def load_parameters(self, json_parameters: str) -> None:
        parameters = json.loads(json_parameters)
        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            raise ValueError("Invalid parameters")

    def create_buffer(self) -> BaseBuffer:
        """
        TODO

        :return:
        """

        if self.parameters["buffer"] == PossibleBuffers.UNIFORM:
            # TODO FIX !!!
            buffer = UniformBuffer((100, 100), 10)
        else:
            raise WrongBufferTypeError()

        return buffer

    def create_explorer(self) -> BaseExploration:
        """
        TODO

        :return:
        """

        if self.parameters["exploration"] == PossibleExploration.EPSILON_GREEDY:
            explorer = EpsilonGreedyExploration(self.parameters["start_epsilon"], self.parameters["end_epsilon"],
                                                self.parameters["update_decay"], self.parameters["steps_per_update"])
        else:
            raise WrongExplorationTypeError()

        return explorer

    def finalize_data(self, episode_buffer: list[tuple[Input, Input, MoveList, float, bool, int]],
                      losing_player: int, buffer: BaseBuffer) -> None:
        """
        TODO

        :param episode_buffer:
        :param losing_player:
        :param buffer:
        """

        for i, (state, next_state, action, reward, done, player) in enumerate(reversed(episode_buffer)):
            if player == losing_player:
                reward *= -1
            reward *= self.parameters["decay"] ** i
            buffer.record(state, next_state, action, reward, done)

    def train(self, agent: TrainableAgent) -> None:
        if not self.is_ready_for_training():
            raise NoParametersError

        buffer = self.create_buffer()
        explorer = self.create_explorer()
        env = BackgammonEnv()
        for episode in range(self.parameters["episodes"]):
            env.reset()
            done = False
            trunc = False
            episode_buffer = []
            while not done and not trunc:
                dice = env.roll_dice()
                if explorer.should_explore():
                    actions = explorer.explore(env.get_all_complete_moves(dice)) # TODO CHECK MOVE STRUCTURE
                else:
                    actions = agent.choose_move(env, dice)

                # TODO CHECK
                state = env.get_input()
                for _, action in actions:
                    reward, done, trunc, _ = env.step(action)

                next_state = env.get_input()
                episode_buffer.append((state, next_state, actions, reward, done, 1)) # TODO ADD CURRENT PLAYER

                # Only train agent when at least a batch of data in the buffer
                if buffer.has_element_count(self.parameters["batch_size"]):
                    agent.train(buffer)
