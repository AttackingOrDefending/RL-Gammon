import json

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers.base_buffer import BaseBuffer
from rlgammon.buffers.buffer_types import PossibleBuffers
from rlgammon.buffers.uniform_buffer import UniformBuffer
from rlgammon.environment import BackgammonEnv
from rlgammon.exploration.base_exploration import BaseExploration
from rlgammon.exploration.epsilon_greedy_exploration import EpsilonGreedyExploration
from rlgammon.exploration.exploration_types import PossibleExploration
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError, WrongExplorationTypeError, \
    WrongBufferTypeError

from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid


class StepTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()

    def load_parameters(self, json_parameters: str) -> None:
        parameters = json.loads(json_parameters)
        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            raise ValueError("Invalid parameters")

    def train(self, agent: TrainableAgent) -> None:
        """
        Train the provided agent with the parameters provided at the Trainer constructor.

        :param agent: agent to be trained
        """

        if not self.is_ready_for_training():
            raise NoParametersError

        buffer = self.create_buffer_from_parameters()
        explorer = self.create_explorer_from_parameters()
        env = BackgammonEnv()
        for episode in range(self.parameters["episodes"]):
            env.reset()
            done = False
            trunc = False
            episode_buffer = []
            while not done and not trunc:
                dice = env.roll_dice()
                if explorer.should_explore():
                    actions = explorer.explore(env.get_all_complete_moves(dice))
                else:
                    actions = agent.choose_move(env, dice)

                # TODO CHECK
                state = env.get_input()
                for _, action in actions:
                    reward, done, trunc, _ = env.step(action)

                next_state = env.get_input()
                episode_buffer.append((state, next_state, actions, reward, done, env.current_player))

                # Only train agent when at least a batch of data in the buffer
                if buffer.has_element_count(self.parameters["batch_size"]):
                    agent.train(buffer)
