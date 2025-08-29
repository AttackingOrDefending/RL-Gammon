"""Construct the trainer by initializing its parameters in the BaseTrainer class."""
import time
import uuid

import numpy as np
import pyspiel
from tqdm import tqdm

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.buffers import BaseBuffer
from rlgammon.rlgammon_types import WHITE
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_types import TrainerType


class IterationTrainer(BaseTrainer):
    """Implementation of trainer where training is performed after each iteration."""

    def __init__(self) -> None:
        """Construct the trainer by initializing its parameters in the BaseTrainer class."""
        super().__init__()

    def generate_episode_data(self, buffer: BaseBuffer, agent: TrainableAgent) -> None:
        """
        TODO.

        :param agent:
        """
        env = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
        explorer = self.create_explorer_from_parameters()

        # Roll dice at the start of the game
        agent.episode_setup()
        state = env.new_initial_state()
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)  # noqa: B905
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)

        reward = 0
        total_steps = 0
        while not state.is_terminal():
            # Remove the last 2 elements, which are the dice. Always from white perspective.
            features = np.array(state.observation_tensor(WHITE)[:198])
            legal_actions = state.legal_actions()
            player = state.current_player()

            action, action_info = explorer.explore(legal_actions) \
                if explorer.should_explore() else agent.choose_move(legal_actions, state)
            state.apply_action(action)
            player_after = state.current_player()

            if state.is_terminal():
                # Terminal state, use actual reward (negative is black wins).
                reward = state.returns()[WHITE]
                next_features = np.zeros(198)  # dummy state, as no next state after end of episode
            else:
                if not state.is_terminal() and state.is_chance_node():
                    agent.roll_dice(state)

                # Remove the last 2 elements, which are the dice. Always from white perspective.
                next_features = np.array(state.observation_tensor(WHITE)[:198])

            buffer.record(features, next_features, action, reward, state.is_terminal(), player, player_after, action_info)
            total_steps += 1
        return total_steps

    def train(self, agent: TrainableAgent) -> None:
        """
        TODO.

        :param agent:
        :return:
        """
        session_id = uuid.uuid4()

        buffer = self.create_buffer_from_parameters(pyspiel.load_game("backgammon(scoring_type=full_scoring)"))
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id, TrainerType.ITERATION_TRAINER)

        total_steps = 0
        training_time_start = time.time()
        for iteration in tqdm(range(self.parameters["iterations"]), desc="Training Iterations"):
            for _ in range(self.parameters["episodes_per_iteration"]):
                total_steps += self.generate_episode_data(buffer, agent)

            # Train on the entire dataset -> TODO
            batch = buffer.get_all_elements()
            for i in range(buffer.get_num_elements()):
                state = batch["state"][i]
                reward = batch["reward"][i]
                action_info = batch["action_info"][i]

                v, p = agent.evaluate_position(state)
                loss = agent.train(action_info, p, reward, v)

            results = testing.test(agent)
            training_time = time.time() - training_time_start
            logger.update_log(iteration, total_steps, results, training_time)
            logger.print_log()
