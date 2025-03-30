"""Sequential trainer with training at each step."""

import time
import uuid

from tqdm import tqdm  # type-ignore[import-untyped]

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import Input, MovePart
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError

# Check validity of implementation


class StepTrainer(BaseTrainer):
    """Sequential trainer with training at each step."""

    def __init__(self) -> None:
        """Construct the trainer by initializing its parameters in the BaseTrainer class."""
        super().__init__()

    def train(self, agent: TrainableAgent) -> None:
        """
        Train the provided agent with the parameters provided at the Trainer constructor.

        :param agent: agent to be trained
        """
        if not self.is_ready_for_training():
            raise NoParametersError

        session_id = uuid.uuid4()
        env = BackgammonEnv()
        buffer = self.create_buffer_from_parameters(env)
        explorer = self.create_explorer_from_parameters()
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id)

        total_steps = 0
        training_time_start = time.time()
        for episode in tqdm(range(self.parameters["episodes"]), desc="Training Episodes"):
            env.reset()
            done = False
            trunc = False
            episode_buffer: list[tuple[Input, Input, MovePart, bool, int, int]] = []
            reward = 0.0
            while not done and not trunc:
                state = env.get_input()

                # Get actions from the explorer and agent
                if explorer.should_explore():
                    action = explorer.explore(env.get_all_complete_moves())
                else:
                    action = agent.choose_move(env)

                player = env.current_player
                # Make action and receive observation from state
                reward, done, trunc, _ = env.step(action)
                player_after_move = env.current_player

                next_state = env.get_input()

                if action:
                    episode_buffer.append((state, next_state, action[1], done, player, player_after_move))
                    total_steps += 1

                # Only train agent when at least a batch of data in the buffer
                if buffer.has_element_count(self.parameters["batch_size"]):
                    agent.train(buffer)

            # Update the collected data based on the final result of the game
            self.finalize_data(episode_buffer, env.get_loser(), reward, buffer)

            if (episode + 1) % self.parameters["episodes_per_test"] == 0:
                results = testing.test(agent)
                training_time = time.time() - training_time_start
                logger.update_log(episode, total_steps, results["win_rate"], training_time)
                logger.print_log()

            if self.parameters["save_progress"] and ((episode + 1) % self.parameters["save_every"] == 0):
                logger.save(session_id, episode // self.parameters["save_every"])
                agent.save(session_id, episode // self.parameters["save_every"])
