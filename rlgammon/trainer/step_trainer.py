"""Sequential trainer with training at each step."""

import time
import uuid

from tqdm import tqdm

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment.backgammon_env import BackgammonEnv
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError


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
        explorer = self.create_explorer_from_parameters()
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id)

        total_steps = 0
        training_time_start = time.time()
        for episode in tqdm(range(self.parameters["episodes"]), desc="Training Episodes"):
            agent_color, first_roll, state = env.reset()
            done = False
            while not done:
                # If this is the first step, take the roll from env, else roll yourself
                if first_roll:
                    roll = first_roll
                    first_roll = None
                else:
                    roll = agent.roll_dice()

                # Get actions from the explorer and agent
                actions = env.get_valid_actions(roll)
                action = explorer.explore(actions) if explorer.should_explore() else agent.choose_move(actions, env)

                # Make action and receive observation from state
                next_state, reward, done, winner = env.step(action)

                # Update agent, exploration, and total step count
                _ = agent.train(state, next_state, reward, done)
                explorer.update()
                total_steps += 1

            if (episode + 1) % self.parameters["episodes_per_test"] == 0:
                results = testing.test(agent)
                training_time = time.time() - training_time_start
                logger.update_log(episode, total_steps, results["win_rate"], training_time)
                logger.print_log()

            if self.parameters["save_progress"] and ((episode + 1) % self.parameters["save_every"] == 0):
                logger.save(session_id, episode // self.parameters["save_every"])
                agent.save(session_id, episode // self.parameters["save_every"])
