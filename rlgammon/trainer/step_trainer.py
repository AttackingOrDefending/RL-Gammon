"""Trainer where training is performed at each step."""

import time
import uuid

import numpy as np
import pyspiel  # type: ignore[import-not-found]
from tqdm import tqdm

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.rlgammon_types import WHITE
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError

"""TODO: Move load and save from td-agent to trainable agent"""

class StepTrainer(BaseTrainer):
    """Implementation of trainer where training is performed at each step."""

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
        env = pyspiel.load_game("backgammon(scoring_type=full_scoring)")

        explorer = self.create_explorer_from_parameters()
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id)

        total_steps = 0
        training_time_start = time.time()
        for episode in tqdm(range(self.parameters["episodes"]), desc="Training Episodes"):

            agent.episode_setup()

            state = env.new_initial_state()
            agent.roll_dice(state)

            while not state.is_terminal():
                # Remove the last 2 elements, which are the dice. Always from white perspective.
                features = state.observation_tensor(WHITE)[:198]

                p = agent.evaluate_position(features)
                legal_actions = state.legal_actions()

                action = explorer.explore(legal_actions) \
                    if explorer.should_explore() else agent.choose_move(legal_actions, state)

                state.apply_action(action)

                if state.is_terminal():
                    # Terminal state, use actual reward (negative is black wins).
                    reward = state.returns()[WHITE]
                    _ = agent.train(p, reward)
                else:
                    if not state.is_terminal() and state.is_chance_node():
                        agent.roll_dice(state)

                    # Remove the last 2 elements, which are the dice. Always from white perspective.
                    next_features = state.observation_tensor(WHITE)[:198]
                    p_next = agent.evaluate_position(next_features, decay=True)
                    _ = agent.train(p, p_next)

            if (episode + 1) % self.parameters["episodes_per_test"] == 0:
                results = testing.test(agent)
                training_time = time.time() - training_time_start
                logger.update_log(episode, total_steps, results["win_rate"], training_time)
                logger.print_log()

            if self.parameters["save_progress"] and ((episode + 1) % self.parameters["save_every"] == 0):
                logger.save(session_id, episode // self.parameters["save_every"])
                agent.save(session_id, episode // self.parameters["save_every"])
