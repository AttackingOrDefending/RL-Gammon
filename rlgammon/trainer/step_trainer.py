"""Sequential trainer with training at each step."""

import time
import uuid

import numpy as np
import torch as th
from tqdm import tqdm

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.game import (
    PossibleEngine,
    apply_sampled_chance,
    board_features,
    create_game,
)
from rlgammon.rlgammon_types import WHITE
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
        :raises NoParametersError: if the trainer has not been given parameters yet
        """
        if not self.is_ready_for_training():
            raise NoParametersError

        session_id = uuid.uuid4()
        game = create_game(PossibleEngine.OPEN_SPIEL)
        rng = np.random.default_rng()

        explorer = self.create_explorer_from_parameters()
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id)

        total_steps = 0
        training_time_start = time.time()
        for episode in tqdm(range(1, self.parameters["episodes"] + 1), desc="Training Episodes"):

            agent.episode_setup()

            state = game.new_initial_state()
            apply_sampled_chance(state, rng)

            while not state.is_terminal():
                # Always evaluate from the WHITE perspective to keep the bootstrap consistent.
                features = board_features(state, WHITE)

                p = agent.evaluate_position(features)
                legal_actions = state.legal_actions()

                action = (explorer.explore(legal_actions)
                    if explorer.should_explore() else agent.choose_move(legal_actions, state))
                assert isinstance(action, int)
                state.apply_action(action)

                if state.is_terminal():
                    # Terminal state, use the actual WHITE-centric reward (negative means black wins).
                    reward = th.tensor(state.returns()[WHITE], dtype=p.dtype)
                    _ = agent.train(p, reward)
                else:
                    if state.is_chance_node():
                        # Resolve the pending dice roll so the side to move is included in the input.
                        apply_sampled_chance(state, rng)

                    next_features = board_features(state, WHITE)
                    p_next = agent.evaluate_position(next_features)
                    _ = agent.train(p, p_next)
                total_steps += 1

            if episode % self.parameters["episodes_per_test"] == 0:
                results = testing.test(agent)
                training_time = time.time() - training_time_start
                logger.update_log(episode, total_steps, results, training_time)
                logger.print_log()

            if self.parameters["save_progress"] and ((episode + 1) % self.parameters["save_every"] == 0):
                logger.save(session_id, episode // self.parameters["save_every"])
                agent.save(session_id, episode // self.parameters["save_every"])
