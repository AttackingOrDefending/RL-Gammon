"""Construct the trainer by initializing its parameters in the BaseTrainer class."""
import uuid
from pyexpat import features

import numpy as np
import pyspiel

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.rlgammon_types import WHITE, EpisodeObservation
from rlgammon.trainer.base_trainer import BaseTrainer


class IterationTrainer(BaseTrainer):
    """Implementation of trainer where training is performed after each iteration."""

    def __init__(self) -> None:
        """Construct the trainer by initializing its parameters in the BaseTrainer class."""
        super().__init__()

    def generate_episode_data(self, agent: TrainableAgent) -> EpisodeObservation:
        """
        TODO.

        :param agent:
        :return:
        """
        episode_data = []

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
        while not state.is_terminal():
            # Remove the last 2 elements, which are the dice. Always from white perspective.
            features = state.observation_tensor(WHITE)[:198]
            legal_actions = state.legal_actions()

            action, action_info = explorer.explore(legal_actions) \
                if explorer.should_explore() else agent.choose_move(legal_actions, state)

            state.apply_action(action)

            if state.is_terminal():
                # Terminal state, use actual reward (negative is black wins).
                reward = state.returns()[WHITE]
                next_features = np.zeros(198)  # dummy state, as no next state after end of episode
            else:
                if not state.is_terminal() and state.is_chance_node():
                    agent.roll_dice(state)

                # Remove the last 2 elements, which are the dice. Always from white perspective.
                next_features = state.observation_tensor(WHITE)[:198]
            episode_data.append((features, next_features, reward, state.is_terminal(), action, action_info))
        return episode_data

    def train(self, agent: TrainableAgent) -> None:
        """
        TODO.

        :param agent:
        :return:
        """
        session_id = uuid.uuid4()
        env = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
        buffer = self.create_buffer_from_parameters(env)
        testing = self.create_testing_from_parameters()
        logger = self.create_logger_from_parameters(session_id)

        for iteration in range(self.parameters["iterations"]):
            episode_data = self.generate_episode_data(agent)
            features = [episode_data[i][0] for i in range(len(episode_data))]
            reward = [episode_data[i][2] for i in range(len(episode_data))]
            action_info = [episode_data[i][5] for i in range(len(episode_data))]

            out = [agent.evaluate_position(feature) for feature in features]
            actor_pred_probs = [out[i][0] for i in range(len(out))]
            critic_pred_values = [out[i][1] for i in range(len(out))]

            _ = agent.train(action_info, actor_pred_probs, reward, critic_pred_values)
