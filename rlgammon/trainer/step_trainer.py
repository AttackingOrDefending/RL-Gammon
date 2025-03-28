"""Sequential trainer with training at each step."""

import json
from pathlib import Path
import time
import uuid

from tqdm import tqdm

from rlgammon.agents.trainable_agent import TrainableAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.rlgammon_types import Input, MoveList, MovePart
from rlgammon.trainer.base_trainer import BaseTrainer
from rlgammon.trainer.trainer_errors.trainer_errors import NoParametersError
from rlgammon.trainer.trainer_parameters.parameter_verification import are_parameters_valid

# Check validity of implementation


class StepTrainer(BaseTrainer):
    """Sequential trainer with training at each step."""

    def __init__(self) -> None:
        """Construct the trainer by initializing its parameters in the BaseTrainer class."""
        super().__init__()

    def load_parameters(self, json_parameters_name: str) -> None:
        """
        Load parameters to be used for training, and verify their validity.


        :param json_parameters_name: name of the json parameters file
        :raises: ValueError: the parameters are invalid, i.e. don't contain some data, or have invalid types
        """
        parameter_file_path = Path(__file__).parent
        parameter_file_path = parameter_file_path.joinpath("trainer_parameters/parameters/")
        path = parameter_file_path.joinpath(json_parameters_name)

        with path.open() as json_parameters:
            parameters = json.load(json_parameters)

        if are_parameters_valid(parameters):
            self.parameters = parameters
        else:
            msg = "Invalid parameters"
            raise ValueError(msg)

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
                # If no actions can be made, then end iteration early and step env with empty move
                # Which will trigger a switch to the next player
                if not env.is_movement_possible():
                    env.step(())
                    continue

                state = env.get_input()

                # Get actions from the explorer and agent
                if explorer.should_explore():
                    action = explorer.explore(env.get_all_complete_moves())
                else:
                    action = agent.choose_move(env)

                player = env.current_player
                # Make action and receive observation from state
                reward, done, trunc, _ = env.step(action)
                player_after = env.current_player

                next_state = env.get_input()
                episode_buffer.append((state, next_state, action[1], done, player, player_after))
                total_steps += 1

                # Only train agent when at least a batch of data in the buffer
                if buffer.has_element_count(self.parameters["batch_size"]):
                    agent.train(buffer)
            """
            print("PLAYER", player)
            print("LOSER", env.get_loser())
            print("HAS LOST", env.has_lost(player))
            print("MOVES", env.moves)
            print(env)
            """

            # Update the collected data based on the final result of the game
            self.finalize_data(episode_buffer, env.get_loser(), reward, buffer)

            if  (episode + 1) % self.parameters["episodes_per_test"] == 0:
                results = testing.test(agent)
                training_time = time.time() - training_time_start
                logger.update_log(episode, total_steps, results["win_rate"], training_time)
                logger.print_log()

            if self.parameters["save_progress"] and ((episode + 1) % self.parameters["save_every"] == 0):
                logger.save(session_id, episode // self.parameters["save_every"])
                agent.save(session_id, episode // self.parameters["save_every"])

