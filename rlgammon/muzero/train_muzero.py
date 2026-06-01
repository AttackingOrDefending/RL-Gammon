"""Training entrypoint wiring Stochastic MuZero self-play, replay and the learner together.

Running this module plays self-play games with :class:`SelfPlayActor`, stores them in a
:class:`MuZeroReplayBuffer` and, once enough decision steps are buffered, repeatedly samples a batch
and applies a :class:`MuZeroLearner` gradient step, printing the component losses periodically.

A ``--smoke`` flag builds a tiny configuration on the in-process :class:`MockGame` and runs only a
handful of games and train steps so the whole loop completes in well under a minute without
``pyspiel``; without it a full-sized configuration is built on the requested real engine.
"""
import argparse

import numpy as np

from rlgammon.game import PossibleEngine, create_game
from rlgammon.game.backgammon_protocol import BackgammonGame
from rlgammon.game.mock_game import MockGame
from rlgammon.muzero.muzero_factory import build_network
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.replay.replay_buffer import MuZeroReplayBuffer
from rlgammon.muzero.self_play.actor import SelfPlayActor
from rlgammon.muzero.training.learner import MuZeroLearner

# Number of self-play games played in a smoke run.
SMOKE_GAMES = 4
# Number of learner train steps taken per smoke iteration once the buffer is warm.
SMOKE_TRAIN_STEPS_PER_GAME = 2
# How often (in train steps) to print the running losses.
PRINT_EVERY = 1
# Number of self-play games and train steps for a (non-smoke) demonstration run.
FULL_GAMES = 50
FULL_TRAIN_STEPS_PER_GAME = 8
FULL_PRINT_EVERY = 4


def smoke_config() -> MuZeroConfig:
    """
    Build a tiny configuration whose self-play and training loop completes in well under a minute.

    :return: a small :class:`MuZeroConfig` sized for the mock game and a fast smoke loop
    """
    return MuZeroConfig(
        observation_size=198,
        num_actions=8,
        state_channels=16,
        hidden_sizes=(16,),
        codebook_size=2,
        num_simulations=4,
        unroll_steps=3,
        td_steps=5,
        batch_size=8,
        value_support_size=3,
        reward_support_size=3,
        replay_capacity=1000,
    )


def full_config() -> MuZeroConfig:
    """
    Build a full-sized configuration for a real training run.

    :return: the default :class:`MuZeroConfig`
    """
    return MuZeroConfig()


def _build_game(engine: PossibleEngine, *, smoke: bool) -> BackgammonGame:
    """
    Build the game backend for the run, forcing the mock engine in smoke mode.

    :param engine: the requested engine for a real run
    :param smoke: whether the run is a quick smoke run (always uses the mock game)
    :return: the game satisfying the :class:`BackgammonGame` protocol
    """
    if smoke:
        return MockGame()
    return create_game(engine)


def _print_losses(prefix: str, losses: dict[str, float]) -> None:
    """
    Print the component losses on a single line.

    :param prefix: a short prefix identifying the train step
    :param losses: the loss dict returned by :meth:`MuZeroLearner.train_step`
    """
    formatted = " ".join(f"{key}={losses[key]:.4f}" for key in ("total", "value", "policy", "reward", "chance", "commitment"))
    print(f"{prefix} {formatted}")


def run_training(config: MuZeroConfig, game: BackgammonGame, *, games: int,
                 train_steps_per_game: int, print_every: int) -> None:
    """
    Run the self-play / replay / learner loop for a fixed number of games.

    For each game: a self-play trajectory is played and stored, then once the buffer holds at least one
    batch worth of steps a number of train steps are taken on freshly sampled batches, printing the
    losses every ``print_every`` train steps.

    :param config: the configuration shared across self-play, replay and the learner
    :param game: the game backend producing fresh initial states
    :param games: the number of self-play games to play
    :param train_steps_per_game: the number of train steps to take after each game once the buffer is warm
    :param print_every: print the losses every this many train steps
    """
    rng = np.random.default_rng(config.seed)
    network = build_network(config)
    buffer = MuZeroReplayBuffer(config)
    actor = SelfPlayActor(config, game, network, rng)
    learner = MuZeroLearner(config, network)

    train_step_index = 0
    for game_index in range(games):
        buffer.save(actor.play_game())
        print(f"[self-play] game {game_index + 1}/{games} buffer_steps={len(buffer)}")

        if len(buffer) < config.batch_size:
            continue

        for _ in range(train_steps_per_game):
            batch = buffer.sample_batch(rng)
            losses = learner.train_step(batch)
            train_step_index += 1
            if train_step_index % print_every == 0:
                _print_losses(f"[train] step {train_step_index}", losses)


def main() -> None:
    """Parse the command-line arguments and run the Stochastic MuZero training loop."""
    parser = argparse.ArgumentParser(description="Train the Stochastic MuZero agent.")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a tiny, fast self-play and training loop on the mock game and exit",
    )
    parser.add_argument(
        "--engine", type=str, default=PossibleEngine.OPEN_SPIEL.value,
        help="game engine selector for a real run (e.g. 'OS' for OpenSpiel, 'MOCK' for the mock game)",
    )
    args = parser.parse_args()

    if args.smoke:
        config = smoke_config()
        game = _build_game(PossibleEngine.MOCK, smoke=True)
        run_training(
            config, game, games=SMOKE_GAMES,
            train_steps_per_game=SMOKE_TRAIN_STEPS_PER_GAME, print_every=PRINT_EVERY,
        )
        return

    engine = PossibleEngine.get_enum_from_string(args.engine)
    config = full_config()
    game = _build_game(engine, smoke=False)
    run_training(
        config, game, games=FULL_GAMES,
        train_steps_per_game=FULL_TRAIN_STEPS_PER_GAME, print_every=FULL_PRINT_EVERY,
    )


if __name__ == "__main__":
    main()
