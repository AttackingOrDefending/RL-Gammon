"""Resumable, GPU-saturating long-training loop for Stochastic MuZero vs a random opponent.

This script is the multi-hour counterpart of :mod:`scripts.train_muzero`. It wires the three speed /
correctness wins of Work Unit P into one resumable loop:

* **Batched self-play** -- a :class:`~rlgammon.muzero.self_play.batched_actor.BatchedSelfPlayActor`
  advances ``--parallel`` real games in lockstep so every network inference batches many positions in
  a single GPU call (instead of the launch-bound batch-1 search of the single-game actor).
* **Gumbel MuZero root selection** -- the batched search uses Gumbel-top-k + sequential halving and the
  Gumbel-improved policy as the stored policy target, giving strong policy improvement with few
  simulations and removing the need for Dirichlet noise.
* **Correct two-player value targets** -- the perspective/sign fix in
  :meth:`~rlgammon.muzero.replay.trajectory.Trajectory._compute_value_target` so value targets point
  the right way on alternating plies.

It checkpoints every ``--checkpoint-minutes`` minutes (a rolling ``latest`` plus a ``best`` kept by the
highest random-opponent win-rate), evaluates against random every ``--eval-every`` games, stops at the
``--max-seconds`` wall-clock cap, can ``--resume`` from a checkpoint (restoring the network, optimizer,
step and game counters and the best win-rate), and prints a clear, timestamped win-rate curve so a
~6-hour run can be followed live and restarted safely.
"""
import argparse
from collections.abc import Callable
from dataclasses import dataclass
import datetime
import pathlib
import time

import numpy as np
import torch as th

from rlgammon.game import PossibleEngine, create_game
from rlgammon.game.backgammon_protocol import BackgammonGame
from rlgammon.muzero.muzero_factory import build_network, resolve_device
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.replay.replay_buffer import MuZeroReplayBuffer
from rlgammon.muzero.replay.trajectory import Trajectory
from rlgammon.muzero.self_play.actor import SelfPlayActor
from rlgammon.muzero.self_play.batched_actor import BatchedSelfPlayActor
from rlgammon.muzero.training.learner import MuZeroLearner
from scripts.eval_vs_random import play_muzero_vs_random

# MuZero observation size (board features only, dice dropped) and action-space size for OpenSpiel.
MUZERO_OBSERVATION_SIZE = 198
MUZERO_NUM_ACTIONS = 1352
# Default capable-but-memory-modest network for a 6 GB GPU (state width, hidden widths, codebook).
DEFAULT_STATE_CHANNELS = 256
DEFAULT_HIDDEN = 256
DEFAULT_CODEBOOK_SIZE = 32
# Default categorical value/reward support size.
DEFAULT_SUPPORT_SIZE = 21
# Default unroll / TD / replay / optimization hyper-parameters.
DEFAULT_UNROLL_STEPS = 5
DEFAULT_TD_STEPS = 10
DEFAULT_DISCOUNT = 1.0
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4
# Value-loss weight equal to the policy weight (AlphaZero-style): the Gumbel search leans on the value
# head to improve the policy, so the value must learn at full strength (the MuZero default 0.25 is too
# weak for a board game). Confirmed to give a clearer win-rate climb than 0.25 in the short validation.
DEFAULT_VALUE_LOSS_WEIGHT = 1.0
DEFAULT_REPLAY_CAPACITY = 200_000
# Default self-play search budget and Gumbel root width.
DEFAULT_TRAIN_SIMS = 32
DEFAULT_CONSIDERED = 16
# Default number of games advanced simultaneously per batched self-play call (the GPU throughput key).
DEFAULT_PARALLEL = 32
# Default self-play path: the batched Gumbel FEATURE actor (preserves this script's current behaviour
# and the in-flight run's resume). Pass ``--self-play single`` for the proven BASELINE single-game actor.
DEFAULT_SELF_PLAY = "batched"
# Default evaluation cadence / budget (a decent sim count and enough games for low variance).
DEFAULT_EVAL_SIMS = 50
DEFAULT_EVAL_EVERY = 256
DEFAULT_EVAL_GAMES = 50
# Default number of learner train steps taken per completed self-play game (once the buffer is warm).
DEFAULT_TRAIN_STEPS_PER_GAME = 4
# Default wall-clock cap (~6 hours) and checkpoint cadence.
DEFAULT_MAX_SECONDS = 21000.0
DEFAULT_CHECKPOINT_MINUTES = 10.0
# Default random seed.
DEFAULT_SEED = 0
# A learning-rate decay floor and horizon so the rate gently anneals over the run.
LR_DECAY_FLOOR = 0.2
# The loss keys printed on each training line.
LOSS_KEYS = ("total", "value", "policy", "reward", "chance", "commitment")
# Sub-directory (under this script's parent) where checkpoints are written.
CHECKPOINT_DIR_NAME = "muzero_checkpoints"


@dataclass
class TrainState:
    """The resumable training counters persisted alongside the network and optimizer."""

    game_index: int = 0
    train_step_index: int = 0
    best_win_rate: float = -1.0
    elapsed_seconds: float = 0.0


def _default_device() -> str:
    """
    Pick the default training device: ``"cuda"`` when a CUDA GPU is present, else ``"cpu"``.

    :return: ``"cuda"`` if :func:`torch.cuda.is_available` is ``True``, otherwise ``"cpu"``
    """
    return "cuda" if th.cuda.is_available() else "cpu"


def _timestamp() -> str:
    """
    Return a compact local timestamp for the live training log.

    :return: the current local time formatted as ``HH:MM:SS``
    """
    return datetime.datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005


def build_config(args: argparse.Namespace) -> MuZeroConfig:
    """
    Assemble the :class:`MuZeroConfig` for the long run from the parsed command-line arguments.

    :param args: the parsed command-line arguments
    :return: the assembled :class:`MuZeroConfig`
    """
    return MuZeroConfig(
        observation_size=MUZERO_OBSERVATION_SIZE,
        num_actions=MUZERO_NUM_ACTIONS,
        state_channels=args.state_channels,
        hidden_sizes=(args.hidden, args.hidden),
        codebook_size=args.codebook_size,
        num_simulations=args.sims,
        eval_num_simulations=args.eval_sims,
        unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        value_support_size=args.support_size,
        reward_support_size=args.support_size,
        replay_capacity=args.replay_capacity,
        seed=args.seed,
        device=args.device,
    )


def _checkpoint_dir() -> pathlib.Path:
    """
    Return (creating if needed) the directory checkpoints are written to.

    :return: the checkpoint directory path
    """
    path = pathlib.Path(__file__).parent.joinpath(CHECKPOINT_DIR_NAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(name: str, network: StochasticMuZeroNetwork, learner: MuZeroLearner,
                    state: TrainState, config: MuZeroConfig) -> pathlib.Path:
    """
    Write a full resumable checkpoint (network, optimizer, counters and architecture) to disk.

    :param name: the base file name of the checkpoint (e.g. ``"latest"`` or ``"best"``)
    :param network: the network whose state dict to save
    :param learner: the learner whose optimizer state to save
    :param state: the resumable training counters
    :param config: the configuration whose architecture fields are stored for a safe resume
    :return: the path the checkpoint was written to
    """
    path = _checkpoint_dir().joinpath(f"{name}.pt")
    th.save(
        {
            "network": network.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "state": vars(state),
            "config": vars(config),
        },
        path,
    )
    return path


def load_checkpoint(path: str, network: StochasticMuZeroNetwork, learner: MuZeroLearner) -> TrainState:
    """
    Restore a checkpoint's network, optimizer and counters in place and return the counters.

    :param path: the checkpoint file path to load
    :param network: the network to load the saved parameters into
    :param learner: the learner whose optimizer state to restore
    :return: the restored :class:`TrainState` counters
    """
    payload = th.load(pathlib.Path(path), map_location=network.device, weights_only=False)
    network.load_state_dict(payload["network"])
    learner.optimizer.load_state_dict(payload["optimizer"])
    return TrainState(**payload["state"])


def _apply_lr(learner: MuZeroLearner, base_lr: float, fraction_done: float) -> float:
    """
    Linearly anneal the learning rate from ``base_lr`` down to ``LR_DECAY_FLOOR * base_lr``.

    :param learner: the learner whose optimizer learning rate to set
    :param base_lr: the starting (maximum) learning rate
    :param fraction_done: the fraction of the wall-clock budget already spent, in ``[0, 1]``
    :return: the learning rate that was applied
    """
    scale = 1.0 - (1.0 - LR_DECAY_FLOOR) * max(0.0, min(1.0, fraction_done))
    lr = base_lr * scale
    for group in learner.optimizer.param_groups:
        group["lr"] = lr
    return lr


def _format_losses(losses: dict[str, float]) -> str:
    """
    Format the component losses on a single, space-separated line.

    :param losses: the loss dict returned by :meth:`MuZeroLearner.train_step`
    :return: the formatted loss string
    """
    return " ".join(f"{key}={losses[key]:.4f}" for key in LOSS_KEYS if key in losses)


def _evaluate(network: StochasticMuZeroNetwork, config: MuZeroConfig, args: argparse.Namespace,
              state: TrainState) -> float:
    """
    Play a fixed number of games against random and return the win-rate, printing a timestamped line.

    :param network: the network under evaluation
    :param config: the configuration the network was built with
    :param args: the parsed command-line arguments (eval games / sims / seed)
    :param state: the current training counters (for the printed game index)
    :return: the win-rate over the evaluation games
    """
    result = play_muzero_vs_random(
        network, config, args.eval_games,
        np.random.default_rng(args.seed + state.game_index), args.eval_sims,
    )
    print(
        f"[{_timestamp()}] [eval] game={state.game_index} step={state.train_step_index} "
        f"win_rate={result['win_rate']:.4f} avg_points={result['avg_points']:.4f}",
        flush=True,
    )
    return result["win_rate"]


def _make_play_round(config: MuZeroConfig, game: BackgammonGame, network: StochasticMuZeroNetwork,
                     rng: np.random.Generator, args: argparse.Namespace) -> Callable[[], list[Trajectory]]:
    """
    Build the per-round self-play function for the ``--self-play`` path of the long run.

    The ``"batched"`` (default) path returns the FEATURE actor's :meth:`BatchedSelfPlayActor.play_games`
    unchanged, preserving this script's behaviour and the in-flight run's resume. The ``"single"`` path
    runs the proven BASELINE :class:`SelfPlayActor`, returning ``--parallel`` sequential
    :meth:`SelfPlayActor.play_game` games per round so the per-round trajectory count (and thus the
    train-step accounting and eval cadence) matches the batched path exactly.

    :param config: the configuration shared with the actor and its search
    :param game: the game factory producing fresh initial states
    :param network: the learned network driving the self-play search
    :param rng: the random number generator for chance sampling and action selection
    :param args: the parsed command-line arguments (the self-play path, parallel width and Gumbel m)
    :return: a zero-argument callable returning the round's list of ``--parallel`` trajectories
    """
    if args.self_play == "single":
        baseline_actor = SelfPlayActor(config, game, network, rng)
        return lambda: [baseline_actor.play_game() for _ in range(args.parallel)]
    batched_actor = BatchedSelfPlayActor(
        config, game, network, rng, num_parallel=args.parallel, num_considered=args.considered,
    )
    return batched_actor.play_games


def train(args: argparse.Namespace) -> None:
    """
    Run the resumable Stochastic-MuZero training loop until the wall-clock cap.

    Self-play follows ``args.self_play``: the batched Gumbel FEATURE actor (default) or the baseline
    single-game actor; both produce ``args.parallel`` trajectories per round so the rest of the loop
    (replay, learner steps, eval cadence and checkpointing) is identical for the two paths.

    :param args: the parsed command-line arguments controlling every aspect of the run
    """
    config = build_config(args)
    device_name = th.cuda.get_device_name(0) if config.device == "cuda" else "cpu"
    print(
        f"[{_timestamp()}] [device] {config.device} ({device_name}) state_channels={config.state_channels} "
        f"hidden={config.hidden_sizes} self_play={args.self_play} parallel={args.parallel} "
        f"sims={config.num_simulations} considered={args.considered} batch={config.batch_size}",
        flush=True,
    )

    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(args.seed)
    network = build_network(config)
    buffer = MuZeroReplayBuffer(config)
    play_round = _make_play_round(config, game, network, rng, args)
    learner = MuZeroLearner(config, network)

    state = TrainState()
    if args.resume:
        state = load_checkpoint(args.resume, network, learner)
        print(f"[{_timestamp()}] [resume] from {args.resume} game={state.game_index} "
              f"step={state.train_step_index} best_win_rate={state.best_win_rate:.4f}", flush=True)

    start = time.time() - state.elapsed_seconds
    last_checkpoint = time.time()
    last_losses: dict[str, float] = {}

    while True:
        elapsed = time.time() - start
        if elapsed >= args.max_seconds:
            print(f"[{_timestamp()}] [budget] reached {args.max_seconds:.0f}s wall-clock; stopping", flush=True)
            break

        last_losses = _play_and_train(play_round, buffer, learner, config, args, state, rng, start)

        if state.game_index % args.eval_every < args.parallel:
            win_rate = _evaluate(network, config, args, state)
            state.elapsed_seconds = time.time() - start
            if win_rate > state.best_win_rate:
                state.best_win_rate = win_rate
                save_checkpoint("best", network, learner, state, config)
                print(f"[{_timestamp()}] [best] new best win_rate={win_rate:.4f} (checkpoint saved)", flush=True)

        if time.time() - last_checkpoint >= args.checkpoint_minutes * 60.0:
            state.elapsed_seconds = time.time() - start
            path = save_checkpoint("latest", network, learner, state, config)
            last_checkpoint = time.time()
            print(f"[{_timestamp()}] [checkpoint] {path} game={state.game_index} "
                  f"step={state.train_step_index} {_format_losses(last_losses)}", flush=True)

    state.elapsed_seconds = time.time() - start
    save_checkpoint("latest", network, learner, state, config)
    final_win_rate = _evaluate(network, config, args, state)
    if final_win_rate > state.best_win_rate:
        state.best_win_rate = final_win_rate
        save_checkpoint("best", network, learner, state, config)
    print(f"[{_timestamp()}] [done] games={state.game_index} steps={state.train_step_index} "
          f"best_win_rate={state.best_win_rate:.4f} final_win_rate={final_win_rate:.4f}", flush=True)


def _play_and_train(play_round: Callable[[], list[Trajectory]], buffer: MuZeroReplayBuffer,
                    learner: MuZeroLearner, config: MuZeroConfig, args: argparse.Namespace,
                    state: TrainState, rng: np.random.Generator, start: float) -> dict[str, float]:
    """
    Play one round of self-play games, store them, and take the per-game learner steps.

    :param play_round: the per-round self-play function (the batched feature actor or baseline actor)
    :param buffer: the replay buffer the trajectories are stored in
    :param learner: the learner taking the gradient steps
    :param config: the configuration providing the batch size
    :param args: the parsed command-line arguments (train steps per game, base learning rate)
    :param state: the training counters, advanced in place
    :param rng: the random number generator driving batch sampling
    :param start: the monotonic start time used for the learning-rate annealing schedule
    :return: the most recent loss dict (empty until the buffer is warm)
    """
    trajectories = play_round()
    games_added = 0
    for trajectory in trajectories:
        buffer.save(trajectory)
        if len(trajectory) > 0:
            games_added += 1
    state.game_index += games_added

    elapsed = time.time() - start
    print(
        f"[{_timestamp()}] [self-play] games={state.game_index} (+{games_added}) "
        f"buffer_steps={len(buffer)} elapsed={elapsed:.0f}s",
        flush=True,
    )

    last_losses: dict[str, float] = {}
    if len(buffer) >= config.batch_size:
        fraction_done = min(1.0, elapsed / args.max_seconds)
        lr = _apply_lr(learner, args.lr, fraction_done)
        total_steps = args.train_steps_per_game * games_added
        for _ in range(total_steps):
            last_losses = learner.train_step(buffer.sample_batch(rng))
            state.train_step_index += 1
        if last_losses:
            print(f"[{_timestamp()}] [train] step={state.train_step_index} lr={lr:.2e} "
                  f"{_format_losses(last_losses)}", flush=True)
    return last_losses


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for the long-training script.

    :return: the configured :class:`argparse.ArgumentParser`
    """
    parser = argparse.ArgumentParser(description="Resumable batched Gumbel Stochastic-MuZero long training.")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="wall-clock budget (s)")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL, help="parallel self-play games (K)")
    parser.add_argument("--sims", type=int, default=DEFAULT_TRAIN_SIMS, help="self-play search simulations")
    parser.add_argument("--considered", type=int, default=DEFAULT_CONSIDERED, help="Gumbel considered actions m")
    parser.add_argument(
        "--self-play", choices=["batched", "single"], default=DEFAULT_SELF_PLAY,
        help="self-play path: batched Gumbel feature actor (default) or baseline single-game actor",
    )
    parser.add_argument(
        "--train-steps-per-game", type=int, default=DEFAULT_TRAIN_STEPS_PER_GAME,
        help="learner steps per completed game once warm",
    )
    parser.add_argument("--eval-sims", type=int, default=DEFAULT_EVAL_SIMS, help="evaluation search simulations")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY, help="evaluate every N games")
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES, help="games per evaluation")
    parser.add_argument(
        "--checkpoint-minutes", type=float, default=DEFAULT_CHECKPOINT_MINUTES,
        help="checkpoint the latest network every N minutes",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="base (max) learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay")
    parser.add_argument(
        "--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT,
        help="weight of the value loss term (AlphaZero-style 1.0 by default)",
    )
    parser.add_argument("--unroll-steps", type=int, default=DEFAULT_UNROLL_STEPS, help="unroll length K")
    parser.add_argument("--td-steps", type=int, default=DEFAULT_TD_STEPS, help="n-step bootstrap horizon")
    parser.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT, help="reward discount")
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY, help="replay step capacity")
    parser.add_argument("--state-channels", type=int, default=DEFAULT_STATE_CHANNELS, help="latent state width")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help="hidden-layer width (both layers)")
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE, help="chance codebook size")
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT_SIZE, help="value/reward support size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument("--resume", type=str, default="", help="path to a checkpoint to resume from")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=_default_device(),
        help="torch device (default: cuda if available else cpu)",
    )
    return parser


def main() -> None:
    """Parse the command-line arguments and run the resumable long-training loop."""
    args = _build_parser().parse_args()
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
