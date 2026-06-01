"""Stochastic MuZero self-play training with periodic evaluation against a random opponent.

This wires the three Stochastic MuZero components into one loop: a
:class:`~rlgammon.muzero.self_play.actor.SelfPlayActor` plays a full game and stores its trajectory in a
:class:`~rlgammon.muzero.replay.replay_buffer.MuZeroReplayBuffer`; once the buffer holds at least one
batch, a :class:`~rlgammon.muzero.training.learner.MuZeroLearner` takes several gradient steps; and every
few games the network is evaluated against a uniform-random opponent via
:func:`~scripts.eval_vs_random.play_muzero_vs_random`. Training stops at a fixed number of games or once a
wall-clock budget is exhausted (whichever comes first), so the run finishes deterministically. The final
network state dict is saved under ``rlgammon/muzero/training/saved_agents``.
"""
import argparse
from collections.abc import Callable
import time
import uuid

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

# Default maximum number of self-play games (the wall-clock budget usually stops the run first).
DEFAULT_GAMES = 256
# Default number of learner train steps taken per completed game once the buffer is warm.
DEFAULT_TRAIN_STEPS_PER_GAME = 4
# Default number of search simulations during self-play (Gumbel needs few).
DEFAULT_TRAIN_SIMS = 32
# Default number of Gumbel considered root actions (the ``m`` of Gumbel-top-k + sequential halving).
DEFAULT_CONSIDERED = 16
# Default number of games advanced simultaneously per batched self-play call (the GPU throughput key).
DEFAULT_PARALLEL = 32
# Default self-play path: the batched Gumbel FEATURE actor (preserves the current script behaviour).
# Pass ``--self-play single`` to instead run the proven BASELINE single-game actor + single-tree search.
DEFAULT_SELF_PLAY = "batched"
# Default number of search simulations during evaluation (kept modest as eval is the wall-clock bottleneck).
DEFAULT_EVAL_SIMS = 50
# Default cadence (in games) at which to evaluate against random.
DEFAULT_EVAL_EVERY = 64
# Default number of games per periodic evaluation (enough for a low-variance win-rate).
DEFAULT_EVAL_GAMES = 50
# Default random seed.
DEFAULT_SEED = 0
# Default wall-clock training budget in seconds (so a run finishes deterministically; ~18 minutes).
DEFAULT_MAX_SECONDS = 1080

# MuZero observation size (board features only, dice dropped).
MUZERO_OBSERVATION_SIZE = 198
# MuZero action-space size for the OpenSpiel backgammon engine.
MUZERO_NUM_ACTIONS = 1352
# Latent state width used for this bounded-budget training configuration (affordable on GPU).
STATE_CHANNELS = 256
# Hidden-layer widths used throughout the network.
HIDDEN_SIZES = (256, 256)
# Codebook size for the chance encoder / stochastic transitions.
CODEBOOK_SIZE = 32
# Unroll length used by the learner.
UNROLL_STEPS = 5
# Batch size for each gradient step.
BATCH_SIZE = 256
# Learning rate.
LEARNING_RATE = 2e-3
# Categorical value/reward support size.
SUPPORT_SIZE = 21
# Value-loss weight: equal to the policy weight (AlphaZero-style) so the value head -- which the
# Gumbel search leans on to improve the policy -- learns at full strength (the MuZero/Atari default of
# 0.25 leaves the value too weak for a board game where search quality hinges on the value estimate).
VALUE_LOSS_WEIGHT = 1.0

# The loss keys printed on each training line.
LOSS_KEYS = ("total", "value", "policy", "reward", "chance", "commitment")
# Learning-rate decay floor: the rate is linearly annealed from the base ``lr`` down to this fraction
# of it over the wall-clock budget, which stabilises the late-run policy/value heads (a fixed high
# rate was observed to let the win-rate regress after an early peak).
LR_DECAY_FLOOR = 0.2


def _default_device() -> str:
    """
    Pick the default training device: ``"cuda"`` when a CUDA GPU is present, else ``"cpu"``.

    :return: ``"cuda"`` if :func:`torch.cuda.is_available` is ``True``, otherwise ``"cpu"``
    """
    return "cuda" if th.cuda.is_available() else "cpu"


def build_config(*, train_sims: int, seed: int, device: str = "cpu",
                 state_channels: int = STATE_CHANNELS,
                 hidden_sizes: tuple[int, ...] = HIDDEN_SIZES) -> MuZeroConfig:
    """
    Build a reasonably small but capable configuration sized for a bounded-budget run.

    :param train_sims: the number of search simulations used during self-play
    :param seed: the configuration seed shared across self-play, replay and the learner
    :param device: the torch device the network and training tensors live on (``"cpu"`` or ``"cuda"``)
    :param state_channels: the latent state width (a wider net is affordable on GPU)
    :param hidden_sizes: the hidden-layer widths (wider layers are affordable on GPU)
    :return: the assembled :class:`MuZeroConfig`
    """
    return MuZeroConfig(
        observation_size=MUZERO_OBSERVATION_SIZE,
        num_actions=MUZERO_NUM_ACTIONS,
        state_channels=state_channels,
        hidden_sizes=hidden_sizes,
        codebook_size=CODEBOOK_SIZE,
        num_simulations=train_sims,
        unroll_steps=UNROLL_STEPS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        value_support_size=SUPPORT_SIZE,
        reward_support_size=SUPPORT_SIZE,
        seed=seed,
        device=device,
    )


def _format_losses(losses: dict[str, float]) -> str:
    """
    Format the component losses on a single, space-separated line.

    :param losses: the loss dict returned by :meth:`MuZeroLearner.train_step`
    :return: the formatted loss string
    """
    return " ".join(f"{key}={losses[key]:.4f}" for key in LOSS_KEYS if key in losses)


def _apply_lr_decay(learner: MuZeroLearner, base_lr: float, fraction_done: float) -> float:
    """
    Linearly anneal the optimizer learning rate from ``base_lr`` to ``LR_DECAY_FLOOR * base_lr``.

    :param learner: the learner whose optimizer learning rate to set
    :param base_lr: the starting (maximum) learning rate
    :param fraction_done: the fraction of the wall-clock budget already spent, clamped to ``[0, 1]``
    :return: the learning rate that was applied
    """
    scale = 1.0 - (1.0 - LR_DECAY_FLOOR) * max(0.0, min(1.0, fraction_done))
    lr = base_lr * scale
    for group in learner.optimizer.param_groups:
        group["lr"] = lr
    return lr


def _make_play_round(self_play: str, config: MuZeroConfig, game: BackgammonGame,
                     network: StochasticMuZeroNetwork, rng: np.random.Generator,
                     parallel: int, considered: int) -> Callable[[], list[Trajectory]]:
    """
    Build the per-round self-play function for the chosen ``self_play`` path.

    The ``"batched"`` (default) path returns the FEATURE actor's :meth:`BatchedSelfPlayActor.play_games`
    unchanged (``parallel`` games batched together). The ``"single"`` path runs the proven BASELINE
    :class:`SelfPlayActor` and returns ``parallel`` sequential :meth:`SelfPlayActor.play_game` games per
    round, so each round yields the same number of trajectories as the batched path and the outer
    loop's train-step accounting is identical.

    :param self_play: ``"batched"`` for the feature actor or ``"single"`` for the baseline actor
    :param config: the configuration shared with the actor and its search
    :param game: the game factory producing fresh initial states
    :param network: the learned network driving the self-play search
    :param rng: the random number generator for self-play chance sampling and action selection
    :param parallel: the number ``K`` of games produced per round
    :param considered: the number ``m`` of Gumbel considered root actions (the ``batched`` path only)
    :return: a zero-argument callable returning the round's list of ``parallel`` trajectories
    """
    if self_play == "single":
        baseline_actor = SelfPlayActor(config, game, network, rng)
        return lambda: [baseline_actor.play_game() for _ in range(parallel)]
    batched_actor = BatchedSelfPlayActor(
        config, game, network, rng, num_parallel=parallel, num_considered=considered,
    )
    return batched_actor.play_games


def train(*, games: int, train_steps_per_game: int, train_sims: int, eval_sims: int,
          eval_every: int, eval_games: int, seed: int, max_seconds: float,
          out: str, device: str = "cpu", state_channels: int = STATE_CHANNELS,
          hidden_sizes: tuple[int, ...] = HIDDEN_SIZES, parallel: int = DEFAULT_PARALLEL,
          considered: int = DEFAULT_CONSIDERED,
          self_play: str = DEFAULT_SELF_PLAY) -> list[tuple[int, float, float]]:
    """
    Run the Stochastic-MuZero self-play / replay / learner loop with periodic evaluation.

    With the default ``self_play="batched"`` the FEATURE actor advances ``parallel`` real games at once
    with one batched Gumbel search per joint move (so every network inference batches many positions on
    the GPU); ``self_play="single"`` instead runs the proven BASELINE single-game actor + single-tree
    pUCT search, playing ``parallel`` games per round so the train-step accounting is unchanged.
    Training stops after ``games`` self-play games or once ``max_seconds`` of wall-clock time has
    elapsed, whichever comes first. Every ``eval_every`` games the network plays ``eval_games`` games
    against a uniform-random opponent and the win-rate / average points are printed and recorded.

    :param games: the maximum number of self-play games to play
    :param train_steps_per_game: the number of gradient steps per completed game once the buffer is warm
    :param train_sims: the number of search simulations during self-play
    :param eval_sims: the number of search simulations during evaluation
    :param eval_every: evaluate against random roughly every this many games
    :param eval_games: the number of games per evaluation
    :param seed: the seed shared across the config, self-play and evaluation generators
    :param max_seconds: the wall-clock training budget in seconds
    :param out: the base file name to save the final network state dict under
    :param device: the torch device the network and training tensors live on (``"cpu"`` or ``"cuda"``)
    :param state_channels: the latent state width (a wider net is affordable on GPU)
    :param hidden_sizes: the hidden-layer widths (wider layers are affordable on GPU)
    :param parallel: the number ``K`` of games per self-play round (batched together for ``batched``)
    :param considered: the number ``m`` of Gumbel considered root actions (the ``batched`` path only)
    :param self_play: ``"batched"`` for the feature actor (default) or ``"single"`` for the baseline
    :return: the recorded ``(game_index, win_rate, avg_points)`` evaluation curve
    """
    config = build_config(
        train_sims=train_sims, seed=seed, device=device,
        state_channels=state_channels, hidden_sizes=hidden_sizes,
    )
    device_name = th.cuda.get_device_name(0) if config.device == "cuda" else "cpu"
    print(
        f"[device] using {config.device} ({device_name}) "
        f"state_channels={config.state_channels} hidden_sizes={config.hidden_sizes} "
        f"self_play={self_play} parallel={parallel} sims={config.num_simulations} considered={considered}",
    )
    game = create_game(PossibleEngine.OPEN_SPIEL)
    rng = np.random.default_rng(seed)

    network = build_network(config)
    buffer = MuZeroReplayBuffer(config)
    play_round = _make_play_round(self_play, config, game, network, rng, parallel, considered)
    learner = MuZeroLearner(config, network)

    curve: list[tuple[int, float, float]] = []
    last_losses: dict[str, float] = {}
    start = time.time()
    game_index = 0
    train_step_index = 0
    last_eval_at = 0

    while game_index < games:
        trajectories = play_round()
        added = sum(1 for trajectory in trajectories if len(trajectory) > 0)
        for trajectory in trajectories:
            buffer.save(trajectory)
        game_index += added
        elapsed = time.time() - start
        print(f"[self-play] games {game_index}/{games} (+{added}) buffer_steps={len(buffer)} elapsed={elapsed:.1f}s")

        if len(buffer) >= config.batch_size:
            lr = _apply_lr_decay(learner, config.lr, min(1.0, elapsed / max_seconds))
            for _ in range(train_steps_per_game * added):
                last_losses = learner.train_step(buffer.sample_batch(rng))
                train_step_index += 1
            print(f"[train] step {train_step_index} lr={lr:.2e} {_format_losses(last_losses)}")

        if game_index - last_eval_at >= eval_every or game_index >= games:
            last_eval_at = game_index
            result = play_muzero_vs_random(
                network, config, eval_games, np.random.default_rng(seed + game_index), eval_sims,
            )
            curve.append((game_index, result["win_rate"], result["avg_points"]))
            print(
                f"[eval] game {game_index} win_rate={result['win_rate']:.4f} "
                f"avg_points={result['avg_points']:.4f}",
            )

        if time.time() - start >= max_seconds:
            print(f"[budget] reached wall-clock budget of {max_seconds:.0f}s after {game_index} games; stopping")
            break

    session_id = uuid.uuid4()
    learner.save(session_id, train_step_index, main_filename=out)
    print(
        f"[muzero] saved network rlgammon/muzero/training/saved_agents/"
        f"{out}-{session_id}-({train_step_index}).pt",
    )
    _print_curve(curve, last_losses)
    return curve


def _print_curve(curve: list[tuple[int, float, float]], last_losses: dict[str, float]) -> None:
    """
    Print the recorded win-rate-vs-random curve and the final loss values.

    :param curve: the recorded ``(game_index, win_rate, avg_points)`` evaluation points
    :param last_losses: the most recent loss dict from the learner
    """
    print("[curve] game win_rate avg_points")
    for game_index, win_rate, avg_points in curve:
        print(f"[curve] {game_index} {win_rate:.4f} {avg_points:.4f}")
    if last_losses:
        print(f"[final-loss] {_format_losses(last_losses)}")


def main() -> None:
    """Parse the command-line arguments and run the bounded-budget Stochastic MuZero training loop."""
    parser = argparse.ArgumentParser(description="Train Stochastic MuZero with periodic random-opponent evaluation.")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="maximum number of self-play games")
    parser.add_argument(
        "--train-steps-per-game", type=int, default=DEFAULT_TRAIN_STEPS_PER_GAME,
        help="gradient steps per game once the buffer is warm",
    )
    parser.add_argument("--sims", type=int, default=DEFAULT_TRAIN_SIMS, help="self-play search simulations")
    parser.add_argument("--eval-sims", type=int, default=DEFAULT_EVAL_SIMS, help="evaluation search simulations")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY, help="evaluate every N games")
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES, help="games per evaluation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    parser.add_argument(
        "--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
        help="wall-clock training budget in seconds",
    )
    parser.add_argument("--out", type=str, default="stochastic-muzero", help="base file name for the saved network")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=_default_device(),
        help="torch device for the network and training (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--state-channels", type=int, default=STATE_CHANNELS,
        help="latent state width (a wider net is affordable on GPU)",
    )
    parser.add_argument(
        "--hidden", type=int, default=HIDDEN_SIZES[0],
        help="hidden-layer width applied to both layers (wider is affordable on GPU)",
    )
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL,
        help="parallel self-play games advanced per batched search (the GPU throughput key)",
    )
    parser.add_argument(
        "--considered", type=int, default=DEFAULT_CONSIDERED,
        help="Gumbel considered root actions (m of Gumbel-top-k + sequential halving)",
    )
    parser.add_argument(
        "--self-play", choices=["batched", "single"], default=DEFAULT_SELF_PLAY,
        help="self-play path: batched Gumbel feature actor (default) or baseline single-game actor",
    )
    args = parser.parse_args()

    train(
        games=args.games,
        train_steps_per_game=args.train_steps_per_game,
        train_sims=args.sims,
        eval_sims=args.eval_sims,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        seed=args.seed,
        max_seconds=args.max_seconds,
        out=args.out,
        device=resolve_device(args.device),
        state_channels=args.state_channels,
        hidden_sizes=(args.hidden, args.hidden),
        parallel=args.parallel,
        considered=args.considered,
        self_play=args.self_play,
    )


if __name__ == "__main__":
    main()
