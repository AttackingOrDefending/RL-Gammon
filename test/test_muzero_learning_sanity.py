"""Sanity tests proving the Stochastic MuZero value learning works with the correct sign.

These tests guard the two-player value-target perspective fix in
:meth:`~rlgammon.muzero.replay.trajectory.Trajectory._compute_value_target`. In a two-player game
the player to move alternates every ply, so the n-step value target at a step must be expressed from
THAT step's mover's perspective (bootstrap and folded rewards sign-flipped across plies). A wrong
sign makes the value target point at the opponent's outcome and the network learns the inverse.

The first test asserts the targets themselves carry the right sign on a hand-made won game; the
second overfits a tiny fixed batch and checks the predicted root values converge to the sign of the
real outcome (positive for the winner-to-move, negative for the loser-to-move) and that the value
loss drops sharply.
"""
import numpy as np
import torch as th

from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.replay.replay_buffer import MuZeroReplayBuffer
from rlgammon.muzero.replay.trajectory import Step, Trajectory
from rlgammon.muzero.training.learner import MuZeroLearner

# Tiny network/search dimensions keeping the overfit fast and pyspiel-free.
NUM_ACTIONS = 4
OBSERVATION_SIZE = 198
STATE_CHANNELS = 32
HIDDEN_SIZES = (32,)
CODEBOOK_SIZE = 2
VALUE_SUPPORT_SIZE = 7
REWARD_SUPPORT_SIZE = 7
UNROLL_STEPS = 2
TD_STEPS = 10
BATCH_SIZE = 16
# A value-dominated loss so the overfit isolates the value signal under test.
VALUE_LOSS_WEIGHT = 1.0
# Number of overfit gradient steps; ample for a single fixed game on a tiny net.
OVERFIT_STEPS = 600
# A fixed seed so the overfit is deterministic.
SEED = 0
# The winner-to-move value must end positive and the loser-to-move value negative (the SIGN is what
# the perspective fix corrects; the buggy version drives these to the opposite signs). The exact
# magnitude is seed-sensitive (categorical head + compressing support transform), so the assertions
# check the sign with a tiny margin plus a clear separation between the two perspectives.
WIN_VALUE_FLOOR = 0.0
LOSS_VALUE_CEILING = 0.0
# The winner-to-move value must exceed the loser-to-move value by at least this margin.
PERSPECTIVE_SEPARATION = 0.3
# The value loss must drop to a small fraction of its initial value after overfitting.
VALUE_LOSS_DROP_RATIO = 0.5


def _build_config() -> MuZeroConfig:
    """
    Build a tiny MuZeroConfig sized for the overfit sanity test.

    :return: a small :class:`MuZeroConfig` with a wide-enough value support for ``+-1`` outcomes
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        state_channels=STATE_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        codebook_size=CODEBOOK_SIZE,
        unroll_steps=UNROLL_STEPS,
        td_steps=TD_STEPS,
        batch_size=BATCH_SIZE,
        value_support_size=VALUE_SUPPORT_SIZE,
        reward_support_size=REWARD_SUPPORT_SIZE,
        value_loss_weight=VALUE_LOSS_WEIGHT,
        seed=SEED,
    )


def _game_trajectory(*, mover0_wins: bool) -> Trajectory:
    """
    Build a short three-ply game whose to-move-0 player either wins or loses.

    Step 0 is the to-move-0 player's, step 1 the opponent's, step 2 the to-move-0 player's terminal
    move. Each step's observation is a distinct constant vector. When ``mover0_wins`` the terminal
    return is ``+1`` to the to-move-0 player (so the step-0 value target is positive), otherwise the
    same move loses (the terminal return is ``-1`` to the to-move-0 player, flipping every target).

    :param mover0_wins: whether the player to move at step 0 wins the game
    :return: the hand-made :class:`Trajectory`
    """
    steps = [
        Step(
            observation=[float(index + 1)] * OBSERVATION_SIZE,
            action=index % NUM_ACTIONS,
            reward=0.0,
            policy={index % NUM_ACTIONS: 1.0},
            value=0.0,
            to_play=index % 2,
        )
        for index in range(3)
    ]
    # The terminal move at step 2 belongs to the to-move-0 player; its reward is that player's return.
    steps[-1].reward = 1.0 if mover0_wins else -1.0
    returns = [1.0, -1.0] if mover0_wins else [-1.0, 1.0]
    return Trajectory(steps=steps, returns=returns)


def test_value_targets_follow_the_mover_perspective() -> None:
    """Test that the value targets carry the sign of the outcome from each step's mover perspective."""
    config = _build_config()
    won = _game_trajectory(mover0_wins=True)
    lost = _game_trajectory(mover0_wins=False)

    # Step 0 is the to-move-0 player's. When that player wins its step-0 target is positive; when it
    # loses, negative. The opponent's step 1 always carries the opposite sign of step 0.
    won_root = won.make_target(start_index=0, config=config).target_values[0]
    won_opponent = won.make_target(start_index=1, config=config).target_values[0]
    lost_root = lost.make_target(start_index=0, config=config).target_values[0]

    assert won_root > 0.0, f"winner-to-move target should be positive, got {won_root}"
    assert won_opponent < 0.0, f"opponent-to-move target should be negative, got {won_opponent}"
    assert lost_root < 0.0, f"loser-to-move target should be negative, got {lost_root}"


def _overfit_root_value(*, mover0_wins: bool) -> tuple[float, float, float]:
    """
    Overfit one fixed three-ply game and return the predicted root value and the value-loss drop.

    The SAME root observation is used whether the to-move-0 player wins or loses; only the outcome
    (and hence the perspective-correct target sign) changes, so the sign of the recovered root value
    is a direct read-out of the value-target perspective fix.

    :param mover0_wins: whether the player to move at the root wins the game
    :return: a tuple ``(root_value, first_value_loss, last_value_loss)``
    """
    config = _build_config()
    th.manual_seed(SEED)
    network = StochasticMuZeroNetwork(config)
    buffer = MuZeroReplayBuffer(config)
    buffer.save(_game_trajectory(mover0_wins=mover0_wins))
    learner = MuZeroLearner(config, network)
    rng = np.random.default_rng(SEED)

    first_value_loss = None
    last_value_loss = 0.0
    for _ in range(OVERFIT_STEPS):
        losses = learner.train_step(buffer.sample_batch(rng))
        if first_value_loss is None:
            first_value_loss = losses["value"]
        last_value_loss = losses["value"]
    assert first_value_loss is not None

    network.eval()
    root_obs = th.tensor([[1.0] * OBSERVATION_SIZE], dtype=th.float32)
    with th.no_grad():
        root_value = float(network.value_to_scalar(network.initial_inference(root_obs).value)[0])
    return root_value, first_value_loss, last_value_loss


def test_overfit_recovers_outcome_sign() -> None:
    """Test that overfitting recovers a positive root value for a win and a negative one for a loss."""
    win_value, win_first_loss, win_last_loss = _overfit_root_value(mover0_wins=True)
    loss_value, _, _ = _overfit_root_value(mover0_wins=False)

    assert win_last_loss < VALUE_LOSS_DROP_RATIO * win_first_loss, (
        f"value loss should drop sharply: {win_first_loss:.4f} -> {win_last_loss:.4f}"
    )
    assert win_value > WIN_VALUE_FLOOR, f"winner-to-move root value should be positive, got {win_value}"
    assert loss_value < LOSS_VALUE_CEILING, f"loser-to-move root value should be negative, got {loss_value}"
    assert win_value - loss_value > PERSPECTIVE_SEPARATION, (
        f"win/loss root values should separate: win={win_value:.4f} loss={loss_value:.4f}"
    )
