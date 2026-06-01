"""Smoke tests for the Stochastic MuZero self-play actor on the mock game (no pyspiel)."""
import numpy as np

from rlgammon.game.mock_game import MAX_STEP, MockGame
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.self_play.actor import SelfPlayActor

# Tiny network/search dimensions keeping a full self-play game fast.
NUM_ACTIONS = 8
OBSERVATION_SIZE = 198
STATE_CHANNELS = 16
HIDDEN_SIZES = (16,)
CODEBOOK_SIZE = 2
VALUE_SUPPORT_SIZE = 3
REWARD_SUPPORT_SIZE = 3
NUM_SIMULATIONS = 4
# A fixed seed so the smoke test is deterministic.
SEED = 7
# Tolerance for the policy-sums-to-one check.
SUM_TOLERANCE = 1e-5
# The set of valid terminal returns for the mock race game.
VALID_RETURNS = {-1.0, 0.0, 1.0}
# The mock observation index carrying ``dice / MAX_STEP`` (see MockState.observation_tensor).
DICE_OBS_INDEX = 2
# The number of players in the mock game (WHITE and BLACK).
NUM_PLAYERS = 2


def _build_config() -> MuZeroConfig:
    """
    Build a tiny MuZeroConfig sized for a fast self-play smoke test.

    :return: a MuZeroConfig with small network and search dimensions
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        state_channels=STATE_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        codebook_size=CODEBOOK_SIZE,
        value_support_size=VALUE_SUPPORT_SIZE,
        reward_support_size=REWARD_SUPPORT_SIZE,
        num_simulations=NUM_SIMULATIONS,
    )


def test_play_game_returns_non_empty_trajectory() -> None:
    """Test that a self-play game on the mock returns a non-empty trajectory with stored returns."""
    config = _build_config()
    actor = SelfPlayActor(config, MockGame(), StochasticMuZeroNetwork(config), np.random.default_rng(SEED))

    trajectory = actor.play_game()

    assert len(trajectory) > 0
    assert len(trajectory.returns) == NUM_PLAYERS


def test_recorded_policies_and_actions_are_legal() -> None:
    """Test that each recorded policy sums to one over its (legal) keys and the played action is legal."""
    config = _build_config()
    actor = SelfPlayActor(config, MockGame(), StochasticMuZeroNetwork(config), np.random.default_rng(SEED))

    trajectory = actor.play_game()

    for step in trajectory.steps:
        # The mock encodes ``dice / MAX_STEP`` at observation index 2; legal steps are ``1 .. dice``.
        dice = round(step.observation[DICE_OBS_INDEX] * MAX_STEP)
        legal = set(range(1, dice + 1))
        assert set(step.policy) == legal
        assert step.action in legal
        assert abs(sum(step.policy.values()) - 1.0) < SUM_TOLERANCE


def test_final_reward_is_a_valid_return() -> None:
    """Test that the final recorded step's reward is one of the mock game's signed returns."""
    config = _build_config()
    actor = SelfPlayActor(config, MockGame(), StochasticMuZeroNetwork(config), np.random.default_rng(SEED))

    trajectory = actor.play_game()

    assert trajectory.steps[-1].reward in VALID_RETURNS
    # Every non-terminal step keeps a zero reward.
    assert all(step.reward == 0.0 for step in trajectory.steps[:-1])
