"""Tests for the Gumbel MuZero root primitives, the batched search and the batched self-play actor."""
import math

import numpy as np
import torch as th

from rlgammon.game.mock_game import MockGame
from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS
from rlgammon.muzero.mcts.gumbel import (
    gumbel_improved_policy,
    sample_gumbel,
    sequential_halving_schedule,
    sigma,
)
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.self_play.batched_actor import BatchedSelfPlayActor

# Tiny dimensions keeping the batched search / self-play tests fast and pyspiel-free.
NUM_ACTIONS = 8
OBSERVATION_SIZE = 198
STATE_CHANNELS = 16
HIDDEN_SIZES = (16,)
CODEBOOK_SIZE = 2
VALUE_SUPPORT_SIZE = 3
REWARD_SUPPORT_SIZE = 3
NUM_SIMULATIONS = 16
NUM_CONSIDERED = 4
NUM_PARALLEL = 6
SEED = 7
# Tolerance for the distribution-sums-to-one checks.
SUM_TOLERANCE = 1e-5
# The number of players in the mock game (WHITE and BLACK).
NUM_PLAYERS = 2


def _build_config() -> MuZeroConfig:
    """
    Build a tiny MuZeroConfig sized for the batched-search and self-play tests.

    :return: a small :class:`MuZeroConfig`
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
        seed=SEED,
    )


def test_sequential_halving_schedule_respects_budget() -> None:
    """Test that the schedule spans the right number of phases and never overspends the budget."""
    schedule = sequential_halving_schedule(NUM_SIMULATIONS, NUM_CONSIDERED)

    assert len(schedule) == math.ceil(math.log2(NUM_CONSIDERED))
    assert all(per_action >= 1 for per_action in schedule)

    # Replaying the halving, the total simulations spent must not exceed the budget.
    remaining_actions = NUM_CONSIDERED
    spent = 0
    for per_action in schedule:
        spent += per_action * remaining_actions
        remaining_actions = max(1, remaining_actions // 2)
    assert spent <= NUM_SIMULATIONS


def test_sequential_halving_single_action() -> None:
    """Test that a single considered action collapses to one phase consuming the whole budget."""
    assert sequential_halving_schedule(NUM_SIMULATIONS, 1) == [NUM_SIMULATIONS]


def test_sample_gumbel_shape_and_finiteness() -> None:
    """Test that the Gumbel sampler returns the requested number of finite variates."""
    gumbel = sample_gumbel(NUM_ACTIONS, np.random.default_rng(SEED))

    assert gumbel.shape == (NUM_ACTIONS,)
    assert bool(th.isfinite(gumbel).all())


def test_gumbel_improved_policy_prefers_high_q() -> None:
    """Test that, at equal priors, the improved policy puts more mass on higher completed-Q actions."""
    logits = th.zeros(3)
    completed_q = th.tensor([0.0, 1.0, -1.0])

    improved = gumbel_improved_policy(logits, completed_q, max_visit=4)

    assert abs(float(improved.sum()) - 1.0) < SUM_TOLERANCE
    assert improved[1] > improved[0] > improved[2]


def test_sigma_scales_with_visits() -> None:
    """Test that the sigma transform grows with the maximum visit count for a fixed positive value."""
    value = th.tensor([1.0])

    low = float(sigma(value, max_visit=0)[0])
    high = float(sigma(value, max_visit=100)[0])

    assert high > low > 0.0


def test_batched_search_returns_valid_results() -> None:
    """Test that the batched Gumbel search returns one legal action and a valid policy per tree."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    mcts = BatchedGumbelMCTS(config, network, np.random.default_rng(SEED), num_considered=NUM_CONSIDERED)

    legal_actions = [[0, 1, 2, 3], [1, 3, 5], [7], [0, 2, 4, 6]]
    observations = th.randn(len(legal_actions), OBSERVATION_SIZE)
    results = mcts.run_batch(observations, legal_actions)

    assert len(results) == len(legal_actions)
    for result, legal in zip(results, legal_actions, strict=True):
        assert result.action in legal
        assert set(result.policy) == set(legal)
        assert abs(sum(result.policy.values()) - 1.0) < SUM_TOLERANCE


def test_batched_search_single_legal_action_is_forced() -> None:
    """Test that a tree with one legal action always selects it with full policy mass."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    mcts = BatchedGumbelMCTS(config, network, np.random.default_rng(SEED), num_considered=NUM_CONSIDERED)
    forced_action = 5

    results = mcts.run_batch(th.randn(1, OBSERVATION_SIZE), [[forced_action]])

    assert results[0].action == forced_action
    assert abs(results[0].policy[forced_action] - 1.0) < SUM_TOLERANCE


def test_batched_self_play_produces_valid_trajectories() -> None:
    """Test that batched self-play returns one valid trajectory per parallel game on the mock."""
    config = _build_config()
    network = StochasticMuZeroNetwork(config)
    actor = BatchedSelfPlayActor(
        config, MockGame(), network, np.random.default_rng(SEED),
        num_parallel=NUM_PARALLEL, num_considered=NUM_CONSIDERED,
    )

    trajectories = actor.play_games()

    assert len(trajectories) == NUM_PARALLEL
    for trajectory in trajectories:
        assert len(trajectory) > 0
        assert len(trajectory.returns) == NUM_PLAYERS
        # Only the final step carries a reward; every recorded policy is a valid distribution.
        assert all(step.reward == 0.0 for step in trajectory.steps[:-1])
        for step in trajectory.steps:
            assert abs(sum(step.policy.values()) - 1.0) < SUM_TOLERANCE
