"""Tests for the Stochastic MuZero Monte-Carlo tree search using a deterministic fake network."""
import numpy as np
import torch as th

from rlgammon.muzero.mcts.node import ChanceNode, DecisionNode, MinMaxStats
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_types import AfterstateOutput, MuZeroConfig, NetworkOutput

# Tiny dimensions keeping the fake network trivial and the search fast.
NUM_ACTIONS = 4
CODEBOOK_SIZE = 2
VALUE_SUPPORT_SIZE = 3
REWARD_SUPPORT_SIZE = 3
STATE_CHANNELS = 8
OBSERVATION_SIZE = 6
NUM_SIMULATIONS = 16
LEGAL_ACTIONS = [0, 1, 2, 3]
# The action whose subtree the biased fake network makes look the most valuable.
PREFERRED_ACTION = 2
# Reference points for the MinMaxStats normalization test.
UNSEEN_VALUE = 0.7
MIDPOINT_VALUE = 1.0
MIDPOINT_NORMALIZED = 0.5


def _build_config() -> MuZeroConfig:
    """
    Build a tiny MuZeroConfig sized for the fake network.

    :return: a MuZeroConfig with small dimensions and a modest simulation budget
    """
    return MuZeroConfig(
        observation_size=OBSERVATION_SIZE,
        num_actions=NUM_ACTIONS,
        state_channels=STATE_CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        value_support_size=VALUE_SUPPORT_SIZE,
        reward_support_size=REWARD_SUPPORT_SIZE,
        num_simulations=NUM_SIMULATIONS,
    )


class _FakeNetwork:
    """
    A deterministic stand-in for :class:`StochasticMuZeroNetwork` exposing the inference interface.

    All heads return fixed outputs sized from the tiny configuration: uniform policy logits, uniform
    chance logits and a constant value/reward. The categorical-to-scalar converters read the first
    channel of the latent so that a caller can steer the decoded value through the produced state.
    """

    def __init__(self, preferred_action: int | None = None) -> None:
        """
        Construct the fake network, optionally biasing one action's afterstate to look valuable.

        :param preferred_action: if set, the afterstate for this action encodes a higher scalar value
        """
        self.preferred_action = preferred_action

    def eval(self) -> "_FakeNetwork":
        """
        Mimic :meth:`torch.nn.Module.eval` by returning ``self`` (the fake has no mode to toggle).

        :return: this network instance
        """
        return self

    def initial_inference(self, observation: th.Tensor) -> NetworkOutput:
        """
        Produce a root latent with uniform policy and a constant value.

        :param observation: the observation tensor of shape ``[B, observation_size]``
        :return: a NetworkOutput with uniform policy logits and zero reward/value logits
        """
        batch = observation.shape[0]
        return NetworkOutput(
            value=th.zeros((batch, VALUE_SUPPORT_SIZE)),
            reward=th.zeros((batch, REWARD_SUPPORT_SIZE)),
            policy_logits=th.zeros((batch, NUM_ACTIONS)),
            state=th.zeros((batch, STATE_CHANNELS)),
        )

    def recurrent_inference_afterstate(self, state: th.Tensor, action_onehot: th.Tensor) -> AfterstateOutput:
        """
        Produce an afterstate with a uniform chance prior, encoding the selected action's value.

        The first channel of the afterstate is set to ``1`` for the preferred action and ``0``
        otherwise, which :meth:`value_to_scalar` then decodes into the afterstate's scalar value.

        :param state: the latent state tensor of shape ``[B, state_channels]``
        :param action_onehot: the one-hot action tensor of shape ``[B, num_actions]``
        :return: an AfterstateOutput with uniform chance logits and an action-dependent value channel
        """
        batch = state.shape[0]
        afterstate = th.zeros((batch, STATE_CHANNELS))
        if self.preferred_action is not None:
            preferred_mass = action_onehot[:, self.preferred_action]
            afterstate[:, 0] = preferred_mass
        return AfterstateOutput(
            chance_logits=th.zeros((batch, CODEBOOK_SIZE)),
            afterstate_value=th.zeros((batch, VALUE_SUPPORT_SIZE)),
            afterstate=afterstate,
        )

    def recurrent_inference_state(self, afterstate: th.Tensor,
                                  chance_onehot: th.Tensor) -> NetworkOutput:  # noqa: ARG002
        """
        Produce a next-state latent that propagates the afterstate's value channel.

        :param afterstate: the afterstate tensor of shape ``[B, state_channels]``
        :param chance_onehot: the one-hot chance tensor of shape ``[B, codebook_size]``
        :return: a NetworkOutput whose state carries the afterstate's first channel forward
        """
        batch = afterstate.shape[0]
        state = th.zeros((batch, STATE_CHANNELS))
        state[:, 0] = afterstate[:, 0]
        return NetworkOutput(
            value=th.zeros((batch, VALUE_SUPPORT_SIZE)),
            reward=th.zeros((batch, REWARD_SUPPORT_SIZE)),
            policy_logits=th.zeros((batch, NUM_ACTIONS)),
            state=state,
        )

    def value_to_scalar(self, value_logits: th.Tensor) -> th.Tensor:
        """
        Decode value logits to a scalar (always zero for this fake's all-zero value heads).

        :param value_logits: the value logits tensor of shape ``[B, value_support_size]``
        :return: a zero scalar tensor of shape ``[B]``
        """
        return th.zeros(value_logits.shape[0])

    def reward_to_scalar(self, reward_logits: th.Tensor) -> th.Tensor:
        """
        Decode reward logits to a scalar (always zero for this fake's all-zero reward heads).

        :param reward_logits: the reward logits tensor of shape ``[B, reward_support_size]``
        :return: a zero scalar tensor of shape ``[B]``
        """
        return th.zeros(reward_logits.shape[0])


class _BiasedValueNetwork(_FakeNetwork):
    """
    A fake network that makes the preferred action's afterstate decode a strictly higher value.

    The scalar value is read straight from the first channel of whatever value logits are passed
    in (a stateless, call-order-independent decoding). The preferred action's afterstate prediction
    sets that channel to one, so its subtree looks more valuable and attracts the most visits, while
    every other transition (and the root) decodes to zero.
    """

    def value_to_scalar(self, value_logits: th.Tensor) -> th.Tensor:
        """
        Decode the first channel of the value logits directly as the scalar value.

        :param value_logits: the value logits tensor of shape ``[B, value_support_size]``
        :return: the first logit channel as a scalar tensor of shape ``[B]``
        """
        return value_logits[:, 0]

    def recurrent_inference_afterstate(self, state: th.Tensor, action_onehot: th.Tensor) -> AfterstateOutput:
        """
        Produce an afterstate whose value logits encode a positive scalar for the preferred action.

        :param state: the latent state tensor of shape ``[B, state_channels]``
        :param action_onehot: the one-hot action tensor of shape ``[B, num_actions]``
        :return: an AfterstateOutput whose value-logit channel zero flags the preferred action
        """
        output = super().recurrent_inference_afterstate(state, action_onehot)
        if self.preferred_action is not None:
            output.afterstate_value[:, 0] = action_onehot[:, self.preferred_action]
        return output


def _build_search(network: _FakeNetwork, *, seed: int = 0) -> StochasticMuZeroMCTS:
    """
    Build a search bound to the fake network with a seeded generator.

    :param network: the fake network driving the search
    :param seed: the seed for the numpy generator used for exploration noise
    :return: a configured StochasticMuZeroMCTS instance
    """
    config = _build_config()
    # The fake network is structurally compatible with the real network's inference interface.
    return StochasticMuZeroMCTS(config, network, np.random.default_rng(seed))  # type: ignore[arg-type]


def test_min_max_stats_normalize() -> None:
    """Test that MinMaxStats is identity before two values and rescales to [0, 1] afterwards."""
    stats = MinMaxStats()
    # With no usable bounds the value passes through unchanged.
    assert stats.normalize(UNSEEN_VALUE) == UNSEEN_VALUE
    stats.update(0.0)
    stats.update(2.0)
    assert stats.normalize(0.0) == 0.0
    assert stats.normalize(2.0) == 1.0
    assert stats.normalize(MIDPOINT_VALUE) == MIDPOINT_NORMALIZED


def test_run_visit_counts_sum_to_simulations() -> None:
    """Test that the visit counts sum to the simulation budget and all keys are legal actions."""
    search = _build_search(_FakeNetwork())

    visit_counts = search.run(
        th.zeros(OBSERVATION_SIZE), legal_actions=LEGAL_ACTIONS, add_exploration_noise=False,
    )

    assert set(visit_counts) == set(LEGAL_ACTIONS)
    assert all(action in LEGAL_ACTIONS for action in visit_counts)
    assert sum(visit_counts.values()) == NUM_SIMULATIONS


def test_run_prefers_high_value_action() -> None:
    """Test that the action whose subtree is made to look valuable receives the most visits."""
    search = _build_search(_BiasedValueNetwork(preferred_action=PREFERRED_ACTION))

    visit_counts = search.run(
        th.zeros(OBSERVATION_SIZE), legal_actions=LEGAL_ACTIONS, add_exploration_noise=False,
    )

    best_action = max(visit_counts, key=lambda action: visit_counts[action])
    assert best_action == PREFERRED_ACTION


def test_run_node_alternation_and_to_play() -> None:
    """Test the decision/chance alternation and the to_play sign flip across a chance roll."""
    search = _build_search(_FakeNetwork())

    search.run(th.zeros(OBSERVATION_SIZE), legal_actions=LEGAL_ACTIONS, add_exploration_noise=False)

    root = search._root
    assert root is not None
    assert root.to_play == 0
    for chance_child in root.children.values():
        assert isinstance(chance_child, ChanceNode)
        # A chance node keeps the parent decision node's player (still pre-opponent-roll).
        assert chance_child.to_play == root.to_play
        for decision_grandchild in chance_child.children.values():
            assert isinstance(decision_grandchild, DecisionNode)
            # The decision node after a chance roll belongs to the opponent.
            assert decision_grandchild.to_play == 1 - root.to_play


def test_run_with_exploration_noise() -> None:
    """Test that enabling seeded Dirichlet noise still yields valid visit counts summing correctly."""
    search = _build_search(_FakeNetwork(), seed=42)

    visit_counts = search.run(
        th.zeros(OBSERVATION_SIZE), legal_actions=LEGAL_ACTIONS, add_exploration_noise=True,
    )

    assert set(visit_counts) == set(LEGAL_ACTIONS)
    assert sum(visit_counts.values()) == NUM_SIMULATIONS
