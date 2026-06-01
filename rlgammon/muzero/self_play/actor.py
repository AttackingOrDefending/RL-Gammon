"""Self-play actor driving Stochastic MuZero search over the real game to produce trajectories.

The actor plays a full game from a fresh initial state, running :class:`StochasticMuZeroMCTS` at
every decision node to obtain a visit-count policy and a root value, sampling the played action in
proportion to the visit counts (a temperature of one) and recording one
:class:`~rlgammon.muzero.replay.trajectory.Step` per decision. Chance nodes (the dice rolls) are
resolved on the real game with :func:`apply_sampled_chance`, so the trajectory's observations and
returns are the true game's. The resulting :class:`~rlgammon.muzero.replay.trajectory.Trajectory`
is what the replay buffer (and ultimately the trainer, WU-D4) consumes.
"""
import numpy as np
import torch as th

from rlgammon.game import apply_sampled_chance, board_features
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.replay.trajectory import Step, Trajectory

# Hard cap on the number of decision steps per game, guarding against a non-terminating game.
MAX_MOVES = 1000
# The sampling temperature applied to the root visit counts when choosing the played action.
SAMPLING_TEMPERATURE = 1.0


class SelfPlayActor:
    """Play full games with MCTS-guided self-play and record them as trajectories."""

    def __init__(self, config: MuZeroConfig, game: BackgammonGame,
                 network: StochasticMuZeroNetwork, rng: np.random.Generator) -> None:
        """
        Construct the actor around a configuration, a game factory, a network and a generator.

        :param config: the configuration shared with the search (simulations, unroll/td steps, ...)
        :param game: the game factory producing fresh initial states
        :param network: the learned network driving the search
        :param rng: the random number generator for chance sampling and action selection
        """
        self.config = config
        self.game = game
        self.network = network
        self.rng = rng
        self.mcts = StochasticMuZeroMCTS(config, network, rng)

    def play_game(self) -> Trajectory:
        """
        Play one full self-play game and return its recorded trajectory.

        Starting from a fresh state the opening chance node is resolved, then the actor loops over
        decision nodes (up to :data:`MAX_MOVES`): it extracts the side-to-move's board features, runs
        the search to get visit counts, records a :class:`Step` with the normalized visit-count policy
        and the network's root value, samples an action in proportion to the visit counts, applies it
        and resolves any following chance node. At termination the reward of the last recorded step is
        set to the mover's signed return so the per-step rewards are zero everywhere but the final
        transition, and the per-player returns are stored on the trajectory.

        :return: the :class:`Trajectory` recorded for the played game
        """
        state = self.game.new_initial_state()
        if state.is_chance_node():
            apply_sampled_chance(state, self.rng)

        trajectory = Trajectory()
        for _ in range(MAX_MOVES):
            if state.is_terminal():
                break
            self._play_decision(state, trajectory)

        self._finalize(state, trajectory)
        return trajectory

    def _play_decision(self, state: GameState, trajectory: Trajectory) -> None:
        """
        Run the search at the current decision node, record a step and advance the game.

        :param state: the decision-node game state (mutated in place by the played action and roll)
        :param trajectory: the trajectory the recorded step is appended to
        """
        mover = state.current_player()
        observation = board_features(state, mover)
        legal_actions = state.legal_actions()

        observation_tensor = th.tensor(
            observation, dtype=th.float32, device=self.network.device,
        ).unsqueeze(0)
        visit_counts = self.mcts.run(observation_tensor, legal_actions, add_exploration_noise=True)
        policy = self._normalize_visit_counts(visit_counts)
        value = self._root_value(observation_tensor)
        action = self._select_action(visit_counts)

        trajectory.steps.append(
            Step(observation=observation, action=action, reward=0.0, policy=policy, value=value, to_play=mover),
        )

        state.apply_action(action)
        if state.is_chance_node():
            apply_sampled_chance(state, self.rng)

    def _finalize(self, state: GameState, trajectory: Trajectory) -> None:
        """
        Stamp the terminal returns onto the trajectory and the last step's reward.

        The last recorded step's reward is the terminal return from that step's mover's perspective,
        leaving every other step's reward at zero.

        :param state: the (terminal) game state providing the per-player returns
        :param trajectory: the trajectory to finalize in place
        """
        if not state.is_terminal():
            return
        returns = state.returns()
        trajectory.returns = list(returns)
        if trajectory.steps:
            last_mover = trajectory.steps[-1].to_play
            trajectory.steps[-1].reward = returns[last_mover]

    def _root_value(self, observation_tensor: th.Tensor) -> float:
        """
        Estimate the root value as the network's initial-inference value at the observation.

        This mirrors the value the search itself seeds its normalizer with, giving a single,
        call-order-independent root-value estimate (rather than the noise-perturbed visit-weighted mean).

        :param observation_tensor: the root observation tensor of shape ``[1, observation_size]``
        :return: the scalar root value
        """
        self.network.eval()
        with th.no_grad():
            output = self.network.initial_inference(observation_tensor)
            return float(self.network.value_to_scalar(output.value)[0])

    @staticmethod
    def _normalize_visit_counts(visit_counts: dict[int, int]) -> dict[int, float]:
        """
        Normalize raw visit counts into a probability distribution over actions.

        :param visit_counts: the action -> visit count mapping returned by the search
        :return: a sparse action -> probability mapping summing to one (uniform if every count is zero)
        """
        total = sum(visit_counts.values())
        if total <= 0:
            uniform = 1.0 / len(visit_counts) if visit_counts else 0.0
            return dict.fromkeys(visit_counts, uniform)
        return {action: count / total for action, count in visit_counts.items()}

    def _select_action(self, visit_counts: dict[int, int]) -> int:
        """
        Sample an action in proportion to the visit counts raised to ``1 / temperature``.

        With the default temperature of one this samples directly in proportion to the visit counts;
        if every count is zero the action is drawn uniformly over the candidates.

        :param visit_counts: the action -> visit count mapping returned by the search
        :return: the sampled action id
        """
        actions = list(visit_counts)
        counts = np.array([visit_counts[action] for action in actions], dtype=np.float64)
        if counts.sum() <= 0:
            probabilities = np.full(len(actions), 1.0 / len(actions))
        else:
            scaled = counts ** (1.0 / SAMPLING_TEMPERATURE)
            probabilities = scaled / scaled.sum()
        return int(self.rng.choice(actions, p=probabilities))
