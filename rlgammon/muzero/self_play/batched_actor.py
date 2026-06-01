"""Batched self-play driving ``K`` real backgammon games in lockstep for GPU-saturating throughput.

The single-game :class:`~rlgammon.muzero.self_play.actor.SelfPlayActor` runs a batch-1 search per
move, so on a GPU every network call is launch-bound. :class:`BatchedSelfPlayActor` keeps ``K`` real
:class:`~rlgammon.game.backgammon_protocol.GameState` games alive at once and, at every joint move,
runs ONE :class:`~rlgammon.muzero.mcts.batched_search.BatchedGumbelMCTS` over all live roots so each
network inference batches up to ``K`` (or ``K x considered-actions``) positions in a single call.

Each game records the same :class:`~rlgammon.muzero.replay.trajectory.Step` stream as the single-game
actor (the Gumbel-improved policy as the policy target, the network root value, the played action),
resolves its own chance node on the real engine, and is handed back as a finished
:class:`~rlgammon.muzero.replay.trajectory.Trajectory` the moment it terminates while the others keep
playing. The played action is the Gumbel-argmax (no Dirichlet noise is needed at the root).
"""
import numpy as np
import torch as th

from rlgammon.game import apply_sampled_chance, board_features
from rlgammon.game.backgammon_protocol import BackgammonGame, GameState
from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS, GumbelRootResult
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork
from rlgammon.muzero.replay.trajectory import Step, Trajectory

# Hard cap on the number of decision steps per game, guarding against a non-terminating game.
MAX_MOVES = 1000


class BatchedSelfPlayActor:
    """Play ``K`` real games concurrently with a single batched Gumbel search per joint move."""

    def __init__(self, config: MuZeroConfig, game: BackgammonGame,
                 network: StochasticMuZeroNetwork, rng: np.random.Generator, *,
                 num_parallel: int, num_considered: int) -> None:
        """
        Construct the batched actor around a configuration, game factory, network and generator.

        :param config: the configuration shared with the batched search (simulations, discount, ...)
        :param game: the game factory producing fresh initial states
        :param network: the learned network driving the batched search
        :param rng: the random number generator for chance sampling and the Gumbel noise
        :param num_parallel: the number ``K`` of games advanced simultaneously
        :param num_considered: the number ``m`` of root actions Gumbel considers per move
        """
        self.config = config
        self.game = game
        self.network = network
        self.rng = rng
        self.num_parallel = num_parallel
        self.mcts = BatchedGumbelMCTS(config, network, rng, num_considered=num_considered)

    def play_games(self) -> list[Trajectory]:
        """
        Play ``num_parallel`` full games in lockstep and return their recorded trajectories.

        A fresh state is created for each of the ``K`` slots (its opening chance node resolved), then
        the actor loops: it gathers the live games' side-to-move observations into one batch, runs the
        batched Gumbel search over all live roots, and for each live game records a step, plays the
        Gumbel-argmax action and resolves the following chance node. Terminated games are finalised and
        collected; the loop ends once every game is terminal (or the move cap is hit).

        :return: the list of ``num_parallel`` finished :class:`Trajectory` objects
        """
        states = [self._fresh_state() for _ in range(self.num_parallel)]
        trajectories = [Trajectory() for _ in range(self.num_parallel)]
        done = [False] * self.num_parallel

        for _ in range(MAX_MOVES):
            live = [index for index in range(self.num_parallel) if not done[index]]
            if not live:
                break
            results = self._search_live(states, live)
            for index, result in zip(live, results, strict=True):
                self._apply_result(states[index], trajectories[index], result)
                if states[index].is_terminal():
                    self._finalize(states[index], trajectories[index])
                    done[index] = True

        for index in range(self.num_parallel):
            if not done[index]:
                self._finalize(states[index], trajectories[index])
        return trajectories

    def _search_live(self, states: list[GameState], live: list[int]) -> list[GumbelRootResult]:
        """
        Run one batched Gumbel search over every live game's root and return the per-game results.

        :param states: the per-slot game states (only the live indices are searched)
        :param live: the indices of the games that are not yet terminal
        :return: the list of :class:`GumbelRootResult`, one per live index, in ``live`` order
        """
        observations = [board_features(states[index], states[index].current_player()) for index in live]
        observation_tensor = th.tensor(observations, dtype=th.float32, device=self.network.device)
        legal_actions = [states[index].legal_actions() for index in live]
        return self.mcts.run_batch(observation_tensor, legal_actions)

    def _apply_result(self, state: GameState, trajectory: Trajectory, result: GumbelRootResult) -> None:
        """
        Record a search result as a step and advance the real game by the chosen action and roll.

        :param state: the live decision-node game state (mutated in place)
        :param trajectory: the trajectory the recorded step is appended to
        :param result: the batched-search result for this game (action, policy target and root value)
        """
        mover = state.current_player()
        observation = board_features(state, mover)
        trajectory.steps.append(Step(
            observation=observation, action=result.action, reward=0.0,
            policy=result.policy, value=result.root_value, to_play=mover,
        ))
        state.apply_action(result.action)
        if state.is_chance_node():
            apply_sampled_chance(state, self.rng)

    def _fresh_state(self) -> GameState:
        """
        Create a fresh initial game state and resolve its opening chance node.

        :return: a fresh decision-node game state ready to be searched
        """
        state = self.game.new_initial_state()
        if state.is_chance_node():
            apply_sampled_chance(state, self.rng)
        return state

    def _finalize(self, state: GameState, trajectory: Trajectory) -> None:
        """
        Stamp the terminal returns onto the trajectory and the last step's reward.

        The last recorded step's reward is the terminal return from that step's mover's perspective,
        leaving every other step's reward at zero (mirroring the single-game actor).

        :param state: the (possibly terminal) game state providing the per-player returns
        :param trajectory: the trajectory to finalize in place
        """
        if not state.is_terminal():
            return
        returns = state.returns()
        trajectory.returns = list(returns)
        if trajectory.steps:
            last_mover = trajectory.steps[-1].to_play
            trajectory.steps[-1].reward = returns[last_mover]
