"""Side-by-side A/B demo of the BASELINE single-tree search vs the OPT-IN batched Gumbel search.

This is a tiny, CPU-only, pyspiel-free sanity demo proving that the two Stochastic-MuZero search paths
work side by side on the SAME fresh (untrained) network and produce valid, legal outputs:

* the proven BASELINE :class:`~rlgammon.muzero.mcts.search.StochasticMuZeroMCTS` (single tree, pUCT +
  Dirichlet noise) returning per-action visit counts, built by
  :func:`~rlgammon.muzero.muzero_factory.build_mcts`; and
* the opt-in FEATURE :class:`~rlgammon.muzero.mcts.batched_search.BatchedGumbelMCTS` (many trees in
  lockstep, Gumbel-top-k + sequential halving) returning a Gumbel-argmax action and an improved policy,
  built by :func:`~rlgammon.muzero.muzero_factory.build_batched_gumbel_mcts`.

It plays a couple of moves on the pure-Python :class:`~rlgammon.game.mock_game.MockGame`, runs BOTH
searches at each decision node, and checks each picks a legal action with a valid distribution. It is
NOT a benchmark and trains nothing; it only demonstrates both paths are first-class and selectable.
The baseline remains the DEFAULT everywhere -- the features are opt-in pending A/B win-rate results.
Run with ``python3 -m scripts.muzero_ab_demo``.
"""
import argparse

import numpy as np
import torch as th

from rlgammon.game import apply_sampled_chance, board_features
from rlgammon.game.backgammon_protocol import GameState
from rlgammon.game.mock_game import MockGame
from rlgammon.muzero.mcts.batched_search import BatchedGumbelMCTS
from rlgammon.muzero.mcts.search import StochasticMuZeroMCTS
from rlgammon.muzero.muzero_factory import build_batched_gumbel_mcts, build_mcts, build_network
from rlgammon.muzero.muzero_types import MuZeroConfig
from rlgammon.muzero.networks.stochastic_muzero_network import StochasticMuZeroNetwork

# Tiny, CPU-only network/search dimensions keeping the demo fast and memory-light (a fresh net).
DEMO_OBSERVATION_SIZE = 198
DEMO_NUM_ACTIONS = 8
DEMO_STATE_CHANNELS = 16
DEMO_HIDDEN_SIZES = (16,)
DEMO_CODEBOOK_SIZE = 2
DEMO_VALUE_SUPPORT_SIZE = 3
DEMO_REWARD_SUPPORT_SIZE = 3
DEMO_NUM_SIMULATIONS = 16
DEMO_NUM_CONSIDERED = 4
# Default number of decision nodes to demonstrate the two searches on.
DEFAULT_MOVES = 2
# Default demo seed.
DEFAULT_SEED = 0
# Tolerance for the policy-sums-to-one check.
SUM_TOLERANCE = 1e-5


def _build_demo_config(seed: int) -> MuZeroConfig:
    """
    Build the tiny CPU MuZero configuration used by the A/B demo.

    :param seed: the configuration seed shared by both searches
    :return: a small CPU :class:`MuZeroConfig`
    """
    return MuZeroConfig(
        observation_size=DEMO_OBSERVATION_SIZE,
        num_actions=DEMO_NUM_ACTIONS,
        state_channels=DEMO_STATE_CHANNELS,
        hidden_sizes=DEMO_HIDDEN_SIZES,
        codebook_size=DEMO_CODEBOOK_SIZE,
        value_support_size=DEMO_VALUE_SUPPORT_SIZE,
        reward_support_size=DEMO_REWARD_SUPPORT_SIZE,
        num_simulations=DEMO_NUM_SIMULATIONS,
        seed=seed,
        device="cpu",
    )


def _advance_to_decision(state: GameState, rng: np.random.Generator) -> None:
    """
    Resolve any pending chance node so ``state`` sits at a decision node (or terminal), in place.

    :param state: the game state to advance (mutated in place)
    :param rng: the random number generator used to sample the chance outcome
    """
    while not state.is_terminal() and state.is_chance_node():
        apply_sampled_chance(state, rng)


def run_demo(moves: int, seed: int) -> bool:
    """
    Run the baseline-vs-feature A/B demo for ``moves`` decision nodes and report whether both are valid.

    A single fresh (untrained) network is shared by both searches. At each decision node the BASELINE
    single-tree search and the OPT-IN batched Gumbel search are both run on the same root observation;
    the demo prints each search's chosen action and asserts it is legal with a valid distribution, then
    advances the real mock game by the baseline's choice so the walk is well defined.

    :param moves: the number of decision nodes to demonstrate the two searches on
    :param seed: the seed for the demo (config, network init and both searches)
    :return: ``True`` if every move produced a legal action and a valid distribution from both searches
    """
    config = _build_demo_config(seed)
    network = build_network(config)
    network.eval()

    baseline = build_mcts(config, network, np.random.default_rng(seed))
    feature = build_batched_gumbel_mcts(
        config, network, np.random.default_rng(seed), num_considered=DEMO_NUM_CONSIDERED,
    )

    print("[muzero-ab] baseline=StochasticMuZeroMCTS (single-tree pUCT + Dirichlet, the DEFAULT)")
    print("[muzero-ab] feature =BatchedGumbelMCTS (lockstep trees + Gumbel-top-k, OPT-IN)")
    print(f"[muzero-ab] fresh untrained net, sims={config.num_simulations}, considered={DEMO_NUM_CONSIDERED}\n")

    rng = np.random.default_rng(seed)
    state = MockGame().new_initial_state()
    _advance_to_decision(state, rng)

    all_valid = True
    for move_index in range(moves):
        if state.is_terminal():
            print(f"[muzero-ab] reached a terminal state after {move_index} move(s); stopping early")
            break
        all_valid &= _demo_one_move(move_index, state, baseline, feature, network)
        _advance_to_decision(state, rng)

    print(
        "\n[muzero-ab] both search paths produced valid action/visit outputs side by side."
        if all_valid else "\n[muzero-ab] a search produced an invalid output (see above)",
    )
    print(
        "[muzero-ab] NOTE: the baseline (single-tree) search is the DEFAULT everywhere; the batched "
        "Gumbel\n            search and the other performance features are OPT-IN, pending A/B "
        "win-rate results.",
    )
    return all_valid


def _demo_one_move(move_index: int, state: GameState, baseline: StochasticMuZeroMCTS,
                   feature: BatchedGumbelMCTS, network: StochasticMuZeroNetwork) -> bool:
    """
    Run both searches at one decision node, print their picks, and validate them.

    :param move_index: the 0-based move number (for display)
    :param state: the decision-node game state to search from
    :param baseline: the baseline single-tree :class:`StochasticMuZeroMCTS`
    :param feature: the opt-in batched :class:`BatchedGumbelMCTS`
    :param network: the shared network providing the search device
    :return: ``True`` if both searches returned a legal action with a valid distribution
    """
    mover = state.current_player()
    legal_actions = state.legal_actions()
    observation = board_features(state, mover)
    observation_tensor = th.tensor(observation, dtype=th.float32, device=network.device).unsqueeze(0)

    visit_counts = baseline.run(observation_tensor, legal_actions, add_exploration_noise=True)
    baseline_action = max(visit_counts, key=lambda action: visit_counts[action])
    total_visits = sum(visit_counts.values())

    result = feature.run_batch(observation_tensor, [legal_actions])[0]
    feature_policy_sum = sum(result.policy.values())

    print(f"[move {move_index}] legal={legal_actions}")
    print(
        f"  baseline: action={baseline_action} visits={visit_counts} "
        f"(total={total_visits})",
    )
    print(
        f"  feature : action={result.action} root_value={result.root_value:+.4f} "
        f"policy_sum={feature_policy_sum:.4f} policy={{"
        + ", ".join(f"{action}:{prob:.3f}" for action, prob in sorted(result.policy.items())) + "}",
    )

    baseline_ok = baseline_action in legal_actions and set(visit_counts) == set(legal_actions)
    feature_ok = (
        result.action in legal_actions
        and set(result.policy) == set(legal_actions)
        and abs(feature_policy_sum - 1.0) < SUM_TOLERANCE
    )
    state.apply_action(baseline_action)
    return baseline_ok and feature_ok


def main() -> None:
    """Parse the command-line arguments and run the baseline-vs-feature A/B demo."""
    parser = argparse.ArgumentParser(
        description="Show the baseline single-tree search and the opt-in batched Gumbel search side by side.",
    )
    parser.add_argument("--moves", type=int, default=DEFAULT_MOVES, help="decision nodes to demonstrate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="demo random seed")
    args = parser.parse_args()
    run_demo(args.moves, args.seed)


if __name__ == "__main__":
    main()
