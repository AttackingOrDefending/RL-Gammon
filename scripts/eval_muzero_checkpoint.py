"""Evaluate a saved long-training MuZero checkpoint against a uniform-random opponent.

The ``scripts/train_muzero_long.py`` checkpoints are dicts of ``{network, optimizer, state, config}``
(with ``config`` stored as its dataclass fields), so they are not loadable through
``scripts/eval_vs_random.py``'s plain ``--model`` state-dict path. This script rebuilds the network
from the stored architecture and runs a large-sample evaluation against a random opponent, using
either the Gumbel batched search (the path the agent was trained with) or the baseline single-tree
search for an A/B comparison.
"""
import argparse
from dataclasses import replace
from typing import Any

import numpy as np
import torch as th

from rlgammon.muzero.muzero_factory import build_network, resolve_device
from rlgammon.muzero.muzero_types import MuZeroConfig
from scripts.eval_vs_random import play_muzero_vs_random

DEFAULT_GAMES = 400
DEFAULT_SIMS = 50
DEFAULT_SEED = 0


def main() -> None:
    """Load a long-training checkpoint and print its win-rate and average points versus random."""
    parser = argparse.ArgumentParser(description="Evaluate a MuZero long-training checkpoint vs random.")
    parser.add_argument("--checkpoint", required=True, help="path to a train_muzero_long checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="number of evaluation games")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS, help="search simulations per move")
    parser.add_argument("--device", default="cuda", help="device to evaluate on (cuda or cpu)")
    parser.add_argument("--mcts", default="gumbel", choices=("gumbel", "baseline"),
                        help="search used for the MuZero player (gumbel = trained path, baseline = single-tree)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint: dict[str, Any] = th.load(args.checkpoint, map_location=device, weights_only=False)
    config = replace(MuZeroConfig(**checkpoint["config"]), device=device)
    network = build_network(config)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    rng = np.random.default_rng(args.seed)
    result = play_muzero_vs_random(network, config, args.games, rng, args.sims, mcts=args.mcts)
    print(f"checkpoint={args.checkpoint} games={args.games} sims={args.sims} mcts={args.mcts}")
    print(f"win_rate={result['win_rate']:.4f} avg_points={result['avg_points']:.4f}")


if __name__ == "__main__":
    main()
