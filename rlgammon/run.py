"""Run the trainer."""
import argparse
import os

from rlgammon.agents.td_agent import TDAgent
from rlgammon.models.model_types import ValueHead
from rlgammon.planning.planning_types import PossibleSearch, SearchConfig
from rlgammon.trainer.step_trainer import StepTrainer

# Depth (in decision plies) the agent "thinks" at RANDOM-testing time, deeper than the 1-ply training.
EVAL_SEARCH_DEPTH = 2

# Number of episodes used for a quick smoke run, kept tiny so it finishes in well under a minute.
SMOKE_EPISODES = 2
# Skip the (gnubg-backed) testing pass during a smoke run by testing less often than the episode count.
SMOKE_EPISODES_PER_TEST = SMOKE_EPISODES + 1


def main() -> None:
    """Build a TD-Gammon agent and train it, optionally as a tiny smoke run."""
    parser = argparse.ArgumentParser(description="Train a TD-Gammon value agent.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny training loop for smoke testing (also enabled by RLGAMMON_SMOKE=1).")
    args = parser.parse_args()
    smoke = args.smoke or os.environ.get("RLGAMMON_SMOKE", "") not in ("", "0")

    agent = TDAgent(value_head=ValueHead.EQUITY_SIGMOID, hidden=128, lr=0.1, lamda=0.7)
    trainer = StepTrainer()
    trainer.load_parameters("parameters.json")
    # Training stays plain 1-ply TD; only RANDOM testing "thinks deeper" via this separate config.
    trainer.eval_search_config = SearchConfig(PossibleSearch.STAR_MINIMAX, max_depth=EVAL_SEARCH_DEPTH)
    if smoke:
        trainer.parameters["episodes"] = SMOKE_EPISODES
        trainer.parameters["episodes_per_test"] = SMOKE_EPISODES_PER_TEST
        trainer.parameters["save_progress"] = False
    trainer.train(agent)


if __name__ == "__main__":
    main()
