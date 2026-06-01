#type: ignore
"""Drive the real GNUTesting entry point with a fresh TDAgent against the running bridge."""
import traceback

from rlgammon.agents.td_agent import TDAgent
from rlgammon.trainer.testing.gnu_testing import GNUTesting


def main():
    agent = TDAgent()  # fresh untrained net, color=WHITE
    try:
        results = GNUTesting(episodes_in_test=1).test(agent)
        print("GNUTEST: RESULTS =", results)
    except Exception as exc:  # noqa: BLE001
        print("GNUTEST: EXCEPTION", repr(exc))
        traceback.print_exc()


if __name__ == "__main__":
    main()
