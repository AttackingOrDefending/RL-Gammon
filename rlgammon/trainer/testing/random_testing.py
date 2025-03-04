"""TODO."""
from rlgammon.agents.base_agent import BaseAgent
from rlgammon.agents.random_agent import RandomAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.trainer.testing.base_testing import BaseTesting


class RandomTesting(BaseTesting):
    """TODO."""

    def __init__(self, games_per_test: int) -> None:
        """TODO."""
        self.games_per_test = games_per_test
        self.testing_agent = RandomAgent()

    def test(self, agent: BaseAgent) -> dict[str, float]:
        """TODO."""
        wins = 0
        draws = 0
        losses = 0
        env = BackgammonEnv()
        agent_player = 1
        for _test_game in range(self.games_per_test):
            env.reset()
            done = False
            trunc = False
            reward = 0.0
            while not done and not trunc:
                dice = env.roll_dice()
                if env.current_player == agent_player:
                    actions = agent.choose_move(env, dice)
                else:
                    actions = self.testing_agent.choose_move(env, dice)

                for _, action in actions:
                    reward, done, trunc, _ = env.step(action)

                if not done and not trunc:
                    env.flip()

            if reward == 0:
                draws += 1
            elif env.has_lost(agent_player):
                losses += 1
            else:
                wins += 1

        return {"win_rate": wins / self.games_per_test,
                "draws": draws / self.games_per_test,
                "losses": losses / self.games_per_test}
