"""Plays a game of backgammon."""

from rlgammon.agents.random_agent import RandomAgent
from rlgammon.environment import BackgammonEnv
from rlgammon.exploration import EpsilonGreedyExploration


def print_env(env: BackgammonEnv, i: int) -> None:
    """Prints the environment."""
    if i % 2 == 1:
        env.render(mode="text")
    else:
        # The flips are done to print the board from the perspective of the white player.
        env.flip()
        env.render(mode="text")
        env.flip()


def play_game() -> None:
    """Plays a game of backgammon."""
    env = BackgammonEnv()
    env.reset()
    exploration = EpsilonGreedyExploration(0.5, 0.05, 0.99, 100)
    agent = RandomAgent()
    done = False
    trunc = False
    i = 0
    while not done and not trunc:
        i += 1
        print_env(env, i)
        dice = env.roll_dice()

        print(f"Color: {'White' if i%2==1 else 'Black'} Roll: {dice}")
        actions = agent.choose_move(env, dice)
        actions = exploration.explore(actions, [move for _, move in env.get_all_complete_moves(dice)])
        for _, action in actions:
            reward, done, trunc, _ = env.step(action)

            print(f"Reward: {reward}")

        if not done and not trunc:
            env.flip()
        exploration.update()
    print_env(env, i)


if __name__ == "__main__":
    play_game()
