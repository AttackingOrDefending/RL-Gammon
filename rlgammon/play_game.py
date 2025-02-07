"""Plays a game of backgammon."""
import random

from rlgammon.environment import BackgammonEnv


def play_game() -> None:
    """Plays a game of backgammon."""
    env = BackgammonEnv()
    env.reset()
    done = False
    trunc = False
    i = 0
    while not done and not trunc:
        i += 1
        if i % 2 == 1:
            env.render(mode="text")
        else:
            env.flip()
            env.render(mode="text")
            env.flip()
        dice = env.roll_dice()
        while dice:
            actions_per_roll = env.get_legal_moves(dice)
            actions = []
            for roll in dice:
                actions += [(roll, move) for move in actions_per_roll[roll]]
            if not actions:
                break
            roll, action = random.choice(actions)
            dice.remove(roll)
            _, reward, done, trunc, _ = env.step(action)

            env.render(mode="human")

            print(f"Reward: {reward}")

        if not done and not trunc:
            env.flip()
    env.render(mode="text")


if __name__ == "__main__":
    play_game()
