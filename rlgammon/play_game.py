from rlgammon.env import BackgammonEnv
import random


def play_game():
    env = BackgammonEnv()
    env.reset()
    # print(env.get_legal_moves([5, 6]))
    # None * 2
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
        print(f"Color: {'White' if i%2==1 else 'Black'} Roll: {dice}")
        while dice:
            actions_per_roll = env.get_legal_moves(dice)
            print(actions_per_roll)
            actions = []
            for roll in dice:
                actions += list(map(lambda move: (roll, move), actions_per_roll[roll]))
            if not actions:
                break
            roll, action = random.choice(actions)
            dice.remove(roll)
            _, reward, done, trunc, _ = env.step(action)
            print(f"Reward: {reward}")
        if not done and not trunc:
            env.flip()
    env.render(mode="text")
    print("Game over!")


if __name__ == "__main__":
    play_game()
