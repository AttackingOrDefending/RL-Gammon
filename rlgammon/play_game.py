from rlgammon.environment import BackgammonEnv
from rlgammon.agents.random_agent import RandomAgent


def play_game():
    env = BackgammonEnv()
    env.reset()
    agent = RandomAgent()
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
        actions = agent.choose_move(env, dice)
        print(actions)
        for roll, action in actions:
            _, reward, done, trunc, _ = env.step(action)

            env.render(mode="human")

            print(f"Reward: {reward}")
        if not done and not trunc:
            env.flip()
    env.render(mode="text")
    print("Game over!")


if __name__ == "__main__":
    play_game()
