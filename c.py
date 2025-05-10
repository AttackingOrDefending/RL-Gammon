import pyspiel
from rlgammon.environment.backgammon_env import BackgammonEnv

game = pyspiel.load_game("backgammon")
env = BackgammonEnv()
state = game.new_initial_state()
c, r, obs = env.reset()
while c != 0:
    c, r, obs = env.reset()

print("Start")
print(obs)
print(state.observation_tensor(0)[:198])

# First, roll the dice
for action in state.legal_actions():
    if "roll" in state.action_to_string(state.current_player(), action):
        print(f"Action: {action} Rolling: {state.action_to_string(state.current_player(), action)}")
        #state.apply_action(action)
        #break
state.apply_action(0)
print("Start After Rolling")
print(obs)
print(state.observation_tensor(0)[:198])

# Now print all legal checker moves
for action in state.legal_actions():
    desc = state.action_to_string(state.current_player(), action)
    print(f"{action}: {desc}")
state.apply_action(1188)
print(env.get_valid_actions((1, 2)))
env.step(((5, 3), (3, 2)))
print("After Moving")
print(obs)
print(state.observation_tensor(0)[:198])

for action in state.legal_actions():
    if "roll" in state.action_to_string(state.current_player(), action):
        print(f"Action: {action} Rolling: {state.action_to_string(state.current_player(), action)}")

state.apply_action(0)
print("After Moving and Rolling")
print(obs)
print(state.observation_tensor(0)[:198])