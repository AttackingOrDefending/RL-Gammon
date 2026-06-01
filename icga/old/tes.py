from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel
import random
import numpy as np

games_list = pyspiel.registered_games()
print("Registered games:")
print(games_list)
game = pyspiel.load_game("backgammon")
state = game.new_initial_state()
print("Initial state:")
print(state)

while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    print("Current player: ", state.current_player())
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
    else:
      # Decision node: sample action for the single current player
      action = random.choice(state.legal_actions(state.current_player()))
      action_string = state.action_to_string(state.current_player(), action)
      print("Player ", state.current_player(), ", randomly sampled action: ",
            action_string)
      state.apply_action(action)
    print(str(state))

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))