import pyspiel
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from pyspiel_move import slash_to_px_distance
from converter import flipped_to_normal_moves
game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
state = game.new_initial_state()

moves = [11, 962, 6, 142, 6, 1142, 6, 142, 13, 694, 10, 137, 2, 536, 17, 189, 246, 5, 986, 1, 606, 6, 986, 18,
         517, 168, 1, 986, 2, 327, 8, 700, 9, 1192, 6, 1054, 14, 301,
         8, 726, 17, 165, 219, 4, 876, 13, 164, 3, 1110, 0, 83, 9, 1189, 0, 86, 4, 1035, 15, 81, 108,
         1, 1165, 11, 812, 17, 42, 512, 5, 54, 0, 799, 16, 54, 54, 5, 850, 7, 57, 0, 581, 18, 81]

for i, m in enumerate(moves):
    actions = state.legal_actions(state.current_player())
    for action in actions:
        if action == m:
            print(f"{i}: {state.action_to_string(state.current_player(), action)}")
            state.apply_action(m)
            break
