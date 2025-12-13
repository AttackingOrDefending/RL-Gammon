import pyspiel

from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE

from pyspiel_move import slash_to_px_distance
from converter import flipped_to_normal_moves

game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
state = game.new_initial_state()

moves = [6, 1160, 15, 135, 595, 4, 986, 16, 324, 324, 3, 1109, 8, 1272, 0, 1168, 0, 569, 2, 648, 4, 267, 11, 11, 14, 527,
         4, 979, 14, 249, 3, 999, 6, 830, 6, 979, 0, 420, 17, 24, 442, 5, 596, 5, 518, 6, 1090, 18, 96, 81, 18, 648, 254,
         5, 697, 0, 37, 11, 882, 2, 239, 12, 730, 17, 224, 135, 9, 674, 14, 150, 4, 492, 19, 140, 410, 16, 1351, 7, 221, 9, 674,
         10]
moves = [6, 1160, 15]
think = True

for m in moves:
    state.apply_action(m)

if state.is_chance_node():
    outcomes = state.chance_outcomes()
    num_actions = len(outcomes)
    print("Chance node, got " + str(num_actions) + " outcomes")
    action_list, prob_list = zip(*outcomes, strict=False)
    for action, prob in outcomes:
        print("Action: ", state.action_to_string(state.current_player(), action), " Probability: ", prob, "int: ", action)
else:
    # Decision node: sample action for the single current player
    actions = state.legal_actions(state.current_player())
    for action in actions:
        action_string = state.action_to_string(state.current_player(), action)
        discord = slash_to_px_distance(action_string.split(" - ")[1])
        print("Player ", state.current_player(), " action: ", action_string, "int: ", action, "    discord:", discord)

    if think:
        agent = TDAgent("/mnt/c/Users/panti/PycharmProjects/RL-Gammon/rlgammon/agents/saved_agents/td-backgammon-cd3f053a-1c5e-490e-ad7f-feceba70802c-(124).pt")
        action, evaluation = agent.choose_move(actions, state, return_eval=True)
        action_string = state.action_to_string(state.current_player(), action)
        discord = slash_to_px_distance(action_string.split(" - ")[1])
        if state.current_player() == WHITE:
            discord = discord.split('-')
            ts = []
            for t in discord:
                if t.startswith('P'):
                    ts.append('P'+str(25 - int(t[1:])))
                else:
                    ts.append(t)
            discord = '-'.join(ts)

        print(f"-----------------------------------------------------------------")
        print("Player ", state.current_player(), " chosen action: ", action_string, "int: ", action, "    discord:", discord)
        print("Evaluation: ", evaluation, "Our side eval: ", evaluation if state.current_player() == WHITE else -evaluation)
