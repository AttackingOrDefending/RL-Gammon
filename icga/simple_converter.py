import pyspiel
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from pyspiel_move import slash_to_px_distance
from converter import flipped_to_normal_moves
game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
state = game.new_initial_state()
think = True



moves = [14, 156, 2, 621, 4, 427, 15, 135, 108, 14, 156, 7, 1218, 6, 401, 9, 336, 19, 302, 427, 6, 139, 4, 979, 19, 383, 410,
         7, 973, 15, 272, 324, 1, 1109, 0, 272, 8, 1109, 3, 85, 4, 1031, 1, 137, 5, 588, 3, 193, 13, 1163, 7, 167,
         13, 1162, 17, 219, 243, 5, 593, 15, 162, 162, 15, 616, 593, 0, 191, 3, 1189, 0, 158, 9, 1214, 8, 809, 2, 1162,
         5, 28, 1, 1241, 11, 135, 13, 538, 13, 83, 9, 1189, 1, 2, 10, 594, 8, 109, 20, 621]
move_to_convert = '''

P22-3-P20-1

'''


move_to_convert = move_to_convert.strip()
for m in moves:
    state.apply_action(m)
print(state)
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
    cors = []
    for action in actions:
        action_string = state.action_to_string(state.current_player(), action)
        discord = ""
        try:
            discord = slash_to_px_distance(action_string.split(" - ")[1])
            if move_to_convert:
                d1 = discord
                d2 = discord.split('-')
                d2 = d2[2:] + d2[:2]
                d2 = '-'.join(d2)
                if state.current_player() == WHITE:
                    d1 = d1.split('-')
                    ts = []
                    for t in d1:
                        if t.startswith('P'):
                            ts.append('P' + str(25 - int(t[1:])))
                        else:
                            ts.append(t)
                    d1 = '-'.join(ts)
                    d2 = d2.split('-')
                    ts = []
                    for t in d2:
                        if t.startswith('P'):
                            ts.append('P' + str(25 - int(t[1:])))
                        else:
                            ts.append(t)
                    d2 = '-'.join(ts)
                if d1 == move_to_convert:
                    cors.append((action, action_string, 1))
                if d2 == move_to_convert:
                    cors.append((action, action_string, 2))
        except Exception as e:
            print("Error converting move: ", action_string, " error: ", e)
        print("Player ", state.current_player(), " action: ", action_string, "int: ", action, "    discord:", discord)
    print('---------------------------------------------------------------------')
    for cor in cors:
        print("Converted move: ", cor[1], " order: ", cor[2], " int: ", cor[0])

    if think:
        agent = TDAgent("/mnt/c/Users/panti/PycharmProjects/RL-Gammon/rlgammon/agents/saved_agents/td-backgammon-cd3f053a-1c5e-490e-ad7f-feceba70802c-(1999).pt")
        action, evaluation = agent.choose_move(actions, state, return_eval=True)
        action_string = state.action_to_string(state.current_player(), action)
        discord = ""
        try:
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
        except Exception as e:
            print("Error converting move: ", action_string, " error: ", e)

        print(f"-----------------------------------------------------------------")
        print("Evaluation: ", evaluation, "Our side eval: ", evaluation if state.current_player() == WHITE else -evaluation)
        print("Player ", state.current_player(), " chosen action: ", action_string, "int: ", action, "    discord:", discord)
