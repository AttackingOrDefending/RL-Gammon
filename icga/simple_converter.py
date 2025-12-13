import pyspiel
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from pyspiel_move import slash_to_px_distance
from converter import flipped_to_normal_moves
game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
state = game.new_initial_state()
think = True



moves = [11, 962, 6, 142, 6, 1142, 6, 142, 13, 694, 10, 137, 2, 536, 17, 189, 246, 5, 986, 1, 606, 6, 986, 18,
         517, 168, 1, 986, 2, 327, 8, 700, 9, 1192, 6, 1054, 14, 301,
         8, 726, 17, 165, 219, 4, 876, 13, 164, 3, 1110, 0, 83, 9, 1189, 0, 86, 4, 1035, 15, 81, 108,
         1, 1165, 11, 812, 17, 42, 512, 5, 54, 0, 799, 16, 54, 54, 5, 850, 7, 57, 0, 581, 18, 81]
move_to_convert = '''

P23-1-P10-2

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
        agent = TDAgent("/mnt/c/Users/panti/PycharmProjects/RL-Gammon/rlgammon/agents/saved_agents/td-backgammon-cd3f053a-1c5e-490e-ad7f-feceba70802c-(256).pt")
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
