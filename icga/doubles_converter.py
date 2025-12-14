import pyspiel
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from pyspiel_move import slash_to_px_distance
from converter import flipped_to_normal_moves
game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
state = game.new_initial_state()
think = True



moves = [17, 220, 12, 297, 4, 993, 5, 468, 5, 1280, 12, 778, 1, 862, 17, ]
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
        agent = TDAgent("/mnt/c/Users/panti/PycharmProjects/RL-Gammon/rlgammon/agents/saved_agents/td-backgammon-cd3f053a-1c5e-490e-ad7f-feceba70802c-(1999).pt")
        moves = []
        action, evaluation = agent.choose_move(actions, state, return_eval=True)
        action_string = state.action_to_string(state.current_player(), action)
        moves.append(action)
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

        state2 = state.clone()
        state2.apply_action(action)
        if not state2.is_terminal():
            actions = state2.legal_actions(state2.current_player())
            action2, evaluation2 = agent.choose_move(actions, state2, return_eval=True)
            action_string2 = state2.action_to_string(state2.current_player(), action2)
            discord2 = slash_to_px_distance(action_string2.split(" - ")[1])
            if state2.current_player() == WHITE:
                discord2 = discord2.split('-')
                ts = []
                for t in discord2:
                    if t.startswith('P'):
                        ts.append('P' + str(25 - int(t[1:])))
                    else:
                        ts.append(t)
                discord2 = '-'.join(ts)
            action_string += " " + action_string2
            d = discord.split('-')[1]
            pos = []
            for i, p in enumerate(discord.split('-')):
                if i % 2 == 0:
                    pos.append(p)
            for i, p in enumerate(discord2.split('-')):
                if i % 2 == 0:
                    pos.append(p)
            discord = d + '-' + '-'.join(pos)
            evaluation = evaluation2
            moves.append(action2)

        print(f"-----------------------------------------------------------------")
        print("Player ", state.current_player(), " chosen action: ", action_string, "int: ", moves, "    discord:", discord)
        print("Evaluation: ", evaluation, "Our side eval: ", evaluation if state.current_player() == WHITE else -evaluation)
