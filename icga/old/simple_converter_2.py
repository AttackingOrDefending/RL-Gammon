import pyspiel

from icga.old.dice_converter_2 import dice_conv
from icga.old.pyspiel_move import slash_to_px_distance
from rlgammon.agents.td_agent import TDAgent
from rlgammon.rlgammon_types import WHITE
from rlgammon.search.search2 import Search

think = True

#moves = [6, 1160, 15, 135, 595, 4, 986, 16, 324, 324, 3, 1109, 8, 1272, 0, 1168, 0, 569, 2, 648, 4, 267, 11, 11, 14, 527,
#         4, 979, 14, 249, 3, 999, 6, 830, 6, 979, 0, 420, 17, 24, 442, 5, 596, 5, 518, 6, 1090, 18, 96, 81, 18, 648, 254,
#         5, 697, 0, 37, 11, 882, 2, 239, 12, 730, 17, 224, 135, 9, 674, 14, 150, 4, 492, 19, 140, 410, 16, 1351, 7, 221, 9, 674,
#         10]
moves = [6, 1160, 15, 135, 595, 4, 986, 16, 324, 324, 3, 1109, 8 ]
moves = []
our_turn = False
first_move = True
move_to_convert = "P25-2-P23-6"

game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")


def re_run(env, moves):
    state = env.new_initial_state()
    for m in moves:
        state.apply_action(m)
    actions = state.legal_actions(state.current_player())
    return state, actions


while True:
    state, actions = re_run(game, moves)
    if not our_turn:

        dice = dice_conv(input("Opponent Dice: "), False)
        moves.append(int(dice))
        state, actions = re_run(game, moves)
        move_to_convert = input("Move to convert: ")

        # Decision node: sample action for the single current player
        cors = []
        for action in actions:
            action_string = state.action_to_string(state.current_player(), action)
            discord = slash_to_px_distance(action_string.split(" - ")[1])
            if move_to_convert:
                d1 = discord
                d2 = discord.split("-")
                d2 = d2[2:] + d2[:2]
                d2 = "-".join(d2)
                if state.current_player() == WHITE:
                    d1 = d1.split("-")
                    ts = []
                    for t in d1:
                        if t.startswith("P"):
                            ts.append("P" + str(25 - int(t[1:])))
                        else:
                            ts.append(t)
                    d1 = "-".join(ts)
                    d2 = d2.split("-")
                    ts = []
                    for t in d2:
                        if t.startswith("P"):
                            ts.append("P" + str(25 - int(t[1:])))
                        else:
                            ts.append(t)
                    d2 = "-".join(ts)
                if d1 == move_to_convert:
                    cors.append((action, action_string, 1))
                if d2 == move_to_convert:
                    cors.append((action, action_string, 2))
                # print("Player ", state.current_player(), " action: ", action_string, "int: ", action, "    discord:", discord)
            # print("---------------------------------------------------------------------")
            # for cor in cors:
            #     print("Converted move: ", cor[1], " int: ", cor[0], " order: ", cor[2])

        moves.append(cors[-1][0])
        state, actions = re_run(game, moves)
        print("Current Moves: ", moves)

        our_dice = dice_conv(input("Our dice: "), False)
        moves.append(int(our_dice))
        state, actions = re_run(game, moves)

        our_turn = not our_turn

        if our_turn:
            agent = TDAgent("/Users/frexmax/Desktop/RL-Gammon/good_models/cd3f053a-1c5e-490e-ad7f-feceba70802c-32200-episodes.pt")

            search = Search(agent)
            evaluation, action = search.expectimax_root(state, 1)
            # action, evaluation = agent.choose_move(actions, state, return_eval=True)

            action_string = state.action_to_string(state.current_player(), action)
            discord = slash_to_px_distance(action_string.split(" - ")[1])
            if state.current_player() == WHITE:
                discord = discord.split("-")
                ts = []
                for t in discord:
                    if t.startswith("P"):
                        ts.append("P"+str(25 - int(t[1:])))
                    else:
                        ts.append(t)
                discord = "-".join(ts)

            print("-----------------------------------------------------------------")
            print("Player ", state.current_player(), " chosen action: ", action_string, "int: ", action, "    discord:", discord)
            print("Evaluation: ", evaluation, "Our side eval: ", evaluation if state.current_player() == WHITE else -evaluation)
            print("-----------------------------------------------------------------")

            moves.append(action)
            print("Current Moves: ", moves)
            state, actions = re_run(game, moves)

            our_turn = not our_turn
