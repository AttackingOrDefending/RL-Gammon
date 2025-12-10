from random import choice

from discord_interface.games.instances.amazons import AmazonsDiscord
from discord_interface.games.instances.breakthrough import BreakthroughDiscord
from discord_interface.games.instances.clobber import Clobber
from discord_interface.games.instances.santorini import SantoriniDiscord

CORRESPONDENCE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',  #
                  12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S'}  #
#
ANTI_CORRESPONDENCE = {'C': 2, 'H': 7, 'S': 18, 'J': 9, 'E': 4, 'F': 5, 'O': 14, 'B': 1, 'R': 17, 'K': 10, 'I': 8,  #
                       'G': 6, 'L': 11, 'N': 13, 'Q': 16, 'P': 15, 'D': 3, 'A': 0, 'M': 12}  #
def reorientation( coup):



    lettre = coup[0]
    chiffre = coup[1:]

    lettre = ANTI_CORRESPONDENCE[lettre]
    chiffre = int(chiffre)

    return chiffre - 1, lettre

def action_to_string(object):
    # print('>>>>>', object)
    i, j = object

    if isinstance(i, tuple) and isinstance(j, tuple):
        i1, j1 = i
        i2, j2 = j

        return CORRESPONDENCE[j1] + str(i1 + 1) + '-' + CORRESPONDENCE[j2] + str(i2 + 1)
    else:
        return CORRESPONDENCE[j] + str(i + 1)


def string_to_action(string):
    if '-' in string:
        cp1, cp2 = string.split('-')

        return (reorientation(cp1), reorientation(cp2))
    else:
        return reorientation(string)

def move_conversion_from_gtp(move):
    return move


def move_conversion_to_gtp(action):
    return action


if __name__ == '__main__':

    game = None

    line = input()
    while line != 'quit':

        command, *arg = line.split(' ')

        if command == 'game':
            if arg[0].lower() == 'amazons':
                game = AmazonsDiscord()
                print('=')
            elif arg[0].lower() == 'clobber':
                game = Clobber()
                print('=')
            elif arg[0].lower() == 'breakthrough':
                game = BreakthroughDiscord()
                print('=')
            elif arg[0].lower() == 'santorini':
                game = SantoriniDiscord()
                print('=')
            else:
                game = None
                print('=')
        else:

            if game is None:
                print('?','#','Game does not chosen. Available games: Amazons, Clobber, Breakthrough, Santorini.')

            elif command == 'player':
                print('=',game.get_current_player())

            elif command == 'clear_board':
                game.reset()
                print('=')

            elif command == 'undo':
                game.undo()
                print('=')

            elif command == 'time_left':
                print('=')

            elif command == 'genmove':
                move = choice(game.textual_legal_moves())

                game.textual_plays(move)

                print('=',move_conversion_to_gtp(move))


            elif command == 'play':
                color, move = arg

                move = move_conversion_from_gtp(move)

                game.textual_plays(move)
                print('=')

        line = input()