from random import choice

from discord_interface.games.instances.amazons import AmazonsDiscord
from discord_interface.games.instances.arimaa import ArimaaDiscord
from discord_interface.games.instances.ataxx import AtaxxDiscord
from discord_interface.games.instances.backgammon import BackgammonDiscord
from discord_interface.games.instances.brazilian_draughts import BrazilianDraughtsDiscord
from discord_interface.games.instances.breakthrough import BreakthroughDiscord
from discord_interface.games.instances.canadian_draughts import CanadianDraughtsDiscord
from discord_interface.games.instances.chinese_chess import ChineseChessDiscord
from discord_interface.games.instances.clobber import Clobber
from discord_interface.games.instances.connect6 import Connect6Discord
from discord_interface.games.instances.havannah10 import Havannah10Discord
from discord_interface.games.instances.havannah8 import Havannah8Discord
from discord_interface.games.instances.hex11 import Hex11Discord
from discord_interface.games.instances.hex13 import Hex13Discord
from discord_interface.games.instances.hex19 import Hex19Discord
from discord_interface.games.instances.international_draughts import InternationalDraughtsDiscord
from discord_interface.games.instances.lines_of_action import LinesOfActionDiscord
from discord_interface.games.instances.othello10 import Othello10Discord
from discord_interface.games.instances.othello16 import Othello16Discord
from discord_interface.games.instances.othello8 import Othello8Discord
from discord_interface.games.instances.outer_open_gomoku import OuterOpenGomokuDiscord
from discord_interface.games.instances.quoridor import QuoridorDiscord
from discord_interface.games.instances.santorini import SantoriniDiscord
from discord_interface.games.instances.shobu import ShobuDiscord
from discord_interface.games.instances.surakarta import SurakartaDiscord

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
                print('=','#','Amazons start!')
            elif arg[0].lower() == 'clobber':
                game = Clobber()
                print('=','#','Clobber start!')
            elif arg[0].lower() == 'breakthrough':
                game = BreakthroughDiscord()
                print('=','#','Breakthrough start!')
            elif arg[0].lower() == 'santorini':
                game = SantoriniDiscord()
                print('=','#','Santorini start!')
            elif arg[0].lower() == 'surakarta':
                game = SurakartaDiscord()
                print('=','#','Surakarta start!')
            elif arg[0].lower() == 'lines-of-action' or arg[0].lower() == 'lines_of_action':
                game = LinesOfActionDiscord()
                print('=','#','Lines of Action start!')
            elif arg[0].lower() == 'brazilian-draughts' or arg[0].lower() == 'brazilian_draughts':
                game = BrazilianDraughtsDiscord()
                print('=','#','Brazilian Draughts start!')
            elif arg[0].lower() == 'canadian-draughts' or arg[0].lower() == 'canadian_draughts':
                game = CanadianDraughtsDiscord()
                print('=','#','Canadian Draughts start!')
            elif arg[0].lower() == 'international-draughts' or arg[0].lower() == 'international_draughts':
                game = InternationalDraughtsDiscord()
                print('=','#','International Draughts start!')
            elif arg[0].lower() == 'outer-open-gomoku' or arg[0].lower() == 'outer_open_gomoku':
                game = OuterOpenGomokuDiscord()
                print('=','#','Outer Open Gomoku start!')
            elif arg[0].lower() == 'connect6':
                game = Connect6Discord()
                print('=','#','Connect6 start!')
            elif arg[0].lower() == 'hex11':
                game = Hex11Discord()
                print('=','#','Hex 11 start!')
            elif arg[0].lower() == 'hex13':
                game = Hex13Discord()
                print('=','#','Hex 13 start!')
            elif arg[0].lower() == 'hex19':
                game = Hex19Discord()
                print('=','#','Hex 19 start!')
            elif arg[0].lower() == 'arimaa':
                game = ArimaaDiscord()
                print('=','#','Arimaa start!')
            elif arg[0].lower() == 'shobu':
                game = ShobuDiscord()
                print('=','#','Shobu start!')
            elif arg[0].lower() == 'havannah8':
                game = Havannah8Discord()
                print('=','#','Havannah 8 start!')
            elif arg[0].lower() == 'havannah10':
                game = Havannah10Discord()
                print('=','#','Havannah 10 start!')
            elif arg[0].lower() == 'othello8':
                game = Othello8Discord()
                print('=','#','Othello 8 start!')
            elif arg[0].lower() == 'othello10':
                game = Othello10Discord()
                print('=', '#', 'Othello 10 start!')
            elif arg[0].lower() == 'othello16':
                game = Othello16Discord()
                print('=','#','Othello 16 start!')
            elif arg[0].lower() == 'ataxx':
                game = AtaxxDiscord()
                print('=','#','Ataxx start!')
            elif arg[0].lower() == 'chinese-chess' or arg[0].lower() == 'chinese_chess':
                game = ChineseChessDiscord()
                print('=','#','Chinese Chess start!')
            elif arg[0].lower() == 'quoridor':
                game = QuoridorDiscord()
                print('=','#','Quoridor start!')
            elif arg[0].lower() == 'backgammon':
                game = BackgammonDiscord()
                print('=','#','Backgammon start!')
            else:
                game = None
                print('=','#','Game does not implemented.')
        else:

            if game is None:
                print('?','#','Game does not chosen.')

            elif command == 'player':
                print('=',game.get_current_player(),'#','Current player:',int(game.get_current_player()),'.')

            elif command == 'clear_board':
                game.reset()
                print('=','#','Board is cleared.')

            elif command == 'undo':
                game.undo()
                print('=','#','Last move canceled.')

            elif command == 'time_left':
                print('=','#','Time left set.')

            elif command == 'genmove':
                move = choice(game.textual_legal_moves())

                game.textual_plays(move)

                print('=',move_conversion_to_gtp(move),'#','Generated move:',move,'(',move_conversion_to_gtp(move),')','.')


            elif command == 'play':
                color, move = arg

                move = move_conversion_from_gtp(move)

                game.textual_plays(move)
                print('=','#','Move ',move,'of player',color,'have been played.')

        line = input()