from discord_interface.games.translator.quentin_games.Arimaa_vrai import Arimaa_vrai
from discord_interface.games.translator.quentin_games.backgammon import Backgammon
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord
from discord_interface.games.translator.stochastic_quentin_games_interface import StochasticInterfaceJeuDiscord


class BackgammonDiscord(StochasticInterfaceJeuDiscord):

    def __init__(self):
        super().__init__('backgammon', Backgammon, params=None, rules='standard', move_keywords=['pass'])


    def string_to_action(self, string):
        if string == 'pass':
            return (None, None)
        if string.count('-') == 1 and string[0] == 'P' or 'P' not in string:
            if 'P' in string: # simple tronqué
                i, j = string.replace('P','').split('-')
                return ((int(i), int(j)), None)
            else: # chance
                i,j = string.split('-')
                return (int(i), int(j))
        else:
            if string.count('P') == string.count('-') and string[0] != 'P': # double
                string = string.replace('P','')

                coord = string.split('-')

                return (tuple(int(i) for i in coord[1:]), int(coord[0]))

            else: # simple
                string = string.replace('P','')
                coord = string.split('-')

                return ((int(coord[0]),int(coord[1])), (int(coord[2]), int(coord[3])))


