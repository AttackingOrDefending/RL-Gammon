from discord_interface.games.translator.quentin_games.Arimaa_vrai import Arimaa_vrai
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord



class ArimaaDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('arimaa', Arimaa_vrai, params=None, rules='standard', move_keywords=['pass'])

    def string_to_action(self, string):
        if string.count('-') == 3:
            l = string.split('-')
            b = l[0]+'-'+l[1]
            a = l[2]
            return (super().string_to_action(a), super().string_to_action(b))
        if string.count('-') == 2:
            l = string.split('-')
            a = l[0]+'-'+l[1]
            b = l[2]
            return (super().string_to_action(a), super().string_to_action(b))
        else:
            return super().string_to_action(string)