from discord_interface.games.translator.quentin_games.Ataxx import Ataxx
from discord_interface.games.translator.quentin_games.Surakarta_olympiad import Surakarta_olympiad
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord



class AtaxxDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('ataxx', Ataxx, params=None, rules='any repetition ends: winner = the most pieces', move_keywords=[])

