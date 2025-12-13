
from discord_interface.games.translator.quentin_games.Surakarta_olympiad import Surakarta_olympiad
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord



class SurakartaDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('surakarta', Surakarta_olympiad, params=None, rules='any repetition or 25 player turns without capture for each player ends: winner = the most pieces.', move_keywords=[])

