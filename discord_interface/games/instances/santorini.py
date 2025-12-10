from discord_interface.games.translator.quentin_games.Santorini import Santorini
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class SantoriniDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('santorini', Santorini, params=None, rules='standard', move_keywords=[])

