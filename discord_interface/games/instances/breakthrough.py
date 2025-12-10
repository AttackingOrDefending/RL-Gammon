from discord_interface.games.translator.quentin_games.Breakthrough import Breakthrough
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord



class BreakthroughDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('breakthrough', Breakthrough, params=None, rules='standard', move_keywords=[])

