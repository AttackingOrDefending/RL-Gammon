from discord_interface.games.translator.quentin_games.Amazons import Amazons
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord




class AmazonsDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('amazons', Amazons, params=None, rules='standard', move_keywords=[])

