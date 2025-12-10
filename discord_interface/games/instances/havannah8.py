from discord_interface.games.translator.quentin_games.Havannah_swap_8 import Havannah_swap_8
from discord_interface.games.translator.quentin_games.Hex_swap_11 import Hex_swap
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord

class Havannah8Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('havannah8', Havannah_swap_8, params=None, rules='with swap', move_keywords=['swap'])



