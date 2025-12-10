
from discord_interface.games.translator.quentin_games.Hex_swap_13 import Hex_swap_13
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class Hex13Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('hex13', Hex_swap_13, params=None, rules='with swap', move_keywords=['swap'])



