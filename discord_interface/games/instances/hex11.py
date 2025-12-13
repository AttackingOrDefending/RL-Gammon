
from discord_interface.games.translator.quentin_games.Hex_swap_11 import Hex_swap
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord

class Hex11Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('hex11', Hex_swap, params=None, rules='with swap', move_keywords=['swap'])



