from discord_interface.games.translator.quentin_games.Havannah_swap_10 import Havannah_swap_10
from discord_interface.games.translator.quentin_games.Havannah_swap_8 import Havannah_swap_8
from discord_interface.games.translator.quentin_games.Hex_swap_11 import Hex_swap
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord

class Havannah10Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('havannah10', Havannah_swap_10, params=None, rules='with swap', move_keywords=['swap'])



