from discord_interface.games.translator.quentin_games.Hex_swap_19 import Hex_swap_19
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class Hex19Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('hex19', Hex_swap_19, params=None, rules='with swap', move_keywords=['swap'])



