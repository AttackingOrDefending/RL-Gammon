from discord_interface.games.translator.quentin_games.Hex_swap_11 import Hex_swap
from discord_interface.games.translator.quentin_games.Othello8 import Othello8


class Othello10(Othello8):

    def __init__(self):
        super().__init__(taille=10)