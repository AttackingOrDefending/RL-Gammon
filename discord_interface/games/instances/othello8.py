from discord_interface.games.translator.quentin_games.Othello8 import Othello8
from discord_interface.games.translator.quentin_games.Santorini import Santorini
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class Othello8Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('othello8', Othello8, params=None, rules='standard', move_keywords=[])

