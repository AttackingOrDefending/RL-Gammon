from discord_interface.games.translator.quentin_games.Othello10 import Othello10
from discord_interface.games.translator.quentin_games.Othello16 import Othello16
from discord_interface.games.translator.quentin_games.Othello8 import Othello8
from discord_interface.games.translator.quentin_games.Santorini import Santorini
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class Othello16Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('othello16', Othello16, params=None, rules='standard', move_keywords=[])

