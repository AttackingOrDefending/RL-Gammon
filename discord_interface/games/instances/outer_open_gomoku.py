
from discord_interface.games.translator.quentin_games.Outer_open_gomoku import Outer_open_gomoku

from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class OuterOpenGomokuDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('outer_open_gomoku', Outer_open_gomoku, params=None, rules='outer-open and free-style', move_keywords=[])

