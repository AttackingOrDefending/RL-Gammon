from discord_interface.games.translator.quentin_games.Lines_of_action_olympiad import Lines_of_action_olympiad
from discord_interface.games.translator.quentin_games.Santorini import Santorini
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class LinesOfActionDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('lines_of_action', Lines_of_action_olympiad, params=None, rules='standard', move_keywords=[])

