from discord_interface.games.translator.quentin_games.Connect6 import Connect6
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class Connect6Discord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('connect6', Connect6, params=None, rules='standard', move_keywords=[])

