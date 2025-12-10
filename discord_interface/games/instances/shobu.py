from discord_interface.games.translator.quentin_games.Santorini import Santorini
from discord_interface.games.translator.quentin_games.Shobu_hiera_passif import Shobu_hiera_passif
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class ShobuDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('shobu', Shobu_hiera_passif, params=None, rules='any repetition is a draw', move_keywords=[])

