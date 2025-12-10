"""File used for game enumeration.

This file contains a class called EnumGames that stores games used by the RefereeBot as well as the PlayerBot.
"""
from enum import Enum#, verify, UNIQUE

from discord_interface.games.instances.amazons import AmazonsDiscord
from discord_interface.games.instances.arimaa import ArimaaDiscord
from discord_interface.games.instances.ataxx import AtaxxDiscord
from discord_interface.games.instances.brazilian_draughts import BrazilianDraughtsDiscord
from discord_interface.games.instances.breakthrough import BreakthroughDiscord
from discord_interface.games.instances.canadian_draughts import CanadianDraughtsDiscord
from discord_interface.games.instances.chinese_chess import ChineseChessDiscord
from discord_interface.games.instances.clobber import *

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

from discord_interface.games.instances.connect6 import Connect6Discord

from discord_interface.games.instances.free_game import FreeGame
from discord_interface.games.instances.havannah10 import Havannah10Discord
from discord_interface.games.instances.havannah8 import Havannah8Discord
from discord_interface.games.instances.hex11 import Hex11Discord
from discord_interface.games.instances.hex13 import Hex13Discord
from discord_interface.games.instances.hex19 import Hex19Discord
from discord_interface.games.instances.international_draughts import InternationalDraughtsDiscord
from discord_interface.games.instances.lines_of_action import LinesOfActionDiscord
from discord_interface.games.instances.othello10 import Othello10Discord
from discord_interface.games.instances.othello16 import Othello16Discord
from discord_interface.games.instances.othello8 import Othello8Discord
from discord_interface.games.instances.outer_open_gomoku import OuterOpenGomokuDiscord
from discord_interface.games.instances.quoridor import QuoridorDiscord
from discord_interface.games.instances.santorini import SantoriniDiscord
from discord_interface.games.instances.shobu import ShobuDiscord
from discord_interface.games.instances.surakarta import SurakartaDiscord

if __name__ != "__main__":

    #@verify(UNIQUE)
    class EnumGames(Enum):
        CANADIAN_DRAUGHTS = CanadianDraughtsDiscord()

        BREAKTHROUGH = BreakthroughDiscord()
        SANTORINI = SantoriniDiscord()
        AMAZONS = AmazonsDiscord()
        CLOBBER = Clobber()
        #AMAZONS_BUGUER = AmazonsDiscordBugger()
        LINES_OF_ACTION = LinesOfActionDiscord()
        HEX11 = Hex11Discord()
        HEX13 = Hex13Discord()
        HEX19 = Hex19Discord()
        SURAKARTA = SurakartaDiscord()
        OUTER_OPEN_GOMOKU = OuterOpenGomokuDiscord()
        CONNECT6 = Connect6Discord()
        INTERNATIONAL_DRAUGHTS = InternationalDraughtsDiscord()
        BRAZILIAN_DRAUGHTS = BrazilianDraughtsDiscord()
        SHOBU = ShobuDiscord()
        ARIMAA = ArimaaDiscord()
        CHINESE_CHESS = ChineseChessDiscord()
        ATAXX = AtaxxDiscord()
        HAVANNAH8 = Havannah8Discord()
        HAVANNAH10 = Havannah10Discord()
        OTHELLO8 = Othello8Discord()
        OTHELLO10 = Othello10Discord()
        OTHELLO16 = Othello16Discord()
        QUORIDOR = QuoridorDiscord()
        FREE_GAME = FreeGame()
        #GTP_GAME = GTP_Game()
        #ECHECS = EchecDiscord()




        #**** .................... ****"
        #**** add other games here ****#

        def __eq__(self, other):
            if type(other) != str and type(other) != self.__class__:
                raise TypeError()
            return other == self.value

        @classmethod
        def get_available_games(cls) -> List[Game]:
            """Class method that returns a list of the different available games"""
            return [cls.__members__[game].value.__class__() for game in cls.__members__]

        @classmethod
        def find_game(cls, game_name:str):
            """Method that tries to find the game specified by game_name in the enumeration"""
            if game_name.upper() not in EnumGames.__members__:
                return None
            return EnumGames.__members__[game_name.upper()].value.__class__()


def Discord_Game(game_name):

    game_name = game_name.lower()

    for game_members in EnumGames.__members__:

        if game_name == game_members.lower():
            return EnumGames[game_members].value

    """if game_name == 'santorini':
        return SantoriniDiscord()
    elif game_name == 'clobber':
        return Clobber()
    elif game_name == 'amazons':
        return AmazonsDiscord()
    elif game_name == 'free_game':
        return FreeGame()
    elif game_name == 'echecs':
        return EchecDiscord()
    elif game_name == 'amazons-bugguer':
        return AmazonsDiscordBugger()

    elif game_name == 'gtp':
        return GTP_Game()"""