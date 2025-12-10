from discord_interface.games.translator.quentin_games.Amazons import Amazons
from discord_interface.games.translator.quentin_games.Xiangqi import Xiangqi
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord




class ChineseChessDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('chinese_chess', Xiangqi, params=None, rules='any repetition is a loss', move_keywords=[])

