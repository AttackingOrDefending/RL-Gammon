from discord_interface.games.translator.draughts_quentin_games_interface import DraughtsInterfaceJeuDiscord
from discord_interface.games.translator.quentin_games.Dames_bresiliennes import BrazilianDraughts
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class BrazilianDraughtsDiscord(DraughtsInterfaceJeuDiscord):

    def __init__(self):
        super().__init__('brazilian_draughts', BrazilianDraughts, params=None, rules='draw if: 25 moves per player with only king movements OR 16 turns per player after 3 kings, 2 kings + 1 piece, or 1 king + 2 pieces vs. 1 king OR 2 kings vs. 1 king OR 1 king vs. 1 king', move_keywords=[])

