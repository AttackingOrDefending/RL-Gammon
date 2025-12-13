from discord_interface.player.instances.gtp_ai_go import GTP_AI_Go
from discord_interface.player.model.textual_ai import Textual_AI
from discord_interface.utils.mytime import Time


class GTP_AI(GTP_AI_Go):


    def self_color(self):
        return str(self.player_number)


    def opponent_color(self):
        return str(1-self.player_number)

    def move_conversion_to_gtp(self, action):
        return action

    def move_conversion_from_gtp(self, move):
        return move