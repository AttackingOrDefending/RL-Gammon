import random
from asyncio import sleep

from discord_interface.player.instances.random_ai import RandomAI
from discord_interface.utils.mytime import Time

class LongRandomAI(RandomAI):

    def plays(self, time_left: Time=None, opponent_time_left: Time=None):

        action = self.best_move(self.game)

        self.game.plays(action)

        sleep(random.randint(1, 24) + int(random.randint(0, 10) == 0) * 60)

        return action
