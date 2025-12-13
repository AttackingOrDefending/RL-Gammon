import asyncio
import random
from math import inf

from discord_interface.player.model.basic_player import BasicPlayer
from discord_interface.player.model.player import *
from discord_interface.utils.mytime import Time

class AdvancedBasicPlayer(BasicPlayer):

    async def end(self):

        async with self.lock:
            await asyncio.to_thread(self.my_end)

            await super().end()

    @abstractmethod
    def my_end(self):
        pass


    async def update_game(self, game_name:str):

        async with self.lock:
            super().update_game(game_name)

            await asyncio.to_thread(self.my_initialisation, self.game_name)

    @abstractmethod
    def my_initialisation(self, game_name):
        pass
