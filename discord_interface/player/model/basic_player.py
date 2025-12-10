import asyncio
import random
from math import inf

from discord_interface.player.model.player import *
from discord_interface.utils.mytime import Time

class BasicPlayer(Player):


    def init_history(self):
        self.textual_move_history = []

    def update_history(self, action):
        self.textual_move_history.append(self.game.action_to_string(action))

    def __init__(self, game:Game=None):
        self.init_history()

        if not hasattr(self, 'lock'):
            self.lock = asyncio.Lock()
        super().__init__(game=game)

    async def reset(self):
        self.init_history()
        await super().reset()

    def opponent_plays(self, action):
        self.game.plays(action)
        self.update_history(action)

    def replays(self, action):
        self.game.plays(action)
        self.update_history(action)


    async def plays(self, time_left: Time=None, opponent_time_left: Time=None):
        async with self.lock:
            textual_move = await asyncio.to_thread(self.my_plays, self.textual_move_history, time_left.to_seconds(), opponent_time_left.to_seconds())

            action = self.game.string_to_action(textual_move)

            self.game.plays(action)
            self.update_history(action)

            return action


    @abstractmethod
    def my_plays(self, game_history, time_left=inf, opponent_time_left=inf):
        """must return my next move in string format (examples: "A1" or "B2-C3") """
        pass
