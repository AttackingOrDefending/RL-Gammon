from discord import User

from discord_interface.games.games_enum import EnumGames
from discord_interface.utils.mytime import Time
from abc import ABCMeta, abstractmethod
from discord_interface.games.mygame import Game
from typing import Any, List
from discord import TextChannel

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

if __name__ != "__main__":

    class Player(metaclass=ABCMeta):
        """class to represent a Player and its attributes

        ATTRIBUTES
        ----------
        last_actions_opponents : list[str]
            List that contains the last actions played by the opponents during the current game.
        last_actions_self : list[str]
            List that contains the last actions played by the instance itself.
        in_game : bool
            Boolean variable that indicates whether the game has started or not.
        starting_time : mytime.Time
            Time instance that represents the moment when the game has started.
        game : mygame.Game
            Game instance associated with this player.
        opponent : discord.User
            User that our instance is competing against in the current game.
        channel : discord.TextChannel
            Text channel associated with the ongoing game. It is where the instance will read/write actions.
        referee_id : int
            Discord id of the referee in the ongoing game.

        game_name: str
            Name of the game that is being played. The information is retrieved at the beginning of the game
        total_time: mytime.Time
            Time instance that represents the amount of time the player is given to play.
        """

        def __init__(self,
                     last_actions_opponents:List[str] = None,
                     last_actions_self:List[str] = None,
                     in_game:bool = False,
                     starting_time:Time = Time(),
                     game:Game = None,
                     opponent:User = None,
                     channel:TextChannel = None,
                     referee_id:int = None,
                     game_name:str = None,
                     total_time:Time = None
                     ) -> None:
            if last_actions_opponents:
                self.last_actions_opponents = last_actions_opponents
            else:
                self.last_actions_opponents = []
            if last_actions_self:
                self.last_actions_self = last_actions_self
            else:
                self.last_actions_self = []
            self.in_game = in_game
            self.starting_time = starting_time
            self.game = game
            self.opponent = opponent
            self.channel = channel
            self.referee_id = referee_id
            self.game_name = game_name
            self.total_time = total_time

            """if game is None:
                self.update_game(game_name)
            else:
                self.game = game
                if game_name is None:
                    self.game_name = self.game.name
                else:
                    assert game_name == self.game.name
                    self.game_name = game_name"""


        def __str__(self) -> str:
            return (f"* last_actions_opponents : {self.last_actions_opponents}\n"
                    f"* last_actions_self : {self.last_actions_self}\n"
                    f"* in_game : {self.in_game}\n"
                    f"* starting_time : {self.starting_time}\n"
                    f"* game : {self.game.name}\n"
                    f"* opponent : {self.opponent}\n"
                    f"* channel : {self.channel}\n"
                    f"* referee_id : {self.referee_id}\n"
                    f"* game_name : {self.game_name}\n"
                    f"* total_time : {self.total_time}")

        def set_starting_time(self, starting_time:Time) -> None:
            self.starting_time = starting_time

        def set_channel(self, channel: TextChannel) -> None:
            self.channel = channel

        def set_opponent(self, opponent: User) -> None:
            self.opponent = opponent

        def set_referee_id(self, referee_id: int) -> None:
            self.referee_id = referee_id

        def set_total_time(self, total_time: Time) -> None:
            self.total_time = total_time

        def start(self) -> None:
            self.in_game = True

        async def end(self) -> None:
            self.in_game = False

        @abstractmethod
        def opponent_plays(self, action: str) -> None:
            pass

        @abstractmethod
        def plays(self, time_left: Time=None, opponent_time_left: Time=None) -> str:
            pass

        def update_game(self, game_name:str):
            game = EnumGames.find_game(game_name)
            if game is None:
                raise Exception(f"Error, {game_name} not found in EnumGames")
            else:
                self.game = game
                self.game_name = game_name



        def string_to_action(self, string: str) -> Any:
            if self.game is None:
                return string
            else:
                return self.game.string_to_action(string)

        def action_to_string(self, action: object) -> str:
            if self.game is None:
                if str(action)[-1].isdigit():
                    return str(action).upper()
                return str(action)#.upper()
            else:
                return self.game.action_to_string(action)

        async def reset(self) -> None:
            #print('+ reset')
            self.__init__(game=self.game)
            if self.game is not None:
                self.game.reset()
            #print('- reset')

        def is_in_game(self) -> bool:
            return self.in_game

        def get_opponent(self) -> User:
            return self.opponent

        async def invalid_action_processing(self):
            """"""