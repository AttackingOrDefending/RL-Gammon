import asyncio
from collections import defaultdict

from discord import Intents
from discord.ext import commands
from discord.ext.commands import DefaultHelpCommand
from discord_interface.referee.model.referee import *
from discord_interface.games.games_enum import *
from json import JSONDecodeError
from datetime import *
import json, os
from discord.ext.commands import Context

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

# Type alias

#load_dotenv()

if __name__ != "__main__":

    class RefereeBot(commands.Bot):
        """Class representing a referee model instance

        It inherits from discord.ext.commands.Bot, and only overrides the constructor and the event on_ready.
        It also has attributes that are not inherited from discord.ext.commands.Bot.

        ATTRIBUTES
        ----------
        ... (for those from commands.Bot see discord.py API reference)
        referee : referee.Referee
            The referee instance associated with the model.
        _date : str
            Date used for logging purpose
        _timer_cancel : bool
            Attribute used to stop the timer during game preparation.
        _json_file : str
            Path leading to the file that contains the game information.
        _bot_ref_log : dict[str, dict[str, [str | int | bool | list[str]]]]
            Structure that is json serializable and that stores game information.
        """
        #ALLOWED_GUILD = int(os.getenv('GUILD_ID'))

        def __init__(self, current_date: str, guild_id: int, channel_id: int) -> None:

            self.guild_id=guild_id
            self.channel_id=channel_id

            # Setting intents of our client
            intents = Intents.default()
            intents.message_content = True
            intents.members = True

            # Call to the parent class' constructor
            super().__init__(command_prefix='!', intents=intents,
                         help_command=DefaultHelpCommand(dm_help=True, show_hidden=True, verify_checks=False))

            # Initialising attributes
            self.referee = Referee(game=EnumGames.get_available_games()[0])
            self._date = current_date
            self._timer_cancel = False
            self._json_file = ''
            self._bot_ref_log = {}

            self.operator = defaultdict(lambda:None)



        def check_in_game(self) -> bool:
            """Check function that verifies whether the referee instance is in game or not"""
            return self.referee.in_game

        def check_in_preparation(self) -> bool:
            """Check function that verifies whether the referee instance is in preparation or not"""
            return self.referee.in_preparation

        def check_in_pause(self) -> bool:
            """Check function that verifies whether the referee instance is paused or not"""
            return self.referee.paused

        def check_in_end_game(self) -> bool:
            """Check function that verifies whether the referee instance is in end-game or not"""
            return self.referee.in_end_game

        def check_resumed_author(self, ctx: Context) -> bool:
            """Check function that verifies if the author of the contextual message is the one who paused the game"""
            return ctx.message.author == self.referee.get_paused_by()

        def check_player_in_game(self, ctx: Context) -> bool:
            """Check function that verifies if the author of the contextual message is one of the players of the ongoing game (if such a game is going on)"""
            #^#
            return ctx.message.author in (self.referee.players if self.referee.players else []) or ctx.message.author in [self.operator[player] for player in self.referee.players]

        def check_stop_activated(self) -> bool:
            """Check function that verifies whether the referee instance has the stop option activated or not"""
            return self.referee.is_stop_activated()


        def set_json_file(self, guild_name: str, channel_name: str, date: str, starting_time: str):
            self._json_file = os.path.join("log/bot_ref_log", f"ref_{'-'.join(guild_name.split())}_{'-'.join(channel_name.split())}_{date}_{starting_time}.json")


        @property
        def json_file(self) -> str:
            return self._json_file

        @property
        def date(self) -> str:
            return self._date

        @property
        def timer_cancel(self) -> bool:
            return self._timer_cancel

        @property
        def bot_ref_log(self) -> dict:
            return self._bot_ref_log

        @bot_ref_log.setter
        def bot_ref_log(self, value):
            self._bot_ref_log = value

        async def setup_hook(self) -> None:
            # Verify if the logging directory already exists in the current working directory
            try:
                if 'bot_play_log' not in await asyncio.to_thread(os.listdir, 'log'):
                    await asyncio.to_thread(os.mkdir, 'log/bot_ref_log')

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        async def on_ready(self) -> None:
            """Coroutine that is triggered when the model is ready and that sets up json file path and logging utils."""
            #print('on ready...')
            for guild in self.guilds:
                print(f'We have logged in as {self.user} on {guild.name}')

        def correct_context(self, ctx: Context):
            #print( ctx.guild.id , self.guild_id, '/' ,ctx.channel.id , self.channel_id)
            return ctx.guild.id == self.guild_id and ctx.channel.id == self.channel_id

        #@classmethod
        #def check_guild(cls):
        def check_guild(self):
            def predicate(ctx: Context):
                return ctx.guild.id == self.guild_id and ctx.channel.id == self.channel_id

            return commands.check(predicate)


