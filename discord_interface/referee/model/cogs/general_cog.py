import aiofiles
from discord import Colour, Embed, User
from discord.ext import commands
from discord_interface.referee.model.referee_bot import RefereeBot
import json
from discord.ext.commands import Context

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"


if __name__ != "__main__":

    class GeneralCog(commands.Cog):
        """Cog containing commands, and listeners related to general use cases.

            ATTRIBUTES
            ----------
            bot : RefereeBot
                Bot instance that is defined in RefereeBot.
            """

        def __init__(self, refbot: RefereeBot):
            self.bot = refbot


        @commands.command(name = 'info', description = 'Show information about the referee and the game.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _info(self, ctx: Context, *args:None) -> None:
            """Command that shows general information about the referee, and the current game associated.

            It sends on the same channel where the command was invoked an embed message that contains all the
            attributed of the referee global instance.

            PARAMETERS
            ----------
            ctx : discord.Context
                The context in which the command was invoked

            RETURNS
            -------
            None
            """

            if not self.bot.correct_context(ctx):
                return

            embedVar = Embed(title="Referee's Settings", color=Colour.random())
            embedVar.add_field(name="Referee's info", value=self.bot.referee.__str__())
            embedVar.add_field(name="Game's info", value=self.bot.referee.game.__str__())

            try:
                await ctx.send(embed=embedVar)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
        @commands.command(name = 'dump', description = 'Dumps into a JSON file the record of all commands that were executed with the time.')
        @commands.has_permissions(administrator = True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _dump(self, ctx: Context, *args:None) -> None:
            """Command that dumps the content of the dictionary _bot_ref_log into its associated JSON file.

            The file is dated using only the day.
            The structure of the file is the following
                {starting_time_game1 :
                    {id_p1 : <[<move1>, <move2> ... moveN]>,
                     id_p2 : [move1, move2 ... moveM]},
                     game_name : <name>,
                     time_per_move : <True/False>,
                     time_per_player : <seconds>,
                     winner : <id_winner>,
                     loser : <id_loser>
                 starting_time_game2 : ...
                }

            PARAMETERS
            ----------
            ctx : discord.Context
                The context in which the command was invoked

            RETURNS
            -------
            None
            """

            if not self.bot.correct_context(ctx):
                return

            if self.bot.check_in_game():
                raise commands.CheckFailure(message ="The model shouldn't be in game when invoking the dump command.")

            if self.bot.check_in_preparation():
                raise commands.CheckFailure(message="The model shouldn't be in preparation when invoking the dump command.")

            # Dump the structure into the file

            await self.save_bot_ref_log()
            try:
                await ctx.send('dumped !')

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name = 'reset', description = 'resets the current referee instance.')
        @commands.has_permissions(administrator = True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _reset(self, ctx: Context, *args:None) -> None:
            """Command that resets the current referee instance."""

            if not self.bot.correct_context(ctx):
                return

            self.bot.referee.reset()
            try:
                await ctx.send(f'The referee was reset...')

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name = 'quit', description = 'Command that shuts down the current model (for debugging purposes).')
        @commands.has_permissions(administrator = True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _quit(self, ctx: Context, *args:None) -> None:
            """Command that shuts down the model."""
            # Save the log structure

            if not self.bot.correct_context(ctx):
                return

            await self.save_bot_ref_log()

            try:
                await ctx.send('Shutting down...')
                await self.bot.close()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            ###

        async def save_bot_ref_log(self):
            try:
                async with aiofiles.open(self.bot.json_file, 'w') as f:
                    """            with open(self.bot.json_file, 'w') as file:
                    json.dump(self.bot.bot_ref_log, file)"""
                    await f.write(json.dumps(self.bot.bot_ref_log))
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e



        @commands.command(name = 'increase_time', description = 'Increase the time of player X by Y seconds.')
        @commands.has_permissions(administrator = True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _increase_time(self, ctx: Context, user:User, time:int) -> None:
            """Command that resets the current referee instance."""

            if not self.bot.correct_context(ctx):
                return

            if not self.bot.check_in_game():
                raise commands.CheckFailure(message ="The model should be in game when invoking the dump command.")


            try:
                self.bot.referee.increase_time(user, time)

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e