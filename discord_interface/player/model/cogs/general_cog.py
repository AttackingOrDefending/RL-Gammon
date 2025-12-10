import os
import json
from json import load, JSONDecodeError

from discord import Colour, Embed
from discord.ext import commands
from discord_interface.player.model.player_bot import PlayerBot
from discord.ext.commands import Context

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"


if __name__ != "__main__":

    class General(commands.Cog):
        """Class that inherits commands.Cog, that represents a fraction of the commands and features of the PlayerBot that are about General cases.

        ATTRIBUTES
        ----------
        bot : player_bot.PlayerBot
            The model to which this Cog will be added.
        """

        def __init__(self, bot: PlayerBot) -> None:
            self.bot = bot

        @commands.command(name="history", description="Shows the history of the player model for today's game session.")
        @commands.dm_only()
        async def _history(self, ctx: Context, *args: None) -> None:
            """Coroutine that implements the history command by displaying the date, the type of time management system, the winner, and the looser of the registered games."""

            # If the author is not the owner
            if not self.bot.check_owner(ctx.author):
                return

            # If the model is in game, impossible to invoke this command...
            if self.bot.check_in_game():
                raise commands.CheckFailure("The player instance shouldn't be in game when invoking the history command.")

            # Going through the logs
            desc = ""

            # Iterating through the player's log files
            for file in os.listdir("log/bot_play_log"):

                # If it is not a json file with "play" prefix, then the file is of no use for us
                if not file.startswith("play") and not file.endswith(".json"):
                    continue

                # ...Otherwise, recover the information stored in the filename
                _, user_id, guild_name, channel_name, date, starting_time = file.strip(".json").split("_")

                # Do not consider opponents' files
                if int(user_id) != self.bot.user.id:
                    continue

                # For each file of use, try to load it and use its information in the final display
                with open(os.path.join("log/bot_play_log", file), mode="r") as fd:
                    try:
                        loading = json.load(fd)
                    except JSONDecodeError:
                        print("ERROR JSON")
                    else:
                        if loading['ended']:
                            desc += f"**On {date.replace('-', '.')} at {'h'.join(starting_time.split('-')[0:-1])}** {guild_name.replace('-', ' ')} (guild) - {loading['game_name'].capitalize()} - {'Winner' if loading['winner'] else 'Loser'}\n"

            # If the description is empty it means that there were no file that recorded games on the current guild
            if desc != "":
                embed = Embed(title="Matches History", description=desc, color=Colour.random())
                await ctx.send(embed=embed)
            else:
                await ctx.message.reply(f"The files have no match recorded for the lasts sessions !")
