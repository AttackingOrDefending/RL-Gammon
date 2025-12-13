import sys
from logging import FileHandler
from datetime import datetime
from discord_interface.referee.model.referee_bot import RefereeBot
from discord_interface.referee.model.cogs.general_cog import GeneralCog
from discord_interface.referee.model.cogs.game_cog import GameCog
import asyncio, os

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

from discord_interface.utils.configuration_files import load_configurations


async def setup_referee(guild_id: str, channel_id: str) -> tuple[RefereeBot, FileHandler]:
    """Coroutine that sets up the RefereeBot, its File Handler for logging, and adds the Cogs"""
    # Retrieve current time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    if 'log' not in os.listdir('.'):
        os.mkdir('log')
    else:
        if 'error_handling' not in os.listdir('log'):
            os.mkdir('log/error_handling')

    if f'discord_ref_{current_date}.log' in os.listdir('log/error_handling'):
        file_handler_referee = FileHandler(filename=f'log/error_handling/discord_ref_{current_date}.log', mode='a',
                                                   encoding='utf-8')  # if the log file already exists, we append new content at the end
    else:
        file_handler_referee = FileHandler(filename=f'log/error_handling/discord_ref_{current_date}.log', mode='w',
                                                   encoding='utf-8')

    # Create the RefereeBot instance
    refbot = RefereeBot(current_date, guild_id=int(guild_id), channel_id=int(channel_id) )

    # Add cogs
    await refbot.add_cog(GameCog(refbot))
    await refbot.add_cog(GeneralCog(refbot))

    return refbot, file_handler_referee


def start_referee(referee_index):

    config=load_configurations()#load_dotenv()


    TOKEN_referee = config['REFEREE_BOT_'+str(referee_index)+'_DISCORD_TOKEN']#os.getenv('REFEREE_BOT_DISCORD_TOKEN')


    if config['BETA_TEST_MODE']:
        ALLOWED_GUILD = config['BETA_TEST_COMPUTER_OLYMPIAD_GUILD_ID']
        ALLOWED_CHANNEL = config['BETA_TEST_REFEREE_BOT_'+str(referee_index)+'_DISCORD_CHANNEL']
    else:
        ALLOWED_GUILD = config['COMPUTER_OLYMPIAD_GUILD_ID']#os.getenv('GUILD_ID')
        ALLOWED_CHANNEL = config['REFEREE_BOT_'+str(referee_index)+'_DISCORD_CHANNEL']

    bot, file_handler = asyncio.run(setup_referee(ALLOWED_GUILD, ALLOWED_CHANNEL))
    bot.run(TOKEN_referee, log_handler=file_handler)