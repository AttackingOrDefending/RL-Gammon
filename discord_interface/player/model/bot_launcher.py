
from discord_interface.player.instances.gtp_ai import GTP_AI
from discord_interface.player.instances.gtp_ai_go import GTP_AI_Go
from discord_interface.player.model.player_bot import PlayerBot
from discord_interface.player.model.player_bot import Player
from discord_interface.utils.configuration_files import load_configurations
from discord_interface.utils.mytime import *
from discord_interface.player.model.cogs.general_cog import General
from discord_interface.player.model.cogs.setting_cog import Setting
from logging import FileHandler
from discord.flags import Intents
import asyncio, os
from typing import Type, Tuple


async def setup(AIProgram_cls: Type[Player] = None, owner_id: str = "", guild_id: str = "", **AIProgram_args) -> Tuple[PlayerBot, FileHandler]:
    """Coroutine that sets up the PlayerBot with its AI program, and its gamemode.

    PARAMETERS
    ----------
    AIProgram_cls : type[player.Player]
        The class inheriting from Player that will be the program of the model.
    Gamemode_cls : type[mygame.Game]
        The class inheriting from Game that will be the game mode of the model.
    owner_id : str
        The string representation of the owner's discord_interface id.

    RETURNS
    -------
    Tuple[player_bot.PlayerBot, logging.FileHandler]
        The model instanciated with its file handler.
    """
    # Retrieve the current date, and format it YYYY-mm-dd
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")

    # Setting up intents for our model
    intents = Intents.default()
    intents.message_content = True

    # File handler setup
    if f'discord_player_{date}.log' in os.listdir('log/error_handling'):
        file_handler = FileHandler(filename=f'log/error_handling/discord_player_{date}.log', mode='a',
                                   encoding='utf-8')  # If the log file already exists, we append new content at the end
    else:
        file_handler = FileHandler(filename=f'log/error_handling/discord_player_{date}.log', mode='w',
                                   encoding='utf-8')

    # Intialise the PlayerBot instance
    bot = PlayerBot(AIProgram_cls(**AIProgram_args), owner_id=int(owner_id), guild_id=int(guild_id), intents=intents, date=date)

    # Adding cogs to the Bot
    await bot.add_cog(General(bot))
    await bot.add_cog(Setting(bot))

    return bot, file_handler

def numbered_bot_starting(AI_Class, player_number=1, **AI_args):

    config=load_configurations()

    if player_number==1:
        TOKEN_player = config['PLAYER_BOT_DISCORD_TOKEN']
    else:
        TOKEN_player = config['PLAYER_BOT_'+str(player_number)+'_DISCORD_TOKEN']

    advanced_bot_starting(AI_Class, TOKEN_player, **AI_args)
def advanced_bot_starting(AI_Class, TOKEN_player, **AI_args):

    config=load_configurations()

    OWNER_ID = config['OWNER_ID']

    if config['BETA_TEST_MODE']:
        GUILD_ID = config['BETA_TEST_COMPUTER_OLYMPIAD_GUILD_ID']
    else:
        GUILD_ID = config['COMPUTER_OLYMPIAD_GUILD_ID']  # os.getenv('GUILD_ID')

    bot, file_handler = asyncio.run(setup(AIProgram_cls=AI_Class, owner_id=OWNER_ID, guild_id=GUILD_ID, **AI_args))

    bot.run(TOKEN_player, log_handler=file_handler)


def bot_starting(AI_Class, **AI_args):
    numbered_bot_starting(AI_Class, player_number=1, **AI_args)

def gpt_bot_starting(program_name, program_arguments='', program_directory=''):

    if program_arguments is None:
        program_arguments = ''

    if not isinstance(program_arguments, str):
        program_arguments = ' '.join([str(e) for e in program_arguments])

    bot_starting(GTP_AI, program_name=program_name, program_arguments=program_arguments, program_directory=program_directory)


def go_gpt_bot_starting(program_name, program_arguments='', program_directory=''):

    if program_arguments is None:
        program_arguments = ''

    if not isinstance(program_arguments, str):
        program_arguments = ' '.join([str(e) for e in program_arguments])

    bot_starting(GTP_AI_Go, program_name=program_name, program_arguments=program_arguments, program_directory=program_directory)
