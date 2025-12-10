

from discord_interface.player.model.bot_launcher import gpt_bot_starting, go_gpt_bot_starting
from discord_interface.utils.configuration_files import load_configurations

if '__main__' == __name__:

    gpt = load_configurations()

    go_gpt_bot_starting(program_name=gpt['program_name'], program_arguments=gpt['program_arguments'], program_directory=gpt['program_directory'] if gpt['program_directory'] is not None else '')