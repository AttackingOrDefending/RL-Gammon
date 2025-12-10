import sys

from discord_interface.referee.referee_launcher import start_referee

i = int(sys.argv[0].split('_')[-1].replace('.py' ,''))

start_referee(i)