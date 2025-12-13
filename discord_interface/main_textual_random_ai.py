import sys
import os
#print(sys.path)
# Ajouter le dossier parent au PYTHONPATH
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(os.path.dirname(os.path.abspath(__file__).split('discord_interface')[0]))
#sys.path.append(os.path.dirname(os.path.abspath(__file__).split('discord_interface')[0]+'/discord_interface'))

#print(os.path.abspath(__file__).split('discord_interface')[0], os.path.dirname(os.path.abspath(__file__).split('discord_interface')[0]))
#print(os.path.abspath(__file__).split('discord_interface')[0]+'/discord_interface', os.path.dirname(os.path.abspath(__file__).split('discord_interface')[0]+'/discord_interface'))
#print(sys.path)

from discord_interface.player.instances.textual_random_ai import TextualRandomAI
from discord_interface.player.model.bot_launcher import bot_starting

if '__main__' == __name__:


    bot_starting(TextualRandomAI)