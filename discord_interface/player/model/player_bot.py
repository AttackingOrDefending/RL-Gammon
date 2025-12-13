from asyncio import sleep
from collections import defaultdict
from datetime import datetime
from json import JSONDecodeError
from time import time

import aiofiles
from discord import DMChannel, Intents, Message, Reaction, Object
from discord.abc import User
from discord.ext import commands

from discord_interface.utils.mymessage import TimedInstructionMessage
from discord_interface.utils.pattern_enum import *
from discord_interface.player.model.player import *
import asyncio, json, os

import inspect

# Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

from discord_interface.utils.terminal import red

#load_dotenv()

if __name__ != "__main__":

    class PlayerBot(commands.Bot):
        """Class that inherits commands.Bot, that represents a player in the game

        ATTRIBUTES
        ----------
        ... (for those from commands.Bot see discord.py API reference)
        player : player.Player
            Player instance that is associated with the model.
        owner_id : int
            Discord identifier of the user that is using the model.
        date : str
            Field used for logging purposes.
        _json_file : str
            Path to the json file that will be loaded for logging game information.
        _bot_play_log : dict[str, dict[list[str] | str | bool]]
            Dictionary that keeps track of games that were played today with the following format :
                {'hh:mm': {'moves': [m1, ..., mk], 'game_name': name, 'winner': true/false}
        opponent_instruction_message : TimedInstructionMessage
            Last instruction message sent on the game channel for the opponent. Used to compute opponent's time left.
        """
        #ALLOWED_GUILD = int(os.getenv('GUILD_ID'))

        def __init__(self, player: Player, owner_id: int, guild_id: int, intents: Intents, date: str) -> None:
            super().__init__(command_prefix='!', intents=intents,
                             help_command=None)  # DefaultHelpCommand(dm_help=True, show_hidden=True, verify_checks=False))


            self.freezed = False
            self.freezed_message = None

            self.player = player  # setting up the Player instance that will be modified through the execution
            self.owner_id = owner_id
            self.guild_id = guild_id
            self.date = date
            self._json_file = ''
            self._bot_play_log = {}
            self.opponent_instruction_message = TimedInstructionMessage()

            self.last_id = None
            #*#self.last_channel = None
            #*#self.last_channel_id = None
            self.processed_ids = set()

            self.message_lock = asyncio.Lock()

            self.async_plays = inspect.iscoroutinefunction(self.player.plays)
            self.async_opponent_plays = inspect.iscoroutinefunction(self.player.opponent_plays)
            self.async_update_game = inspect.iscoroutinefunction(self.player.update_game)
            self.async_replays = inspect.iscoroutinefunction(self.player.replays)

            self.message_profiler = {}
            self.reaction_profiler = {}
            self.message_edit_profiler = {}

            self.deconnection_lost_time = 0
            self.last_deconnection_time = None

            self.self_instruction_message = TimedInstructionMessage()
            self.plays_time = 0


            self.self_referee_delay_instruction_message = TimedInstructionMessage(withdraw=False)
            self.opponent_referee_delay_instruction_message = TimedInstructionMessage(withdraw=False)

            # archives
            self.opponent_instruction_message_time_archived = None
            self.self_instruction_message_time_archived = None

            self.self_referee_delay_instruction_message_time_archived = None
            self.opponent_referee_delay_instruction_message_time_archived = None

            self.plays_time_archived = None
            self.deconnection_lost_time_archived = None

            self.resigning = False

            self.operator = defaultdict(lambda:None)


        def is_opponent_message(self, message):
            """print('?')
            print(message.author)
            print(self.player.get_opponent())
            print(message.author)
            print(self.operator[self.player.get_opponent()])
            print(self.player.game.get_current_player())
            print('x:',self.player.player_number)"""

            #print(message.author , self.player.get_opponent() , '/',message.author , self.operator[self.player.get_opponent()] ,'/', self.player.game.get_current_player() , self.player.player_number)

            return message.author == self.player.get_opponent() or message.author == self.operator[self.player.get_opponent()] and self.player.game.get_current_player() != self.player.player_number

        def is_self_message(self, message):
            #print(message.author , self.user , message.author , self.operator[self.user] , self.player.game.get_current_player() , self.player.player_number)
            return message.author == self.user or message.author == self.operator[self.user] and self.player.game.get_current_player() == self.player.player_number



        async def on_ready(self) -> None:
            """Coroutine that is triggered when the model is ready."""

            self.connected = True

            """guild = self.get_guild(1402264352052084857)
                        channel = guild.get_channel(1402264352052084861)
                        perms = channel.permissions_for(guild.me)
                        print(f"Voir le salon : {perms.view_channel}")
                        print(f"Envoyer des messages : {perms.send_messages}")

                        bot_member = guild.get_member(1402331015396982827)
                        # Crée un nouvel overwrite à partir de l'ancien
                        overwrite = PermissionOverwrite()
                        overwrite.update(view_channel=True, read_messages=True)
                        print('?')
                        # Applique l'overwrite au bot
                        try:
                            #await channel.set_permissions(bot_member, overwrite=overwrite)
                            bot_role = discord.utils.get(guild.roles, name="Joueur 2 Olympiad")
                            if bot_role:
                                overwrite = discord.PermissionOverwrite()
                                overwrite.update(view_channel=True, read_messages=True)
                                await channel.set_permissions(bot_role, overwrite=overwrite)
                        except:
                            import traceback
                            #traceback.print_stack()
                            traceback.print_exc()"""
            #print('on ready...')

            for guild in self.guilds:
                if self.guild_id == guild.id:
                    print(f'We have logged in as {self.user} on {guild.name}')
                else:
                    print(f'The bot is registred in another guild: {guild.name} as {self.user}')


            #*#if self.last_channel_id:
            try:
                #*#self.last_channel = await self.fetch_channel(self.last_channel_id)
                await self.reprise_des_messages()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

            try:
                if self.last_deconnection_time is not None:
                    self.deconnection_lost_time += time() - self.last_deconnection_time
                    self.last_deconnection_time = None
            except Exception:
                import traceback
                traceback.print_exc()


        async def on_disconnect(self) -> None:
            """Coroutine that is triggered when the model is ready."""
            #import traceback
            #traceback.print_stack()
            if self.player.is_in_game():
                print('Déconnexion...')
            self.connected = False
            if self.last_deconnection_time is None:
                self.last_deconnection_time = time()


        async def on_resumed(self) -> None:
            """Coroutine that is triggered when the model is ready."""
            if self.player.is_in_game():
                print('Reconnexion...')

            self.connected = True
            await self.reprise_des_messages()

            try:
                if self.last_deconnection_time is not None:
                    self.deconnection_lost_time += time() - self.last_deconnection_time
                    self.last_deconnection_time = None
            except Exception:
                import traceback
                traceback.print_exc()

        async def reprise_des_messages(self, lock=True) -> None:

            if lock:
                async with self.message_lock:
                    await self.reprise_des_messages_corps()
            else:
                await self.reprise_des_messages_corps()



        async def reprise_des_messages_corps(self) -> None:

                if self.last_id and self.player.channel is not None:#*# and self.last_channel:

                    #print('to reprise_des_messages')
                    #*#async for missed_msg in self.last_channel.history(after=Object(id=self.last_id), oldest_first=True):
                    last = None
                    async for missed_msg in self.player.channel.history(after=Object(id=self.last_id), oldest_first=True):
                        #print('+', missed_msg.id not in self.processed_ids )
                        if missed_msg.id not in self.processed_ids:
                            #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content)

                            self.processed_ids.add(missed_msg.id)

                            self.last_id = missed_msg.id

                            res = await self.if_restart_special_in_game_process_then_todo_else_False(missed_msg)

                            if res is False:
                                traiter = await self.prepreprocessing(missed_msg)

                                if traiter:
                                    await self.preprocess(missed_msg)

                            elif res is True:
                                last = None
                                """already done"""
                            else:
                                assert res == 'self_instruction'
                                last = missed_msg

                    if last is not None:
                        await self.preprocess(last)



        async def on_close(self) -> None:
            """Coroutine that is triggered when the model is closed.""" #  N EXISTE PAS DANS LA LIBRAIRIE DISCORD ....
            #print('Cloture...')



        async def setup_hook(self) -> None:
            # Verify if the logging directory already exists in the current working directory
            """if 'bot_play_log' not in os.listdir('log'):
                os.mkdir('log/bot_play_log')"""

            await self.log_loading()

        async def log_loading(self, only_not_ended=True) -> None:
            try:

                if 'bot_play_log' not in await asyncio.to_thread(os.listdir, 'log'):
                    await asyncio.to_thread(os.mkdir, 'log/bot_play_log')

                last_game_file = ""
                oldest_game_file_mtime = 0
                # Search for a log that contains the last match of the player
                #for file in os.listdir('log/bot_play_log'):
                files = await asyncio.to_thread(os.listdir, 'log/bot_play_log')
                for file in files:
                    if not file.startswith("play_"+str(self.user.id)):
                        continue

                    filepath = os.path.join('log/bot_play_log', file)
                    current_mtime = await asyncio.to_thread(os.path.getmtime, filepath)
                    #current_mtime = os.path.getmtime(os.path.join('log/bot_play_log', file))
                    if current_mtime >= oldest_game_file_mtime:
                        oldest_game_file_mtime = current_mtime
                        last_game_file = os.path.join('log/bot_play_log', file)

                if last_game_file != "":
                    try:

                        async with aiofiles.open(last_game_file, mode="r") as f:
                            content = await f.read()
                            loading = json.loads(content)

                            # Check if the last match ended correctly or not
                            if not only_not_ended or not loading['ended']:
                                self.last_bot_play_log = loading

                    except JSONDecodeError as e:

                        file = last_game_file+ '.last'

                        if not os.path.exists(file):
                            raise e

                        async with aiofiles.open(file, mode="r") as f:
                            content = await f.read()
                            loading = json.loads(content)

                            # Check if the last match ended correctly or not
                            if not only_not_ended or not loading['ended']:
                                self.last_bot_play_log = loading


                        ### RECOVERY SECTION
                        ####################
                        ####################
            except JSONDecodeError as e:
                print('JSONDecodeError:',last_game_file)
                import traceback
                traceback.print_exc()
            except Exception as e:
                import traceback
                print('last_game_file:',last_game_file)
                traceback.print_exc()
                raise e

        async def prepreprocessing(self, message: Message) -> bool:

            if not isinstance(message.channel, DMChannel) and message.guild.id != self.guild_id and message.author != self.user and not message.author.bot:
                self.processed_ids.add(message.id)
                try:
                    await message.channel.send("Wrong guild: set the correct GUILD_ID in parameters.conf! ("+str(message.guild.id)+'!='+str(self.guild_id))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
                return False


            if message.guild.id != self.guild_id:
                return False

            if self.player.is_in_game() and message.channel != self.player.channel: ### ICI blocage channel que j ai ajouter ###
                return False

            if message.author == self.user:
                self.processed_ids.add(message.id)
                return False

            return True

        async def freeze(self):
            #print('hein ?', not self.freeze, self.freeze)
            if not self.freezed:
                #print('freeze!')
                self.freezed = True
                #self.freezed_messages = []
                self.freezed_message = None

        async def unfreeze(self):
            if self.freezed:
                #print('unfreeze!')
                """if self.freezed_message:
                    if self.freezed_message.id in self.processed_ids:
                        self.processed_ids.remove(self.freezed_message.id)
                    await self.on_message(self.freezed_message)
                print('...')"""
                self.freezed = False
                #print(len(self.freezed_messages))
                if self.freezed_message is not None:
                    #print('use fm:',self.freezed_message.content)
                    await self.process(self.freezed_message)
                """for message in self.freezed_messages:
                    await self.on_message(message)"""


        def freeze_continue(self, message):

            #print(' <<<<fc:',self.freezed,self.is_self_message(message))
            if self.freezed:
                if (self.is_self_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel) or EnumCompiledPattern.is_instruction_message(message.content) and message.author.id == self.player.referee_id and self.user not in message.mentions:
                    #self.freezed_messages = []
                    self.freezed_message = None
                    #print('unset fm')

                if EnumCompiledPattern.is_instruction_message(message.content) and message.author.id == self.player.referee_id and self.user in message.mentions:
                    #self.freezed_messages.append(message)
                    self.freezed_message = message
                    #print('set fm:',self.freezed_message.content)
                    return False
            return True

        async def on_message(self, message: Message) -> None:
            """coroutine that filters every message in order to keep only the one sent by other entities

            PARAMETERS
            ----------
            message : discord.Message
                message sent on the channels

            RETURNS
            -------
            None
            """
            #print(message.content,':',self.freezed, self.is_self_message(message), EnumCompiledPattern.is_instruction_message(message.content), self.user in message.mentions)
            """
            if self.freezed:
                if self.is_self_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:
                    self.freezed_messages = []

                if EnumCompiledPattern.is_instruction_message(message.content) and message.author.id == self.player.referee_id and self.user in message.mentions:
                    self.freezed_messages.append(message)
                    return


                if message.content.startswith("%") or message.content.startswith("!"):
                    self.processed_ids.add(message.id)
                    traiter = await self.prepreprocessing(message)
                    if traiter:
                        await self.process_commands(message)
                else:
                    self.freezed_messages.append(message)
                return"""
            #print('?')
            try:
                self.message_profiler[message.author.id] += 1
            except KeyError:
                self.message_profiler[message.author.id] = 1

            #*#self.last_channel = message.channel
            #*#self.last_channel_id = message.channel.id

            key = 'Operator:'
            if message.content[:len(key)] == key:
                self.operator[message.author]=message.mentions[0]
                self.bot_play_log['operator'] = {player.id:self.operator[player].id for player in self.operator}
                await self.save_bot_play_log()
                return


            continuer = await self.prepreprocessing(message)
            if not continuer:
                return

            ### RECUPERATION DES GAPS DE MESSAGE
            if message.id in self.processed_ids:
                return
            
            async with self.message_lock:
                #print('B')
                if self.last_id:
                    manquer = False
                    first_analyser = False
                    async for missed_msg in message.channel.history(after=Object(id=self.last_id), oldest_first=True):
                        if not first_analyser:
                            first_analyser = True
                            if missed_msg.id != message.id:
                                manquer = True
                        if missed_msg.id not in self.processed_ids:
                            if manquer and missed_msg.id != message.id :
                                """"""
                                #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                            elif manquer:
                                #print('M:', missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                                manquer = False
                            else:
                                """"""
                                #print('M:', missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                            self.processed_ids.add(missed_msg.id)
                            traiter = await self.prepreprocessing(missed_msg)
                            if traiter:
                                #print('traitement...')
                                await self.preprocess(missed_msg)

                # Mettre à jour le dernier ID vu
                self.last_id = message.id

                if message.id not in self.processed_ids:
                    #print('M delanchant:',message.author, message.content)
                    await self.preprocess(message)
                    self.processed_ids.add(message.id)

        async def preprocess(self, message: Message) -> None:

            if message.content.startswith("%") or message.content.startswith("!"):
                #print('process cmd')
                await self.process_commands(message)
            else:
                if self.freeze_continue(message):
                    await self.process(message)


        async def process(self, message: Message) -> None:
            """coroutine that processes the message that is in input.

            PARAMETERS
            ----------
            message : discord.Message
                message sent on the channels

            RETURNS
            -------
            None
            """
            #print('+process', self.player.is_in_game())
            # If the message states that the game has ended and the author is a model, then inform the player instance to end...
            if (message.content == "The game has ended" or 'has exceeded his allowed time' in message.content) and message.author.id == self.player.referee_id:
                try:
                    await self.player.end()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            # ...Otherwise, if the message is a winning message
            elif EnumCompiledPattern.PATTERN_WIN.value.fullmatch(message.content) and message.author.id == self.player.referee_id:

                # If the model is mentioned then it won...
                if self.user == message.mentions[0]:
                    self.bot_play_log['winner'] = True  # we set winner to true for logging
                    self.bot_play_log['ended'] = True
                # ...Otherwise it lost...
                else:
                    self.bot_play_log['winner'] = False  # we set winner to false for logging
                    self.bot_play_log['ended'] = True

                # When it is over, we dump all the actions that our player sent during the game
                await self.save_bot_play_log()

                #archive
                self.opponent_instruction_message_time_archived = self.opponent_instruction_message.time
                self.self_instruction_message_time_archived = self.self_instruction_message.time

                self.self_referee_delay_instruction_message_time_archived = self.self_referee_delay_instruction_message.time
                self.opponent_referee_delay_instruction_message_time_archived = self.opponent_referee_delay_instruction_message.time

                self.plays_time_archived = self.plays_time
                self.deconnection_lost_time_archived = self.deconnection_lost_time

                # ...And we reset the player instance
                self.opponent_instruction_message.reset()
                self.self_instruction_message.reset()

                self.self_referee_delay_instruction_message.reset()
                self.opponent_referee_delay_instruction_message.reset()

                self.plays_time = 0
                self.last_deconnection_time = None
                self.deconnection_lost_time = 0
                await self.player.reset()

            # ...Otherwise, if the message is not a winner announcement and the model is currently in game...
            elif self.player.is_in_game():

                await self.in_game_process(message)

            # ...Otherwise, it is a message that has a general purpose, that is a game is about to start.
            else:
                await self.general_process(message)



        def start_start_game(self, message):

            self.resigning = False

            mentions = message.mentions  # Retrieve the mentions

            #print('AA')
            #print(message.content)
            # Recover here some information contained in the starting message
            start_information = dict(EnumCompiledPattern.analyse_start_message(message.content))
            #print(start_information)
            #print('BB')
            #print(start_information)
            game_name = start_information['game name'].lower()
            #print('>',game_name, start_information['game name'], start_information)
            #print('CC')
            total_time = Time.string_to_Time(start_information['total time'])

            #print('DD')


            self.player.set_total_time(total_time)


            return game_name, mentions


        async def get_ordered_mentions(self, message):
            mention_pattern = r"<@!?(\d+)>"
            ordered_ids = re.findall(mention_pattern, message.content)
            # ordered_mentions = [message.guild.get_member(int(uid)) for uid in ordered_ids]
            return [await message.guild.fetch_member(int(uid)) for uid in ordered_ids]

        async def general_process(self, message: Message) -> None:
            """coroutine that apply a general process to the message that is in input.

            It will handle any game start procedure that is currently being processed.

            PARAMETERS
            ----------
            message : discord.Message
                message sent on the channels

            RETURNS
            -------
            None
            """

            def check_reaction(reaction: Reaction, user: User) -> bool:
                """function that checks if the user that reacted is a model, and if the reaction is on the same message as the one in argument of general_process(), and if the reaction is one of those two : 🟩, 🟥"""
                return user.id == message.author.id and reaction.message.id == message.id and (
                        reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

            # If it is about a game start that is involving the PlayerBot instance itself...
            if message.content.startswith("Game is starting !") and self.user in message.mentions and message.author.bot:

                game_name, mentions = self.start_start_game(message)

                ordered_mentions = await self.get_ordered_mentions(message)

                #print('Mise à jour du jeu')
                # Try to load the right game from the enumeration...
                try:

                    if self.async_update_game:
                        await self.player.update_game(game_name)
                    else:
                        self.player.update_game(game_name)

                # ...If an exception is raised print it, and do not set ready for game.
                except Exception as e:
                    import traceback
                    print(red(traceback.format_exc()))
                    #print(e)
                else:
                    #print('Ajout de la réaction')




                    await asyncio.sleep(1)  # Let some time for the RefereeBot to prepare to handle the reaction
                    try:
                        await message.add_reaction('👍')  # Inform the referee that the PlayerBot will play the game

                        reaction, _ = await self.wait_for('reaction_add', check=check_reaction)  # Wait for the Referee to validate or not the game start
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                    # If the emoji is 🟥, then the game is canceled...
                    if reaction.emoji.__str__() == '🟥':
                        return

                    # ...Otherwise, If the emoji is 🟩, then the game will start soon
                    if reaction.emoji.__str__() == '🟩':
                        await self.end_start_game(message=message, mentions=mentions, ordered_mentions=ordered_mentions)


        async def end_start_game(self, message, mentions, ordered_mentions):
            #print('+end_start_game')
            self.player.set_referee_id(message.author.id)

            self.opponent_instruction_message.time = self.player.total_time#game.time_per_player
            self.self_instruction_message.time = self.player.total_time#game.time_per_player
            self.self_referee_delay_instruction_message.message = None
            self.self_referee_delay_instruction_message.time = Time()
            self.opponent_referee_delay_instruction_message.message = None
            self.opponent_referee_delay_instruction_message.time = Time()

            self.plays_time = 0
            self.last_deconnection_time = None
            self.deconnection_lost_time = 0
            #print('>ordered players:',ordered_mentions)
            self.player.player_number = ordered_mentions.index(self.user)
            #print('#',self.player_number )

            mentions.remove(self.user)  # For the moment mentions has size 2, so if self.user is removed only the opponent will remain
            self.player.set_opponent(mentions[0])

            self.player.set_starting_time(Time.now())


            self.bot_play_log = {}
            self.bot_play_log['self_moves'] = []  # Initialises the structure that stores the moves
            self.bot_play_log['moves'] = []
            self.bot_play_log['game_name'] = self.player.game_name
            self.bot_play_log['ended'] = False
            self.bot_play_log['stopped'] = False
            self.bot_play_log['players'] = [self.user.id, self.player.opponent.id] # attention garder cette ordre sinon casse à un autre endroit
            self.bot_play_log['referee_id'] = self.player.referee_id

            self.bot_play_log['total_time'] = str(self.player.total_time)
            self.bot_play_log['starting_time'] = str(Time.now())

            self.bot_play_log['player_number'] = self.player.player_number

            #print('--')

            # Set the json file path
            self.set_json_file(self.user.id, message.guild.name, message.channel.name, self.date, self.player.starting_time.get_logformated())

            self.message_profiler = {}
            self.reaction_profiler = {}
            self.message_edit_profiler = {}

            await self.save_bot_play_log()

            self.player.set_channel(message.channel)

            try:
                user = await self.fetch_user(self.owner_id)
                if self.owner_id and self.owner_id >= 0:
                    await self.player.channel.send('Operator:' + str(user.mention))

                else:
                    await self.player.channel.send('Operator:None')
            except Exception as e:
                import traceback
                traceback.print_exc()

            self.player.start()
            #print('-end_start_game')


        def move_verifier(self, text):
            if self.player.game is not None:
                return  self.player.game.move_verifier.fullmatch(text)
            else:
                return  self.player.move_verifier.fullmatch(text)

        async def get_current_player(self):
            return self.player.game.get_current_player()
            """if self.player.game is not None:
                return self.player.game.get_current_player()
            else:
                return await self.player.get_current_player()"""

        async def resign(self):
            self.resigning = True

        async def in_game_process(self, message: Message) -> None:
            """coroutine that apply an "in-game" process to the message that is in input.

                It will handle any message that is associated with instructions, game moves, opponent moves, ...

                PARAMETERS
                ----------
                message : discord.Message
                    message sent on the channels

                RETURNS
                -------
                None
                """
            #print('traitement... in_game_process')
            #print(self.player.opponent.id)
            #print(message.author , self.player.get_opponent())
            #print(message.author == self.player.get_opponent() , self.move_verifier(message.content) , message.channel == self.player.channel)
            #print(EnumCompiledPattern.is_instruction_message(message.content) , message.author.id == self.player.referee_id , self.user in message.mentions , message.channel == self.player.channel)
            #print(self.user , message.mentions)
            #print()#message.author == self.player.get_opponent() and self.move_verifier(message.content)
            # If the message contains instruction, mentions the PlayerBot instance, is written in the same chanel as the start message, and was sent by a model...
            # (In short: if it is the PlayerBot's turn to play a move)

            #print(message.author , self.player.get_opponent() , self.move_verifier(message.content), message.content)
            #print(self.is_opponent_message(message),'<',message.content)
            if self.is_self_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:#^#
                if self.user == message.author:
                    print('CATA')
                assert self.user != message.author
                await self.in_game_reprocess(message)

            elif EnumCompiledPattern.is_instruction_message(message.content) and message.author.id == self.player.referee_id and self.user in message.mentions and message.channel == self.player.channel:
                #print('>>>', message.content,':', EnumCompiledPattern.is_instruction_message(message.content) , message.author.id == self.player.referee_id , self.user in message.mentions , message.channel == self.player.channel)
                if self.resigning:
                    self.resigning = False

                    move = 'resign'

                    try:
                        msg = await self.player.channel.send(move)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        # print('On recommence dans 5 secondes...')
                        try:
                            await sleep(5)
                            msg = await self.player.channel.send(move)
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            # print('On recommence dans 30 secondes...')
                            try:
                                await sleep(30)
                                msg = await self.player.channel.send(move)
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise e

                    self.player.game.terminate(winner=1 - await self.get_current_player())
                    self.bot_play_log['moves'].append(message.content)

                    await self.save_bot_play_log()

                    return


                current_player = await self.get_current_player()

                try:
                    if self.self_referee_delay_instruction_message.message is not None:
                        self.self_referee_delay_instruction_message.update_time(message)
                except Exception:
                    import traceback
                    traceback.print_exc()
                self.self_instruction_message.message = message

                # Retrieve the Time information from the instructions
                time_left = Time.string_to_Time(message.content)

                #print(len(self.player.jeu.historique))
                # Ask the AI to play a move
                try:
                    t=time()
                    if self.async_plays:
                        action = await self.player.plays(time_left, self.opponent_instruction_message.time)
                    else:
                        action = self.player.plays(time_left, self.opponent_instruction_message.time)
                    self.plays_time += time() - t

                except Exception as e:
                    await self.error_procedure(channel=message.channel, error_message="An error with the AI program occurred. The bot will shut down.", exception=e)

                # Translate the action into an equivalent expression readable by humans
                else:
                    try:
                        move = self.player.action_to_string(action)
                        #print('>', action,'>>', move)
                        assert move is not None, f"the move ({action}) translation into a string is None"
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error with the game action translator occurred. The bot will shut down.",
                                                   exception=e)

                    else:

                        #print('G1')
                        # Send the move on the chanel
                        try:
                            msg = await self.player.channel.send(move)
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            #print('On recommence dans 5 secondes...')
                            try:
                                await sleep(5)
                                msg = await self.player.channel.send(move)
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                #print('On recommence dans 30 secondes...')
                                try:
                                    await sleep(30)
                                    msg = await self.player.channel.send(move)
                                except Exception as e:
                                    import traceback
                                    traceback.print_exc()
                                    raise e

                        try:

                            self.self_instruction_message.update_time(message=msg)
                        except Exception:
                            import traceback
                            traceback.print_exc()
                        #print(f"Envoyé : {msg.id} {msg.content} {move}", datetime.now().strftime("%H:%M"))
                        #print('G2')



                        def check_reaction(reaction: Reaction, user: User) -> bool:
                            """function that checks if the user that reacted is a model, and if the reaction is on the same message as the one sent by the PlayerBot previously, and if the reaction is one of those two : 🟩, 🟥"""
                            #print('G3.5')
                            return user.id == self.player.referee_id and reaction.message.id == msg.id and (
                                    reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                        #print('G3')
                        # Wait for the Referee to validate/invalidate the move you just played


                        #print('Attente de la réaction (A):')

                        if not self.connected:
                            await sleep(5)
                            if not self.connected:
                                #print('NOT CONNECTED !!!')
                                await sleep(30)

                        ###
                        #while True:
                        for _ in range(20):

                            # await message.fetch()
                            try:
                                msg = await msg.channel.fetch_message(msg.id)
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise e

                            emoji = None
                            reacted = False
                            for R in msg.reactions:
                                if self.player.referee_id in [u.id async for u in R.users()]:
                                    if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                                        reacted = True
                                        emoji = R.emoji.__str__()
                                        break

                            if not reacted:
                                def check_reaction(reaction: Reaction, user: User) -> bool:
                                    """function that checks if the user that reacted is a model, and if the reaction is on the same message as the one sent by the PlayerBot previously, and if the reaction is one of those two : 🟩, 🟥"""
                                    #print('G3.5')
                                    return user.id == self.player.referee_id and reaction.message.id == msg.id and (
                                            reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                                # Wait for the Referee to validate/invalidate the move played by the opponent

                                #print('Attente de la réaction (A2):', msg.content, msg.reactions, datetime.now().strftime("%H:%M"))
                                try:
                                    reaction, usr = await self.wait_for('reaction_add', check=check_reaction, timeout=1)
                                    emoji = reaction.emoji.__str__()
                                    break
                                except asyncio.TimeoutError:
                                    """"""
                                except Exception as e:
                                    import traceback
                                    traceback.print_exc()
                                    raise e
                            else:
                                break


                        ###

                        if emoji == None:
                            print('Un problème est survenu: celui-ci est ignoré mais cela peut poser des problèmes. Vérifier que tous les coups ont été validés 🟩')
                            emoji = '🟩'
                        #reaction, usr = await self.wait_for('reaction_add', check=check_reaction)
                        #print('R:',emoji)

                        #print('G4')
                        # If the move is valid...
                        if emoji == '🟩':
                            self.player.last_actions_self.append(move)  # Record it internally
                            self.bot_play_log['self_moves'].append(move)  # Record it for json serialisation
                            self.bot_play_log['moves'].append(move)

                            await self.save_bot_play_log()

                        #print('G5')
                        # Otherwise, do nothing
                        if emoji == '🟥':
                            await self.player.invalid_action_processing()

                        #print('G6')

                        new_current_player = await self.get_current_player()
                        if current_player == new_current_player:
                            self.self_referee_delay_instruction_message.message = msg

                        else:
                            self.opponent_referee_delay_instruction_message.message = msg

                        #print('G7')

                        # The Referee will send new instructions in this case, so the execution will proceed again from the beginning

                #print(len(self.player.jeu.historique))
            # ...Otherwise if the instruction message is for the opponent
            elif EnumCompiledPattern.is_instruction_message(
                    message.content) and message.author.id == self.player.referee_id and self.player.get_opponent() in message.mentions and message.channel == self.player.channel:

                # Store the instruction message to compute opponent_time left
                self.opponent_instruction_message.message = message
                try:
                    if self.opponent_referee_delay_instruction_message.message is not None:
                        #t = self.opponent_referee_delay_instruction_message.time
                        self.opponent_referee_delay_instruction_message.update_time(message)
                        #print('up:', t, '->', self.opponent_referee_delay_instruction_message.time, '(', time(), self.opponent_referee_delay_instruction_message.message.created_at, message.created_at,')', datetime.now())
                    """else:
                        print('up:',self.opponent_referee_delay_instruction_message.time, '(', time(), message.created_at, ')', datetime.now())"""
                except Exception:
                    import traceback
                    traceback.print_exc()

            # ...Otherwise, if the message respects the syntax of a move, and was written in the right chanel by the opponent
            #^#elif message.author == self.player.get_opponent() and self.move_verifier(message.content) and message.channel == self.player.channel:
            elif self.is_opponent_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:
                #print('!!!!!! !!!!!!')

                current_player = await self.get_current_player()

                #print('Analyse de la réaction...')
                '''try:
                    await message.fetch()
                except:
                    import traceback
                    traceback.print_exc()

                reaction = None
                reacted = False
                for R in message.reactions:
                    if self.player.referee_id in [u.id async for u in R.users()]:
                        if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                            reacted = True
                            reaction = R
                            break

                if not reacted:
                    def check_reaction(reaction: Reaction, user: User) -> bool:
                        """Function that checks if the user that reacted is a model, and if the reaction is on the same message as the one in argument of in_game_process(), and if the reaction is one of those two : 🟩, 🟥"""
                        return user.id == self.player.referee_id and reaction.message.id == message.id and (
                                reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                    # Wait for the Referee to validate/invalidate the move played by the opponent

                    print('Attente de la réaction (B):', message.content, message.reactions)
                    reaction, usr = await self.wait_for('reaction_add', check=check_reaction)'''

                if not self.connected:
                    await sleep(5)
                    if not self.connected:
                        #print('NOT CONNECTED !!!')
                        await sleep(30)

                #while True:
                for _ in range(20):



                    emoji = None
                    reacted = False
                    for R in message.reactions:
                        if self.player.referee_id in [u.id async for u in R.users()]:
                            if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                                reacted = True
                                emoji = R.emoji.__str__()
                                break

                    if not reacted:
                        def check_reaction(reaction: Reaction, user: User) -> bool:
                            """Function that checks if the user that reacted is a model, and if the reaction is on the same message as the one in argument of in_game_process(), and if the reaction is one of those two : 🟩, 🟥"""
                            return user.id == self.player.referee_id and reaction.message.id == message.id and (
                                    reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                        # Wait for the Referee to validate/invalidate the move played by the opponent

                        #print('Attente de la réaction (B):', message.content,message.reactions)
                        try:
                            reaction, usr = await self.wait_for('reaction_add', check=check_reaction, timeout=1)
                            emoji = reaction.emoji.__str__()
                            break
                        except asyncio.TimeoutError:
                            """"""
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            raise e
                    else:
                        break

                    #await message.fetch()
                    try:
                        message = await message.channel.fetch_message(message.id)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                if emoji == None:
                    print('Un problème est survenu: celui-ci est ignoré mais cela peut poser des problèmes. Vérifier que tous les coups ont été validés 🟩')
                    emoji = '🟩'

                #print('R:', emoji)

                # Update opponent time left
                self.opponent_instruction_message.update_time(message=message)


                # If the move is valid...
                if emoji == '🟩':

                    self.player.last_actions_opponents.append(message.content)  # Record it internally

                    if message.content == 'resign':
                        self.player.game.terminate(winner=1 - await self.get_current_player())
                        self.bot_play_log['moves'].append(message.content)

                        await self.save_bot_play_log()

                        return

                    try:

                        #print('...')
                        action = self.player.string_to_action(message.content)
                        assert action is not None, f"the move ({message.content}) translation into an action is None"
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error with the game action translator occurred. As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                                   exception=e)
                    else:
                        try:
                            #print('?')

                            if self.async_opponent_plays:
                                await self.player.opponent_plays(action)  # Update the game instance associated with the player

                            else:
                                self.player.opponent_plays(action)  # Update the game instance associated with the player
                        except Exception as e:
                            await self.error_procedure(channel=message.channel,
                                                       error_message="An error while playing opponent's validated move occurred. As the action was validated by the referee, the problem is by player's side. The bot will shut down. Action:"+str(action),
                                                       exception=e)

                    self.bot_play_log['moves'].append(message.content)


                    await self.save_bot_play_log()

                # ...Otherwise, do nothing
                # The Referee will send new instructions for the opponent in this case, so it will trigger again in_game_process()

                try:
                    new_current_player = await self.get_current_player()
                    if current_player == new_current_player:
                        self.opponent_referee_delay_instruction_message.message = message

                    else:
                        self.self_referee_delay_instruction_message.message = message
                except Exception:
                    import traceback
                    traceback.print_exc()

            # chance move
            elif message.author.id == self.player.referee_id and self.move_verifier(message.content) and message.channel == self.player.channel:

                current_player = await self.get_current_player()

                try:
                    action = self.player.string_to_action(message.content)
                    assert action is not None, f"the move ({message.content}) translation into an action is None"
                except Exception as e:
                    await self.error_procedure(channel=message.channel,
                                               error_message="An error with the game action translator occurred. As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                               exception=e)
                else:
                    try:
                        if self.async_opponent_plays:
                            await self.player.opponent_plays(action)  # Update the game instance associated with the player

                        else:
                            self.player.opponent_plays(action)  # Update the game instance associated with the player
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error while playing opponent's validated move occurred. As the action was validated by the referee, the problem is by player's side. The bot will shut down. Action:"+str(action),
                                                   exception=e)

                self.bot_play_log['moves'].append(message.content)

                await self.save_bot_play_log()

                try:
                    new_current_player = await self.get_current_player()
                    if current_player == new_current_player:
                        self.opponent_referee_delay_instruction_message.message = message

                    else:
                        self.self_referee_delay_instruction_message.message = message
                except Exception:
                    import traceback
                    traceback.print_exc()

            # ...Otherwise, nothing has to be done
            else:
                pass



        async def if_restart_special_in_game_process_then_todo_else_False(self, message: Message):
            """coroutine that apply an "in-game" process to the message that is in input.

                It will handle any message that is associated with instructions, game moves, opponent moves, ...

                PARAMETERS
                ----------
                message : discord.Message
                    message sent on the channels

                RETURNS
                -------
                None
                """

            # If the message contains instruction, mentions the PlayerBot instance, is written in the same chanel as the start message, and was sent by a model...
            # (In short: if it is the PlayerBot's turn to play a move)
            if EnumCompiledPattern.is_instruction_message(message.content) and message.author.id == self.player.referee_id and self.user in message.mentions and message.channel == self.player.channel:

                return 'self_instruction'

            # ...Otherwise if the instruction message is for the opponent
            elif EnumCompiledPattern.is_instruction_message(
                    message.content) and message.author.id == self.player.referee_id and self.player.get_opponent() in message.mentions and message.channel == self.player.channel:

                return False

            # ...Otherwise, if the message respects the syntax of a move, and was written in the right chanel by the opponent
            #^#elif message.author == self.player.get_opponent() and self.move_verifier(message.content) and message.channel == self.player.channel:
            elif self.is_opponent_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:
                return False


            # chance move
            elif message.author.id == self.player.referee_id and self.move_verifier(message.content) and message.channel == self.player.channel:

                return False

            # ...Otherwise, if the message respects the syntax of a move, and was written in the right chanel by self
            #^#elif message.author == self.user and self.move_verifier(message.content) and message.channel == self.player.channel:
            elif self.is_self_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:

                current_player = await self.get_current_player()



                try:

                    self.self_instruction_message.update_time(message=message)
                except Exception:
                    import traceback
                    traceback.print_exc()
                #print(f"Envoyé : {message.id} {message.content} {message}", datetime.now().strftime("%H:%M"))
                #print('g2')

                def check_reaction(reaction: Reaction, user: User) -> bool:
                    """function that checks if the user that reacted is a model, and if the reaction is on the same message as the one sent by the PlayerBot previously, and if the reaction is one of those two : 🟩, 🟥"""
                    #print('g3.5')
                    return user.id == self.player.referee_id and reaction.message.id == msg.id and (
                            reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                #print('g3')
                # Wait for the Referee to validate/invalidate the move you just played

                #print('Attente de la réaction (Ag):')

                if not self.connected:
                    await sleep(5)
                    if not self.connected:
                        #print('NOT CONNECTED !!!')
                        await sleep(30)

                ###
                # while True:
                for _ in range(20):

                    # await message.fetch()
                    try:
                        msg = await message.channel.fetch_message(message.id)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                    emoji = None
                    reacted = False
                    for R in msg.reactions:
                        if self.player.referee_id in [u.id async for u in R.users()]:
                            if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                                reacted = True
                                emoji = R.emoji.__str__()
                                break

                    if not reacted:
                        def check_reaction(reaction: Reaction, user: User) -> bool:
                            """function that checks if the user that reacted is a model, and if the reaction is on the same message as the one sent by the PlayerBot previously, and if the reaction is one of those two : 🟩, 🟥"""
                            #print('g3.5')
                            return user.id == self.player.referee_id and reaction.message.id == msg.id and (
                                    reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                        # Wait for the Referee to validate/invalidate the move played by the opponent

                        #print('Attente de la réaction (A2g):', msg.content, msg.reactions, datetime.now().strftime("%H:%M"))
                        try:
                            reaction, usr = await self.wait_for('reaction_add', check=check_reaction, timeout=1)
                            emoji = reaction.emoji.__str__()
                            break
                        except asyncio.TimeoutError:
                            """"""
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            raise e
                    else:
                        break

                ###

                if emoji == None:
                    print('Un problème est survenu: celui-ci est ignoré mais cela peut poser des problèmes. Vérifier que tous les coups ont été validés 🟩')
                    emoji = '🟩'
                # reaction, usr = await self.wait_for('reaction_add', check=check_reaction)
                #print('R:', emoji)

                #print('g4')
                # If the move is valid...
                if emoji == '🟩':

                    move = message.content

                    if self.async_replays:
                        await self.player.replays(self.player.string_to_action(move))
                    else:
                        self.player.replays(self.player.string_to_action(move))

                    self.player.last_actions_self.append(move)  # Record it internally
                    self.bot_play_log['self_moves'].append(move)  # Record it for json serialisation
                    self.bot_play_log['moves'].append(move)

                    await self.save_bot_play_log()

                    new_current_player = await self.get_current_player()
                    if current_player == new_current_player:
                        self.self_referee_delay_instruction_message.message = message

                    else:
                        self.opponent_referee_delay_instruction_message.message = message

                #print('g5')
                # Otherwise, do nothing
                """if emoji == '🟥':
                    await self.player.invalid_action_processing()"""

                #print('g6')



                #print('G7')

                return True

            # ...Otherwise, nothing has to be done
            else:
                return False


        async def error_procedure(self, channel: TextChannel = None, error_message: str = None, exception: Exception = None):
            import traceback
            print(red(traceback.format_exc()))
            try:
                await channel.send(error_message)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e


            await self.save_bot_play_log()
            #print('>>> ARRET !')
            try:
                await self.close()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        def set_json_file(self, player_id: int, guild_name: str, channel_name: str, date: str, starting_time: str):
            self._json_file = os.path.join("log/bot_play_log", f"play_{player_id}_{'-'.join(guild_name.split())}_{'-'.join(channel_name.split())}_{date}_{starting_time}.json")

        def check_owner(self, author: User) -> bool:
            """Method that checks if the author in parameter is the current owner of the model through discord id equality test"""
            return self.owner_id == author.id

        def check_in_game(self) -> bool:
            """Method that checks if the player is currently in game"""
            return self.player.is_in_game()

        @property
        def bot_play_log(self):
            return self._bot_play_log

        @bot_play_log.setter
        def bot_play_log(self, value):
            self._bot_play_log = value

        @property
        def json_file(self):
            return self._json_file

        #@classmethod
        def check_guild(self):
            def predicate(ctx: commands.Context):
                return ctx.guild.id == self.guild_id#cls.ALLOWED_GUILD

            return commands.check(predicate)


        async def save_bot_play_log(self):
            """with open(self.json_file, 'w') as file:
                json.dump(self.bot_play_log, file)"""

            if self.json_file:

                if os.path.exists(self.json_file):
                    os.rename(self.json_file, self.json_file+'.last')


                self.bot_play_log['message_profiler'] = self.message_profiler
                self.bot_play_log['reaction_profiler'] = self.reaction_profiler
                self.bot_play_log['message_edit_profiler'] = self.message_edit_profiler

                self.bot_play_log['processed_ids'] = list(self.processed_ids)
                self.bot_play_log['last_id'] = self.last_id

                try:
                    if self.opponent_instruction_message.message is None:
                        self.bot_play_log['opponent_instruction_message'] = (None, str(self.opponent_instruction_message.time))
                    else:
                        self.bot_play_log['opponent_instruction_message'] = (self.opponent_instruction_message.message.id, str(self.opponent_instruction_message.time))

                    if self.self_instruction_message.message is None:
                        self.bot_play_log['self_instruction_message'] = (None, str(self.self_instruction_message.time))
                    else:
                        self.bot_play_log['self_instruction_message'] = (self.self_instruction_message.message.id, str(self.self_instruction_message.time))

                    if self.opponent_referee_delay_instruction_message.message is None:
                        self.bot_play_log['opponent_referee_delay_instruction_message'] = (None, str(self.opponent_referee_delay_instruction_message.time))
                    else:
                        self.bot_play_log['opponent_referee_delay_instruction_message'] = (self.opponent_referee_delay_instruction_message.message.id, str(self.opponent_referee_delay_instruction_message.time))

                    if self.self_referee_delay_instruction_message.message is None:
                        self.bot_play_log['self_referee_delay_instruction_message'] = (None, str(self.self_referee_delay_instruction_message.time))
                    else:
                        self.bot_play_log['self_referee_delay_instruction_message'] = (self.self_referee_delay_instruction_message.message.id, str(self.self_referee_delay_instruction_message.time))
                except Exception:
                    import traceback
                    traceback.print_exc()


                self.bot_play_log['deconnection_lost_time'] = self.deconnection_lost_time
                self.bot_play_log['plays_time'] = self.plays_time

                try:
                    async with aiofiles.open(self.json_file, 'w') as f:
                        await f.write(json.dumps(self.bot_play_log))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

        async def restart_game(self, message):
            game_name, mentions = self.start_start_game(message)

            ordered_mentions = await self.get_ordered_mentions(message)

            #print('Mise à jour du jeu')
            # Try to load the right game from the enumeration...
            try:

                if self.async_update_game:
                    await self.player.update_game(game_name)
                else:
                    self.player.update_game(game_name)

            # ...If an exception is raised print it, and do not set ready for game.
            except Exception as e:
                import traceback
                print(red(traceback.format_exc()))
                # print(e)
            else:
                await self.end_start_game(message=message, mentions=mentions, ordered_mentions=ordered_mentions)

        async def continue_match(self, context):
            #print('+continue_match')


            if not self.player.is_in_game():
                try:
                    self.bot_play_log = self.last_bot_play_log

                    self.player.player_number = self.bot_play_log['player_number']
                    for player, operator in self.bot_play_log['operator'].items():
                        self.operator[await self.fetch_user(player)] = await self.fetch_user(operator)


                    self.plays_time = self.bot_play_log['plays_time']
                    self.deconnection_lost_time = self.bot_play_log['deconnection_lost_time']

                    self.last_id = self.bot_play_log['last_id']
                    #*#self.last_channel = context.channel
                    #*#self.last_channel_id = context.channel.id
                    self.processed_ids = set(self.bot_play_log['processed_ids'])

                    self.player.set_total_time(Time.string_to_Time(self.bot_play_log['total_time']))
                    #print('CA')
                    try:

                        if self.async_update_game:
                            await self.player.update_game(self.bot_play_log['game_name'])
                        else:
                            self.player.update_game(self.bot_play_log['game_name'])

                    # ...If an exception is raised print it, and do not set ready for game.
                    except Exception as e:
                        import traceback
                        print(red(traceback.format_exc()))
                    #print('CB')
                    self.player.set_referee_id(self.bot_play_log['referee_id'])

                    self.player.set_opponent(await self.fetch_user(self.bot_play_log['players'][1]))

                    self.player.set_starting_time(Time.string_to_Time(self.bot_play_log['starting_time']))

                    self.set_json_file(self.user.id, context.guild.name, context.channel.name, self.date, self.player.starting_time.get_logformated())

                    self.player.set_channel(context.channel)

                    self.message_profiler = self.bot_play_log['message_profiler']
                    self.reaction_profiler = self.bot_play_log['reaction_profiler']
                    self.message_edit_profiler = self.bot_play_log['message_edit_profiler']

                    try:
                        if self.bot_play_log['opponent_instruction_message'][0] is None:
                            self.opponent_instruction_message.message = None

                        else:
                            self.opponent_instruction_message.message = await context.channel.fetch_message(self.bot_play_log['opponent_instruction_message'][0])
                        self.opponent_instruction_message.time = Time.string_to_Time(self.bot_play_log['opponent_instruction_message'][1])

                        if self.bot_play_log['self_instruction_message'][0] is None:
                            self.self_instruction_message.message = None

                        else:
                            self.self_instruction_message.message = await context.channel.fetch_message(self.bot_play_log['self_instruction_message'][0])
                        self.self_instruction_message.time = Time.string_to_Time(self.bot_play_log['self_instruction_message'][1])


                        if self.bot_play_log['opponent_referee_delay_instruction_message'][0] is None:
                            self.opponent_referee_delay_instruction_message.message = None

                        else:
                            self.opponent_referee_delay_instruction_message.message = await context.channel.fetch_message(self.bot_play_log['opponent_referee_delay_instruction_message'][0])
                        self.opponent_referee_delay_instruction_message.time = Time.string_to_Time(self.bot_play_log['opponent_referee_delay_instruction_message'][1])

                        if self.bot_play_log['self_referee_delay_instruction_message'][0] is None:
                            self.self_referee_delay_instruction_message.message = None

                        else:
                            self.self_referee_delay_instruction_message.message = await context.channel.fetch_message(self.bot_play_log['self_referee_delay_instruction_message'][0])
                        self.self_referee_delay_instruction_message.time = Time.string_to_Time(self.bot_play_log['self_referee_delay_instruction_message'][1])
                    except Exception:
                        import traceback
                        traceback.print_exc()

                    #print('CC')
                    self.player.start()
                    #print('CD')
                    for move in self.last_bot_play_log['moves']:
                        if self.async_replays:
                            await self.player.replays(self.player.string_to_action(move))
                        else:
                            self.player.replays(self.player.string_to_action(move))
                        #print('>',move)#, len(self.player.jeu.historique)

                    #print('op. id:',self.player.opponent.id, self.bot_play_log['players'])
                    #print('CE')
                    await self.reprise_des_messages(lock=False)
                    #print('CF')

                    #print('op. id:',self.player.opponent.id, self.bot_play_log['players'])
                except Exception as e:
                    import traceback
                    print(red(traceback.format_exc()))

            #print('-continue_match')



        async def continue_match_slow(self, context):
            #print('+continue_match')
            if not self.processed_ids:
                L = []
                cpt = 0
                ok = False
                #*#async for missed_msg in self.last_channel.history(limit=2500):
                async for missed_msg in self.player.channel.history(limit=3000):
                    if missed_msg.content.startswith("Game will start now !"):#'!s' ==
                        #rint('Detect : "Game will start now !"')
                        ok = True
                        await self.restart_game(missed_msg)
                        break
                    elif '!' != missed_msg.content[0] and '%' != missed_msg.content[0]:
                        cpt += 1
                        if cpt == 10000:
                            break
                        L.append(missed_msg)
                #print(len(L), ok)
                if ok:
                    L.reverse()
                    for missed_msg in L[:-1]:
                        #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content)

                        self.processed_ids.add(missed_msg.id)
                        traiter = await self.reprepreprocessing(missed_msg)
                        if traiter:
                            await self.in_game_reprocess(missed_msg)
                            self.last_id = missed_msg.id

                    if L:
                        missed_msg = L[-1]
                        #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content)

                        self.processed_ids.add(missed_msg.id)
                        traiter = await self.prepreprocessing(missed_msg)
                        if traiter:
                            await self.preprocess(missed_msg)
                            self.last_id = missed_msg.id
            else:
                try:
                    #print(self.processed_ids)#self.last_id, s
                    await context.send('The command can only be used immediately after launching the bot.')
                    #await self.player.channel.send('The command can only be used immediately after launching the bot.')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            #print('-continue_match')
        async def in_game_reprocess(self, message: Message) -> None:
            """coroutine that apply an "in-game" process to the message that is in input.

                It will handle any message that is associated with instructions, game moves, opponent moves, ...

                PARAMETERS
                ----------
                message : discord.Message
                    message sent on the channels

                RETURNS
                -------
                None
                """

            #print('+in_game_reprocess')
            # If the message contains instruction, mentions the PlayerBot instance, is written in the same chanel as the start message, and was sent by a model...
            # (In short: if it is the PlayerBot's turn to play a move)
            #print(message.author == self.user , self.move_verifier(message.content) , message.channel == self.player.channel)
            #^#if message.author == self.user and self.move_verifier(message.content) and message.channel == self.player.channel:
            if self.is_self_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:
                #print('Analyse de la réaction de soi...')

                if not self.connected:
                    await sleep(5)
                    if not self.connected:
                        #print('NOT CONNECTED !!!')
                        await sleep(30)

                # while True:
                for _ in range(20):



                    emoji = None
                    reacted = False
                    for R in message.reactions:
                        if self.player.referee_id in [u.id async for u in R.users()]:
                            if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                                reacted = True
                                emoji = R.emoji.__str__()
                                break

                    if not reacted:
                        def check_reaction(reaction: Reaction, user: User) -> bool:
                            """Function that checks if the user that reacted is a model, and if the reaction is on the same message as the one in argument of in_game_process(), and if the reaction is one of those two : 🟩, 🟥"""
                            return user.id == self.player.referee_id and reaction.message.id == message.id and (
                                    reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                        # Wait for the Referee to validate/invalidate the move played by the opponent

                        #print('Attente de la réaction (B):', message.content, message.reactions)
                        try:
                            reaction, usr = await self.wait_for('reaction_add', check=check_reaction, timeout=1)
                            emoji = reaction.emoji.__str__()
                            break
                        except asyncio.TimeoutError:
                            """"""
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            raise e
                    else:
                        break

                    # await message.fetch()
                    try:
                        message = await message.channel.fetch_message(message.id)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e



                if emoji == None:
                    print('Un problème est survenu: celui-ci est ignoré mais cela peut poser des problèmes. Vérifier que tous les coups ont été validés 🟩')
                    emoji = '🟩'

                #print('R:', emoji)

                # Update opponent time left
                #self.instruction_message.update_time(message=message)

                # If the move is valid...
                if emoji == '🟩':

                    self.player.last_actions_self.append(message.content)  # Record it internally

                    if message.content == 'resign':
                        self.player.game.terminate(winner=1 - await self.get_current_player())
                        self.bot_play_log['moves'].append(message.content)

                        await self.save_bot_play_log()

                        return

                    try:
                        action = self.player.string_to_action(message.content)
                        assert action is not None, f"the move ({message.content}) translation into an action is None"
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error with the game action translator occurred. The bot will shut down.",
                                                   exception=e)
                    else:
                        try:

                            if self.async_replays:
                                await self.player.replays(action)  # Update the game instance associated with the player

                            else:
                                self.player.replays(action)  # Update the game instance associated with the player
                        except Exception as e:
                            await self.error_procedure(channel=message.channel,
                                                       error_message="An error with the AI program occurred. The bot will shut down.",
                                                       exception=e)

                    self.bot_play_log['moves'].append(message.content)

                    await self.save_bot_play_log()

            # ...Otherwise if the instruction message is for the opponent
            elif EnumCompiledPattern.is_instruction_message(
                    message.content) and message.author.id == self.player.referee_id and self.player.get_opponent() in message.mentions and message.channel == self.player.channel:

                # Store the instruction message to compute opponent_time left
                self.opponent_instruction_message.message = message


            # ...Otherwise, if the message respects the syntax of a move, and was written in the right chanel by the opponent
            #^#elif message.author == self.player.get_opponent() and self.move_verifier(message.content) and message.channel == self.player.channel:
            elif self.is_opponent_message(message) and self.move_verifier(message.content) and message.channel == self.player.channel:

                #print('Analyse de la réaction de l adv...')
                #print('!!!!!!')
                if not self.connected:
                    await sleep(5)
                    if not self.connected:
                        #print('NOT CONNECTED !!!')
                        await sleep(30)

                # while True:
                for _ in range(20):



                    emoji = None
                    reacted = False
                    for R in message.reactions:
                        if self.player.referee_id in [u.id async for u in R.users()]:
                            if R.emoji.__str__() == '🟩' or R.emoji.__str__() == '🟥':
                                reacted = True
                                emoji = R.emoji.__str__()
                                break

                    if not reacted:
                        def check_reaction(reaction: Reaction, user: User) -> bool:
                            """Function that checks if the user that reacted is a model, and if the reaction is on the same message as the one in argument of in_game_process(), and if the reaction is one of those two : 🟩, 🟥"""
                            return user.id == self.player.referee_id and reaction.message.id == message.id and (
                                    reaction.emoji.__str__() == '🟩' or reaction.emoji.__str__() == '🟥')

                        # Wait for the Referee to validate/invalidate the move played by the opponent

                        #print('Attente de la réaction (B):', message.content, message.reactions)
                        try:
                            reaction, usr = await self.wait_for('reaction_add', check=check_reaction, timeout=1)
                            emoji = reaction.emoji.__str__()
                            break
                        except asyncio.TimeoutError:
                            """"""
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            raise e
                    else:
                        break

                    # await message.fetch()
                    try:
                        message = await message.channel.fetch_message(message.id)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e


                if emoji == None:
                    print('Un problème est survenu: celui-ci est ignoré mais cela peut poser des problèmes. Vérifier que tous les coups ont été validés 🟩')
                    emoji = '🟩'

                #print('R:', emoji)

                # Update opponent time left
                self.opponent_instruction_message.update_time(message=message)

                # If the move is valid...
                if emoji == '🟩':

                    self.player.last_actions_opponents.append(message.content)  # Record it internally

                    #print('... ...')
                    try:
                        action = self.player.string_to_action(message.content)
                        assert action is not None, f"the move ({message.content}) translation into an action is None"
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error with the game action translator occurred (1). As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                                   exception=e)
                    else:
                        try:

                            #print('??')
                            if self.async_opponent_plays:
                                await self.player.opponent_plays(action)  # Update the game instance associated with the player

                            else:
                                self.player.opponent_plays(action)  # Update the game instance associated with the player
                        except Exception as e:
                            await self.error_procedure(channel=message.channel,
                                                       error_message="An error while playing opponent's validated move occurred (2). As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                                       exception=e)

                    self.bot_play_log['moves'].append(message.content)

                    await self.save_bot_play_log()

                # ...Otherwise, do nothing
                # The Referee will send new instructions for the opponent in this case, so it will trigger again in_game_process()


            elif message.author.id == self.player.referee_id and self.move_verifier(message.content) and message.channel == self.player.channel:

                try:
                    action = self.player.string_to_action(message.content)
                    assert action is not None, f"the move ({message.content}) translation into an action is None"
                except Exception as e:
                    await self.error_procedure(channel=message.channel,
                                               error_message="An error with the game action translator occurred (3). As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                               exception=e)
                else:
                    try:
                        if self.async_opponent_plays:
                            await self.player.opponent_plays(action)  # Update the game instance associated with the player

                        else:
                            self.player.opponent_plays(action)  # Update the game instance associated with the player
                    except Exception as e:
                        await self.error_procedure(channel=message.channel,
                                                   error_message="An error while playing opponent's validated move occurred (4). As the action was validated by the referee, the problem is by player's side. The bot will shut down.",
                                                   exception=e)

                self.bot_play_log['moves'].append(message.content)

                await self.save_bot_play_log()


            # ...Otherwise, nothing has to be done
            else:
                pass



        async def reprepreprocessing(self, message: Message) -> bool:
            #print('+reprepreprocessing')


            if message.content.startswith("%") or message.content.startswith("!"):
                return False

            #print('-reprepreprocessing')
            return True

        async def profile(self, context):
            #print('+profile')
            try:

                txt = 'Since the start of the last match:\n'
                for id, count in self.message_profiler.items():
                    user = await self.fetch_user(id)
                    txt+=str(user)+': '+str(count)+' messages\n'
                if len(self.reaction_profiler.items()):
                    txt+='\n'
                    for id, count in self.reaction_profiler.items():
                        user = await self.fetch_user(id)
                        txt+=str(user)+': '+str(count)+' reactions\n'
                if len(self.message_edit_profiler):
                    txt+='\n'
                    for id, count in self.message_edit_profiler.items():
                        user = await self.fetch_user(id)
                        txt+=str(user)+': '+str(count)+' edit\n'

                total_time = Time.string_to_Time(self.bot_play_log['total_time']).to_seconds()

                if self.bot_play_log['ended']:
                    txt += '\n'
                    txt += 'Deconnection lost time (except program crash): ' + str(Time.seconds_to_Time(int(self.deconnection_lost_time_archived))) + '\n\n'

                    txt += 'Opponent time left (local compute): ' + str(self.opponent_instruction_message_time_archived) + '\n'
                    txt += 'Self time left (local compute): ' + str(self.self_instruction_message_time_archived) + '\n'
                    txt += 'Total Time - Self Search Time: ' + str(Time.seconds_to_Time(int(total_time - self.plays_time_archived))) + '\n\n'

                    txt += 'Opponent referee delay (local compute): ' + str(self.opponent_referee_delay_instruction_message_time_archived) + '\n'
                    txt += 'Self referee delay (local compute): ' + str(self.self_referee_delay_instruction_message_time_archived) + '\n'
                    try:
                        txt += 'Ratio referee delay (local compute): ' + str(
                            self.opponent_referee_delay_instruction_message_time_archived.to_milliseconds() / self.self_referee_delay_instruction_message_time_archived.to_milliseconds()) + '\n\n'
                    except Exception:
                        txt += '\n'

                else:
                    txt+='\n'
                    txt += 'Deconnection lost time (except program crash): '+ str(Time.seconds_to_Time(int(self.deconnection_lost_time)))+ '\n\n'

                    txt += 'Opponent time left (local compute): ' + str(self.opponent_instruction_message.time) + '\n'
                    txt += 'Self time left (local compute): ' + str(self.self_instruction_message.time) + '\n'
                    txt += 'Total Time - Self Search Time: ' + str(Time.seconds_to_Time(int(total_time - self.plays_time))) + '\n\n'

                    txt += 'Opponent referee delay (local compute): ' + str(self.opponent_referee_delay_instruction_message.time) + '\n'
                    txt += 'Self referee delay (local compute): ' + str(self.self_referee_delay_instruction_message.time) + '\n'
                    try:
                        txt += f'Ratio referee delay (local compute): {self.opponent_referee_delay_instruction_message.time.to_milliseconds()/self.self_referee_delay_instruction_message.time.to_milliseconds():.2f}\n\n'
                    except Exception:
                        txt += '\n'


                await context.channel.send(txt)

            except Exception as e:
                import traceback
                print(red(traceback.format_exc()))
            #print('-profile')

        async def on_reaction_add(self, reaction, user):
            try:
                self.reaction_profiler[user.id] += 1
            except KeyError:
                self.reaction_profiler[user.id] = 1


        async def on_message_edit(self, before, after):
            try:
                self.message_edit_profiler[after.author.id] += 1
            except KeyError:
                self.message_edit_profiler[after.author.id] = 1
