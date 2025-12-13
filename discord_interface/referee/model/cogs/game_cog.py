import os
import pickle
from asyncio import CancelledError
from collections import defaultdict
from datetime import datetime
from json import JSONDecodeError
from random import random, choice
from time import time, perf_counter

import aiofiles
from discord import Colour, DeletedReferencedMessage, DMChannel, Embed, Message, Reaction, User, Object, DiscordServerError, TextChannel
from discord.ext import commands, tasks
from discord.ext.commands import Context

import discord

from discord_interface.games.instances.free_game import FreeGame
from discord_interface.utils.mytime import Time, NegativeError
from discord_interface.utils.pattern_enum import EnumCompiledPattern
from discord_interface.games.games_enum import EnumGames
from discord_interface.referee.model.referee_bot import RefereeBot
from discord_interface.utils.mymessage import InstructionMessage
import asyncio, json
from typing import List

import re

# Metadata
__author__ = "Oscar PERIANAYAGASSAMY & Quentin Cohen-Solal"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

from discord_interface.utils.terminal import red, orange

if __name__ != "__main__":

    class GameCog(commands.Cog):
        """Cog containing commands, and listeners related to game playing.

        ATTRIBUTES
        ----------
        bot : RefereeBot
            Bot instance that is defined in RefereeBot.
        time_elapsed : mytime.Time
            Attribute used in chronometer task to keep track of the time elapsed since the task started.
        limit : mytime.time
            Attribute used in chronometer task to avoid letting it run for a too long period of time.
        ended : bool
            Boolean attribute that indicates whether the chronometer task is finished or not.
        error : bool
            Boolean attribute that indicates whether the start_timer task has failed or not. It fails whenever the time limit is reached.
        instruction_message : InstructionMessage
            Last instruction message sent on the game channel. Used to edit the message along with the chronometer run.
        """

        def __init__(self, refbot: RefereeBot) -> None:
            self.bot = refbot
            self.time_elapsed = Time()
            self.limit = None
            self.ended = False
            self.error = False
            self.instruction_message = None

            self.need_instruction_message = False

            self.last_id = None
            self.processed_ids = set()
            #*#self.last_channel = None
            #*#self.last_channel_id = None
            self.message_lock = asyncio.Lock()
            self.message_lock_beginend = asyncio.Lock()

            self.canceled_chronometer_number = 0
            self.chronometer_bugger= False
            self.timeout_player= None

            self.game_in_progress = False

            self.last_players = None


        async def prepreprocessing(self, message: Message) -> bool:
            #print('+prepreprocessing')
            if not isinstance(message.channel, DMChannel) and message.guild.id != self.bot.guild_id and message.author.id != self.bot.user.id and not message.author.bot:
                self.processed_ids.add(message.id)
                try:
                    await message.channel.send("Wrong guild: set the correct GUILD_ID in parameters.conf!")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
                return False

            if message.guild.id != self.bot.guild_id or message.channel.id != self.bot.channel_id:
                return False

            #print(message.content, self.bot.referee.is_in_game() , message.channel ,self.bot.referee.channel)
            if self.bot.referee.is_in_game() and message.channel != self.bot.referee.channel: ### ICI blocage channel que j ai ajouter ###
                return False

            #async with self.message_lock_beginend:
                # If the message was sent by a model and indicates that one player exceeded his allowed time...
            #print('£1', EnumCompiledPattern.PATTERN_EXCEED_TIME.value.fullmatch(message.content), message.author == self.bot.user)
            if EnumCompiledPattern.PATTERN_EXCEED_TIME.value.fullmatch(message.content) and message.author == self.bot.user:

                #print('£2',not self.bot.referee.in_end_game, not self.bot.referee.game.ended())

                # If the game has not ended yet, apply a particular treatment if time limit wes exceeded...
                if not self.bot.referee.in_end_game and not self.bot.referee.game.ended():
                    await self.time_exceed_treatment(message)

            if message.author == self.bot.user and self.bot.referee.game.get_current_player() == -1:
                return True

            # No need to treat our model's own messages except in the previous case
            if message.author == self.bot.user:
                self.processed_ids.add(message.id)
                return False

            #print('-prepreprocessing')
            return True

        @commands.Cog.listener()
        async def on_message(self, message: Message) -> None:
            """Listener that provides an in-game treatment for messages.

            PARAMETERS
            ----------
            message : discord.Message
                The most recent message sent in the channel.
            """


            #print("Message reçu :", message.content)
            #print("Mentions utilisateurs :", [user.name for user in message.mentions])
            #print('on_message:',self.ended,self.timeout_player, self.game_in_progress, self.last_id)


            #print('+on_message')

            try:

                key = 'Operator:'
                if message.content[:len(key)] == key:
                    self.bot.operator[message.author]=message.mentions[0]
                    return


                async with self.message_lock:

                    continuer = await self.prepreprocessing(message)

                    if not continuer:
                        return

                    if hasattr(self, 'messages_history'):#message.content != '!continue' and message.content != '!c':
                        self.messages_history.append((str(message.author), str(message.content), int(message.id), str(datetime.now())))

                    ### RECUPERATION DES GAPS DE MESSAGE
                    if message.id in self.processed_ids:
                        return

                    #*#self.last_channel = message.channel
                    #*#self.last_channel_id = message.channel.id

                    if self.last_id:
                        manquer = False
                        first_analyser = False
                        async for missed_msg in message.channel.history(after=Object(id=self.last_id), oldest_first=True):
                            if not first_analyser:
                                first_analyser = True
                                if missed_msg.id != message.id:
                                    manquer = True
                            if missed_msg.id not in self.processed_ids:
                                if manquer and missed_msg.id != message.id:
                                    """"""
                                    #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                                elif manquer:
                                    #print('M:',missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                                    manquer = False
                                else:
                                    """"""
                                    #print('M:', missed_msg.author, missed_msg.content, datetime.now().strftime("%H:%M"))
                                self.processed_ids.add(missed_msg.id)
                                traiter = await self.prepreprocessing(missed_msg)
                                if traiter:
                                    await self.preprocess(missed_msg)



                    # Mettre à jour le dernier ID vu
                    self.last_id = message.id

                    if message.id not in self.processed_ids:
                        #print('M:',message.author, message.content)
                        await self.preprocess(message)
                        self.processed_ids.add(message.id)

                    #if not self.bot.referee.game.ended() and not self.endedand not self.chronometer.is_running():
                    if self.chronometer_bugger and not self.chronometer.is_running() and not self.bot.referee.game.ended() and not self.ended:
                        #print('chronometer.start A')
                        self.chronometer.start()

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e


            #print('-on_message')


        async def preprocess(self, message: Message, reprise=False) -> None:

            #print('+preprocess')


            # If the referee is in-game and the message contains a unique argument, and was sent by a player in the dedicated channel
            if self.bot.referee.game.move_verifier.fullmatch(
                    message.content) and self.bot.referee.in_game and not self.bot.referee.paused and not self.bot.referee.in_end_game and (message.author in self.bot.referee.players or message.author in [self.bot.operator[player] for player in self.bot.referee.players] or message.author == self.bot.user) and message.channel == self.bot.referee.channel:

                #print('>>',message.content)
                # Retrieve the move
                move = message.content

                # If one of the player plays outside his turn
                if message.author != self.bot.referee.current_turn and message.author != self.bot.operator[self.bot.referee.current_turn]:
                    try:
                        await message.channel.send(f'not your turn {message.author.mention}')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                    return

                await self.ingame_treatment(move, message, reprise=reprise)
            #print('-preprocess')

        @commands.Cog.listener()
        async def on_resumed(self) -> None:
            """Coroutine that is triggered when the model is ready."""

            try:
                if self.bot.referee.is_in_game():
                    print('Reconnexion...')

                self.connected = True
                await self.reprise_des_messages()

                #&#
                if not self.chronometer.is_running() and self.game_in_progress:
                    #print('chronometer.start B')
                    self.chronometer.start()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.Cog.listener()
        async def on_ready(self) -> None:
            """Coroutine that is triggered when the model is ready and that sets up json file path and logging utils."""
            #print('on ready (game)...')

            self.connected = True

            #*#if self.last_channel_id:
            #*#
            """try:
                self.last_channel = await self.bot.fetch_channel(self.last_channel_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e"""

            try:
                if self.bot.referee.channel and self.last_id:#*#
                    await self.reprise_des_messages()
                    #&#
                    if not self.chronometer.is_running() and self.game_in_progress:
                        #print('chronometer.start C')
                        self.chronometer.start()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.Cog.listener()
        async def on_disconnect(self) -> None:
            """Coroutine that is triggered when the model is ready."""
            #import traceback
            #traceback.print_stack()
            if self.bot.referee.is_in_game():
                print('Déconnexion...')
            self.connected = False

        async def reprise_des_messages(self, continue_reprise=False) -> None:

            #print('reprise_des_messages', self.last_id )#, self.last_channel
            if self.last_id and self.bot.referee.channel:# *# and self.last_channel:
                #*#async for missed_msg in self.last_channel.history(after=Object(id=self.last_id), oldest_first=True):
                async for missed_msg in self.bot.referee.channel.history(after=Object(id=self.last_id), oldest_first=True):

                    if missed_msg.id not in self.processed_ids:
                        #print('⚠ M manqué rattrapé 🔄:', missed_msg.author, missed_msg.content)

                        self.processed_ids.add(missed_msg.id)
                        traiter = await self.prepreprocessing(missed_msg)
                        if traiter:
                            #print('<channel:', self.bot.referee.channel)
                            await self.preprocess(missed_msg, reprise=continue_reprise)
                            #print('>channel:', self.bot.referee.channel)

        @commands.Cog.listener()
        async def on_message_delete(self, message: Message):
            """Listener whose role is to flip the value of instruction_message deletion state.

            PARAMETER
            ---------
            message: discord.Message
                The message that was deleted
            """
            try:
                if message.guild.id != self.bot.guild_id or message.channel.id != self.bot.channel_id:
                    return False

                if self.bot.referee.is_in_game() and message.channel != self.bot.referee.channel:  ### ICI blocage channel que j ai ajouter ###
                    return False

                if message.author == self.bot.user:
                    if self.instruction_message is None:
                        return

                    if message.id == self.instruction_message.message.id:
                        #print("ok")
                        self.instruction_message.delete()
                else:
                    if self.bot.referee.game.move_verifier.fullmatch(message.content):
                        await message.channel.send(
                            "Removing game action messages is prohibited! If the removing causes system instability, the opponent will be considered the winner of the match.\nSuppressed message: " + message.content)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        async def time_exceed_treatment(self, message: Message) -> None:
            """Coroutine that apply a treatment when a player exceeded his allowed time

            PARAMETERS
            ----------
            message : discord.Message
                The last message sent on the game channel.

            RETURNS
            -------
            None
            """

            #print('+time_exceed_treatment')


            try:
                # The referee instance enters in 'end-game' mode
                self.bot.referee.enters_end_game()

                self.game_in_progress = False
                self.last_id = None

                # Reset the instruction message
                if self.instruction_message is not None:
                    self.instruction_message.reset()

                # In this time mode, a time exceed leads to defeat
                #self.bot.referee.wins(self.bot.referee.opposite(self.bot.referee.current_turn))
                self.bot.referee.wins(self.bot.referee.opposite(self.timeout_player))

                # Informational messages for the players in the game channel
                try:
                    self.alarme()
                    self.alarme()
                    #print('The game has ended:', datetime.now().strftime("%H:%M"))
                    await self.bot.referee.channel.send(f'The game has ended')
                    await self.bot.referee.channel.send(
                        f'{self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].mention} won')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
                # Storing meaningful information into the json serializable structure
                self.bot.bot_ref_log["winner"] = self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].id
                self.bot.bot_ref_log["loser"] = self.bot.referee.opposite(self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner]).id
                self.bot.bot_ref_log["ended"] = True

                # Reset the referee instance to make it ready for a new game
                self.bot.referee.reset_end_game()

                # Serialize the stored data
                await self.save_bot_ref_log()
            except Exception:
                import traceback
                traceback.print_exc()
                raise e


            #print('-time_exceed_treatment')

        async def ingame_treatment(self, move: str, message: Message, reprise=False) -> None:
            """Coroutine that apply a treatment when managing a message in 'time_per_move' time mode

            PARAMETERS
            ----------
            move : str
                The move that was played (equivalent to message.content here)
            message : discord.Message
                The last message sent on the game channel.

            RETURNS
            -------
            None
            """
            #print('X1')
            # Stop the chronometer
            #&#self.cog_unload_chronometer()
            #print('+ingame_treatment:',move)


            #print('+ingame_treatment')

            current_player = self.bot.referee.game.get_current_player()

            ok = False

            if move == 'resign':

                #print('-1')
                self.bot.referee.game.terminate(winner=1-self.bot.referee.game.get_current_player())

                await message.add_reaction('🟩')

                assert self.bot.referee.game.ended()

                #print('-2')
                #print('Traitement fin de partie classique')
                self.bot.referee.enters_end_game()

                self.game_in_progress = False
                self.last_id = None

                # Reset the instruction message
                self.instruction_message.reset()

                #print('-3')
                try:
                    # Display end game information
                    #print('The game has ended:', datetime.now().strftime("%H:%M"))
                    await message.channel.send(f'The game has ended')

                    #print('-3a')
                    if self.bot.referee.game.winner in [0.5, '0.5', 'draw','nul']:
                        #print('-3b1')
                        await message.channel.send('The game is a draw')
                    else:
                        #print('-3b2')
                        await message.channel.send(f'{self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].mention} won')

                        #print('-3b2>')
                    #print('-3c')
                    self.alarme()
                    self.alarme()

                    #print('-3d')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e


                #print('-4')
                # Store the winner/loser
                if self.bot.referee.game.winner in [0.5, '0.5', 'draw','nul']:
                    self.bot.bot_ref_log["winner"] = 'draw'
                    self.bot.bot_ref_log["loser"] = 'draw'
                else:
                    self.bot.bot_ref_log["winner"] = self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].id
                    self.bot.bot_ref_log["loser"] = self.bot.referee.opposite(self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner]).id
                self.bot.bot_ref_log["ended"] = True

                # Reset the referee instance keeping some property as they are at the moment (display_activated, game.time_per_player, ...)
                self.bot.referee.reset_end_game()

                #print('-5')
                #print('chronometer_stop X')
                self.chronometer_stop()

                #print('-6')
                # Dump into the json file the data stored through the game
                await self.save_bot_ref_log()
                #print('-7')
                return


            if message.author == self.bot.user:

                #print('>>>', message.content)

                if reprise:

                    if self.bot.referee.game.get_current_player() == -1:

                        move = message.content

                        action = self.bot.referee.game.string_to_action(move)

                        if action in self.bot.referee.game.valid_actions():
                            #print('???')
                            self.bot.referee.game.plays(action)

                            await message.add_reaction('🟩')

                            #print('>>>>',message.content)

                            ok = True

            else:

                #print('channel a:', self.bot.referee.channel)

                # Try to translate into an AI-understable language the move, and play it on the referee game instance
                try:
                    """print(move, self.bot.referee.game.string_to_action(move))
                    print(self.bot.referee.game.__class__.__name__)
                    print(self.bot.referee.game.jeu.__class__.__name__)
                    print(self.bot.referee.game.jeu.historique)
                    print(self.bot.referee.game.jeu.coupsLicites())
                    print(self.bot.referee.game.valid_actions())
                    print(self.bot.referee.game.historique_partial_moves)
                    print(self.bot.referee.game.partial_moves)"""

                    action = self.bot.referee.game.string_to_action(move)
                    assert action is not None, f"The move ({move}) translation is a None object"
                    if not self.bot.referee.game.actions_validation_enabled or action in self.bot.referee.game.valid_actions():
                        self.bot.referee.game.plays(action)
                    else:
                        raise Exception("Illegal action in that state :'"+str(action)+"'"+str(self.bot.referee.game.valid_actions()))

                # ...If the move is translated into a None object it means there is an issue in the translation functions
                except AssertionError as e:
                    import traceback
                    print(red(traceback.format_exc()))
                    await self.bot.referee.channel.send("An error occured. Please check the translation function. The referee will shut down.")



                # ...If the action is refused by the game engine...
                except Exception as e:
                    import traceback
                    print(red(traceback.format_exc()))

                    # await asyncio.sleep(1)
                    try:
                        await message.add_reaction('🟥')
                        await message.channel.send('invalid move...')
                        try:
                            print('>> invalid move: ',action, move)
                        except:
                            """ """
                        #print('>>',self.bot.referee.game.valid_actions())
                        #print(list(map(lambda a: self.bot.referee.game.action_to_string(a), self.bot.referee.game.valid_actions())))
                        #if hasattr(self.bot.referee.game, 'jeu'):
                        #    print('H =',[(a,b) for a,b, *_ in self.bot.referee.game.jeu.historique])
                        # Displaying valid moves

                        txt = f'valid moves : {list(map(lambda a: self.bot.referee.game.action_to_string(a), self.bot.referee.game.valid_actions()))}'

                        if len(txt) < 1000:
                            await message.channel.send(txt)
                        else:
                            await message.channel.send(txt[:1000]+'...')

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                # ...Otherwise the action is validated
                else:

                    #print('channel b:', self.bot.referee.channel)
                    #print('x1')
                    # Store the move in memory
                    if current_player != -1:
                        self.bot.bot_ref_log[self.bot.referee.current_turn.id].append(move)
                    #print('*',move)
                    self.bot.bot_ref_log["moves"].append(move)

                    #print('x2')
                    # Serialize directly after
                    await self.save_bot_ref_log()

                    #print('x3')
                    # await asyncio.sleep(1)
                    try:
                        await message.add_reaction('🟩')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                    # Reset the time remaining for this player to its initial value if the time_per_move mode is activated

                    ok = True

            #print('channel c:', self.bot.referee.channel)

            if current_player != -1:
                #print('x4')
                # Update time fields and retrieve the current player
                if self.instruction_message is not None and self.instruction_message.ended_time is None:#-# if self.instruction_message.ended_time is None:
                    self.instruction_message.ended_time = time()

                    player = self.bot.referee.update_turn(self.instruction_message.message, message)
                else:
                    # cas où je joueur jouerait trop vite (avant que le nouveau message d'instruction ne soit créé : si c'est pas None, c'est que le message d'instruction actuel correspond à un autre message (précédent);
                    player = self.bot.referee.current_turn
                #print('x5')

            #print('channel d:', self.bot.referee.channel)
            if ok:
                if current_player != -1:
                    if self.bot.referee.time_per_move_activated:
                        self.bot.referee.time_remaining_player[player] = self.bot.referee.game.time_per_player
                    #print('A')
                    if not reprise:
                        await self.chance_move_treatment()

                # Update turn
                self.bot.referee.next_turn()


            # If the game has ended...
            if self.bot.referee.game.ended() or self.bot.referee.in_end_game:
                #print('Traitement fin de partie classique')
                self.bot.referee.enters_end_game()

                self.game_in_progress = False
                self.last_id = None

                # Reset the instruction message
                self.instruction_message.reset()

                try:
                    # Display end game information
                    #print('The game has ended:', datetime.now().strftime("%H:%M"))
                    await message.channel.send(f'The game has ended')
                    if self.bot.referee.game.winner in [0.5, '0.5', 'draw', 'nul']:
                        await message.channel.send('The game is a draw')
                    else:
                        await message.channel.send(f'{self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].mention} won')

                    self.alarme()
                    self.alarme()

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                # Store the winner/loser
                if self.bot.referee.game.winner in [0.5, '0.5', 'draw', 'nul']:
                    self.bot.bot_ref_log["winner"] = 'draw'
                    self.bot.bot_ref_log["loser"] = 'draw'
                else:
                    self.bot.bot_ref_log["winner"] = self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner].id
                    self.bot.bot_ref_log["loser"] = self.bot.referee.opposite(self.bot.referee.player_anti_correspondence[self.bot.referee.game.winner]).id
                self.bot.bot_ref_log["ended"] = True

                # Reset the referee instance keeping some property as they are at the moment (display_activated, game.time_per_player, ...)
                self.bot.referee.reset_end_game()
                #print('chronometer_stop Y')
                self.chronometer_stop()

                # Dump into the json file the data stored through the game
                await self.save_bot_ref_log()

                return

            #print('channel f:', self.bot.referee.channel)
            #print('X4')
            # Displaying the game state if the option is currently activated and if the game supports it
            await self.bot.referee.display_game(message.channel)

            if self.bot.referee.current_turn is not None and self.bot.referee.current_turn != self.bot.user:
                if not reprise:
                    #print('channel g:', self.bot.referee.channel)
                    #print('X5')
                    # Starting the chronometer
                    #print('$$$',message.content)
                    self.limit = self.bot.referee.time_remaining_player[self.bot.referee.current_turn]

                    #print('X6')
                    # Displaying next turn info

                    #&#self.instruction_message = InstructionMessage(message=await self.bot.referee.display_next_turn_info(self.bot.referee.channel))
                    self.instruction_message = None
                    if self.timeout_player is None:
                        #print('ouici:',self.bot.referee.current_turn.mention)
                        self.instruction_message = InstructionMessage(message=await self.display_next_turn_info(self.bot.referee.channel))
                else:
                    self.need_instruction_message= True


            #print('channel h:', self.bot.referee.channel)
            #print('X7')
            #&#
            """
            try:
                if self.chronometer.is_running():
                    self.chronometer.restart()
                else:
                    await self.chronometer.start()
            except:
                import traceback
                print(red(traceback.format_exc()))
            #"""
            """if self.chronometer.is_running():
                self.chronometer.stop()
            self.chronometer.start()"""


            #print('X8')
            await self.save_backup()
            #print('X9')
            #print('channel i:', self.bot.referee.channel)


            #print('-ingame_treatment')

        def selection_uniforme(self, coups, probas):
            s = 0

            somme = sum([probas[cp] for cp in coups])

            r = random() * somme

            for cp in coups:
                s += probas[cp]
                if s > r:
                    return cp

            assert r == somme
            return choice(coups)

        async def chance_move_treatment(self):

            ok = False

            while self.bot.referee.game.get_current_player() == -1 and not self.bot.referee.game.ended():
                #print('+chance_move_treatment')

                chance = self.selection_uniforme(self.bot.referee.game.valid_actions(), self.bot.referee.game.get_action_probabilities())

                self.bot.referee.game.plays(chance)

                #print('-',chance)

                try:
                    avant_trad = chance
                    chance = self.bot.referee.game.action_to_string(chance)
                    #print(avant_trad, '>',chance, '-', len(self.bot.referee.game.jeu.historique))
                    M = await self.bot.referee.channel.send(chance)

                    self.bot.bot_ref_log["moves"].append(chance)

                    await M.add_reaction('🟩')

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                #print('--')
                await self.save_bot_ref_log()
                #print('---')
                self.bot.referee.next_turn()
                #print('-chance_move_treatment')

                ok = True

            return ok



        #&#
        async def display_next_turn_info(self, channel: TextChannel, custom_elapsed_time: Time=None):
            """
            Coroutine that displays next turn information.

            PARAMETERS
            ----------
            channel: discord.TextChannel
                Channel where the game should be displayed.
            custom_elapsed_time: mytime.Time
                In case it is required to display a certain time.
            message_to_edit: discord.Message
                message to edit in case it is required to refresh the instruction message
            """
            if custom_elapsed_time is None:
                #print('X6.1')
                #print('>',self.current_turn)
                #print('>',self.time_remaining_player)
                #print('>',self.current_turn.mention)
                #print('>',self.time_remaining_player[self.current_turn])
                cpt = 0
                while True:

                    try:
                        #print('>',self.bot.referee.time_remaining_player[self.bot.referee.current_turn])
                        M = await channel.send(
                        f'{self.bot.referee.current_turn.mention} must play (he has {self.bot.referee.time_remaining_player[self.bot.referee.current_turn]} left)')
                        self.canceled_chronometer_number = 0
                        #await asyncio.sleep(0.01)
                        break
                    except DiscordServerError as e:
                        if cpt:
                            print('%',e)
                        await asyncio.sleep(1)
                        #import traceback
                        #traceback.print_exc()
                        cpt+=1
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                return M
            else:

                    #print('X7', self.bot.referee.game.get_current_player(), self.bot.referee.current_turn)
                    if self.instruction_message is not None and self.bot.referee.current_turn is not None and self.bot.referee.current_turn in self.bot.referee.time_remaining_player:
                        player = self.bot.referee.current_turn

                        if self.instruction_message.ended_time is not None:
                            sec = int(self.instruction_message.ended_time - self.instruction_message.creation_time)
                            milli = int((self.instruction_message.ended_time - self.instruction_message.creation_time - sec)*1000)
                            #print('A:',sec, milli)
                            tps = self.bot.referee.time_remaining_player[player] #- Time(second=sec, millisecond=milli)

                            # au moins parfois la bonne valeur c'est sans le moins
                            # si parfois ça requiert le moins, pour corriger, il suffit juste de ne pas mettre alors a jour la valeur du compte dans ce cas

                        else:
                            sec = int(time()-self.instruction_message.creation_time)
                            milli = int((time()-self.instruction_message.creation_time - sec) * 1000)

                            if int(time()-self.instruction_message.creation_time) % 120 == 119:
                                self.alarme('Purr')
                                self.alarme('Purr')
                            #print('B:', sec, milli)



                            tps = self.bot.referee.time_remaining_player[player] - Time(second=sec, millisecond=milli)
                        #print('t:', self.bot.referee.time_remaining_player[player], player)
                        if tps == Time():
                            #print('chronometer_stop Z')
                            self.chronometer_stop()
                            self.timeout_player = player


                        if self.instruction_message.last_time is None or self.instruction_message.last_time.to_seconds() != tps.to_seconds():
                            self.instruction_message.last_time = tps

                            try:
                                #print('>>',tps)
                                await self.instruction_message.message.edit(content=
                                    f'{player.mention} must play (he has {tps} left)')
                                self.canceled_chronometer_number = 0
                                #await asyncio.sleep(0.01)
                            except CancelledError as e:
                                #import traceback
                                #traceback.print_exc()
                                #print(self.instruction_message.deleted, self.instruction_message.message, self.bot.is_closed() , channel.guild.me)
                                #print('% CancelledError')
                                self.canceled_chronometer_number += 1
                                #await asyncio.sleep(self.canceled_chronometer_number)
                                if self.canceled_chronometer_number == 7:
                                    print('AHH:','CancelledError')
                                    self.chronometer_stop()
                                    #await asyncio.sleep(1)
                                    import traceback
                                    traceback.print_exc()
                                #raise e
                                """"""
                            except DiscordServerError as e:
                                #import traceback
                                #traceback.print_exc()
                                #print(self.instruction_message.deleted, self.instruction_message.message, self.bot.is_closed() , channel.guild.me)
                                #print('% DiscordServerError')
                                self.canceled_chronometer_number += 1
                                await asyncio.sleep(self.canceled_chronometer_number)
                                if self.canceled_chronometer_number == 7:
                                    print('AHH:','DiscordServerError')
                                    self.chronometer_stop()
                                    await asyncio.sleep(1)
                                    import traceback
                                    traceback.print_exc()
                                #raise e
                                """"""
                            except Exception:# DiscordServerError as e
                                #print(e)
                                await asyncio.sleep(1)
                                import traceback
                                traceback.print_exc()

        def chronometer_stop(self):
            self.chronometer.ended = True
            self.chronometer.stop()

        @tasks.loop(seconds=1)
        async def chronometer(self) -> None:
            """Task that will be executed during game rounds in order to monitor if a player exceeds its allowed time"""
            #t1 = perf_counter()
            #print('+chronometer')
            if not self.connected:
                return

            if not self.game_in_progress:
                self.chronometer_stop()
                #print('!')
                return

            self.chronometer_bugger = False

            if self.time_elapsed is not None and self.limit is not None:
                #print('?')
                try:
                    """t1 = Time.discord_utc_now()
                    if self.time_elapsed > self.limit:
                        print('>>>', self.time_elapsed , self.limit)
                        self.ended = True
                        self.chronometer.cancel()
                    t2 = Time.discord_utc_now()

                    self.time_elapsed += Time.timedelta_to_Time(t2 - t1) + Time(second=1)"""
                    #print('??')
                    if self.instruction_message is None or self.instruction_message.deleted is False:
                        #if not (self.time_elapsed > self.limit and self.bot.referee.channel is None):
                        if not (self.bot.referee.channel is None):
                            #&#await self.bot.referee.display_next_turn_info(self.bot.referee.channel, custom_elapsed_time=self.time_elapsed, message_to_edit=self.instruction_message.message)
                            #print('oula:',self.bot.referee.current_turn.mention)
                            #print('.??')
                            await self.display_next_turn_info(self.bot.referee.channel, custom_elapsed_time=self.time_elapsed)
                            #print('oula hop')

                        else:
                            """if self.time_elapsed > self.limit:
                                self.ended = True
                                print('là')
                                print('chronometer.stop')
                                self.chronometer.stop()
                            else:"""
                            #print('% OUPS?!',self.time_elapsed , self.limit)
                except Exception:
                    import traceback
                    traceback.print_exc()
            """t2 = perf_counter()
            print(f"Tick exécuté en {t2 - t1:.3f} s")
            print('latency: ',self.bot.latency)"""

            #print('-chronometer')

        async def loop_monitor(self):
            while True:
                t1 = perf_counter()
                await asyncio.sleep(1)
                drift = perf_counter() - t1 - 1
                print(f"Drift event loop : {drift:.3f} s")

        @chronometer.before_loop
        async def before_chronometer(self) -> None:
            """Coroutine that is executed when chronometer.start is called, before the start of the task"""
            # print("starting chronometer...")
            #self.ended = False
            self.time_elapsed = Time()

        @chronometer.after_loop
        async def after_chronometer(self) -> None:
            """Coroutine that is executed when chronometer is cancelled/stopped"""
            #print("ending chronometer...:", self.ended)
            #if self.ended: # je comprends pas pourquoi ended est à Faux ...
            #print('aa')
            if self.timeout_player:
                #print('bb')
                #print('timeout_player => display_time_exceed')
                await self.bot.referee.display_time_exceed(self.bot.referee.channel, self.bot.referee.current_turn)
                #print('cc')
            else:
                """"""


        def cog_unload_chronometer(self) -> None:
            """Function that end the execution of the chronometer task"""
            #print('ee')
            if self.chronometer.is_running():
                #print('ff')
                #print('ff, chronometer.stop P')
                self.chronometer.stop()
                #print('chronometer.stop')
                # print("chronometer canceled...")

        def resolve_member_or_role(self, ctx, arg: str):

            # User mention <@id> ou <@!id>
            match_user = re.match(r'<@!?(\d+)>', arg)
            if match_user:
                user_id = int(match_user[1])
                return ctx.guild.get_member(user_id)

            # Role mention <@&id>
            match_role = re.match(r'<@&(\d+)>', arg)
            if match_role:
                role_id = int(match_role[1])
                role = ctx.guild.get_role(role_id)
                if role.members:
                    return role.members[0]  # premier membre du rôle

                return None

            # Sinon, recherche par nom
            member = discord.utils.find(lambda m: m.name == arg or m.display_name == arg, ctx.guild.members)
            if member:
                return member

            role = discord.utils.get(ctx.guild.roles, name=arg)
            if role and role.members:
                return role.members[0]

            # Cas: mention textuelle du style @Nom
            if arg.startswith("@"):
                name = arg[1:]  # retirer le @
            else:
                name = arg

            # Chercher un membre par nom ou display_name
            # (insensible à la casse)
            name_lower = name.lower()
            for member in ctx.guild.members:
                if (
                        member.name.lower() == name_lower or
                        member.display_name.lower() == name_lower
                ):
                    return member

            # Si rien trouvé
            return None


        @commands.command(name='start', description='Starts a game with the current referee instance.', aliases=['s'])
        @commands.guild_only()
        #@RefereeBot.check_guild()
        #async def _start(self, ctx: Context, player1: User, player2: User, *args: None) -> None:
        async def _start(self, ctx: Context, player1: str = None, player2: str = None, *args: None) -> None:
            #print(player1, player2)
            #print('A')

            try:

                if player1 is None and player2 is not None or  player1 is not None and player2 is None:
                    return

                if player1 is None and player2 is None:
                    if self.last_players is None:

                        try:
                            with open('current_players.log', 'r') as f:
                                player2, player1 = f.read().split(';') # inversion des joueurs
                                player1 = self.resolve_member_or_role(ctx, player1)
                                player2 = self.resolve_member_or_role(ctx, player2)
                        except:
                            await ctx.send(f'Specify the players.')

                        if player1 is None or player2 is None:
                            print('Hein? (A)')

                    else:

                        player2, player1 = self.last_players
                        if player1 is None or player2 is None:
                            print('Hein? (B)')
                else:
                    #print('c')
                    player1 = self.resolve_member_or_role(ctx, player1)
                    player2 = self.resolve_member_or_role(ctx, player2)

                    if player1 is None or player2 is None:
                        print('Hein? (C)')


                with open('current_players.log','w') as f:
                    f.write(str(player1.mention)+';'+str(player2.mention))


                #print('B')
                #print('here', player1, player2)


                """Command that starts the game
    
                After verifying that the bot is neither in game nor in preparation, invoke the function that starts the game
    
                PARAMETERS
                ----------
                ctx : discord.Context
                    The invocation context.
                player1 : discord.User
                    The first player to be enrolled
                player2 : discord.User
                    The second player to be enrolled
    
                RETURNS
                -------
                None
                """


                if not self.bot.correct_context(ctx):
                    return

                #print("???")

                # Checking section
                # To start a game, the model should be in 'in-game' mode
                if self.bot.check_in_game():
                    raise commands.CheckFailure("The model shouldn't be in game when invoking this command.")

                #print('A0')

                # If the model is already in preparation, can't start a game
                if self.bot.check_in_preparation():
                    raise commands.CheckFailure("The model shouldn't be in preparation when invoking this command.")

                #print('B0')

                # The two players must be different
                if player1 == player2:
                    await ctx.send(f'The two players must be different.')
                    return

                #print('?')
                await self.start_game(ctx, player1, player2)

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        async def start_game(self, ctx: Context, player1: User, player2: User):
            """Coroutine that acts the start of the game

            It will start the game in the channel where the command was invoked.
            First, it sends an embed with a thumb up reaction.
            All player enrolled in the game have some time to react.
            If some of them missed, the game will not start (a red box reaction will be added for non-human users).
            Otherwise, the game will start (a green box reaction will be added for non-human users), and the first info are displayed.

            PARAMETERS
            ----------
            ctx : discord.Context
                The invocation context.
            player1 : discord.User
                The first player to be enrolled
            player2 : discord.User
                The second player to be enrolled

            RETURNS
            -------
            None
            """

            #print('+start_game')
            self.messages_history = []

            self.timeout_player = None

            #async with self.message_lock_beginend:
            #async with self.message_lock:
            # Retrieve the mentioned players
            players_involved = [player1.mention, player2.mention]  # transforming the User objects into strings

            #asyncio.create_task(self.loop_monitor())
            #print('M-1')
            # Preparation phase
            self.bot.referee.prepare()
            self.bot.referee.set_players((player1, player2))
            self.last_players = (player1, player2)
            self.bot.referee.set_referee(self.bot.user)

            #print('M0')

            self.bot.referee.set_turns()  # Set the order of turns using a uniform distribution (P1 -> P2 -> P1 -> P2 etc)
            #print('Ma')
            # Duration of the timer that waits for player to be ready
            duration = 15*60#180#20

            # Waiting for all reactions to be added
            players_to_react = [player for player in self.bot.referee.players]  # List of player expected to react

            #print('Mb')

            txt = f'**[rules]** {self.bot.referee.game.rules}\n'
            desc = (f"Game is starting ! \n"
                    f"The game will start when all of you have reacted a 👍 to this message. React 👎 to cancel.\n"
                    f"You have {duration} seconds to react."
                    f"**[players]** {','.join(players_involved)}\n"
                    f"**[game name]** {self.bot.referee.game.name.capitalize()}\n"
                    f"{txt if self.bot.referee.game.rules is not None else ''}"
                    #f"**[time per move mode]** {'Activated' if self.bot.referee.time_per_move_activated else 'Deactivated'}\n"
                    f"**[total time]** {self.bot.referee.game.time_per_player}")

            #print('Mc')
            try:
                prep_message = await ctx.send(desc)
                await prep_message.add_reaction('👍')
                await prep_message.add_reaction('👎')
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

            #print('Md')
            # Launch at the same time three coroutines
            # One that manages the timer of the specified duration
            # Two that wait for the players to react appropriately to the starting message
            try:

                coro_timer = self.launch_timer(prep_message, players_involved, players_to_react, duration)

                coro_wait_player1 = self.wait_react(prep_message, duration, players_to_react,
                                                    self.bot.referee.players[0])

                coro_wait_player2 = self.wait_react(prep_message, duration, players_to_react,
                                                    self.bot.referee.players[1])

                #print('DEBUG INFO:\t:','$')
                _, react1, react2 = await asyncio.gather(coro_timer, coro_wait_player1,
                                     coro_wait_player2)  # it returns each coroutine returned components

                #print('$5')

            # Case where the timer limit is reached
            except TimeoutError:
                try:
                    A = ', '.join(list(map(lambda x: x.mention, players_to_react)))
                    B = "has" if len(players_to_react) == 1 else "have"
                    await prep_message.edit(
                        content=f"Game will not start !\n{A} {B} not accepted.\n" f"The referee is reset")
                    await prep_message.clear_reactions()
                    await prep_message.add_reaction('🟥')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                # Reset the referee instance
                self.bot.referee.reset_end_game()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            # ...Otherwise, if everyone accepted on time
            else:
                ####
                def normalize_emoji(e: str) -> str:
                    return e.replace('\ufe0f', '')

                """print(prep_message.reactions)
                print(react1)
                print(react2)
                print(react1.emoji.__str__() == '👎️')
                print(react2.emoji.__str__() == '👎️')
                print('👎️')
                print(repr('👎️'),repr(react1.emoji.__str__()),repr(react2.emoji.__str__()))"""
                target = normalize_emoji('👎️')

                react1 = normalize_emoji(react1.emoji.__str__())
                react2 = normalize_emoji(react2.emoji.__str__())

                if react1 == target or react2 == target:

                    try:
                        A = ', '.join(list(map(lambda x: x.mention, players_to_react)))
                        B = "has" if len(players_to_react) == 1 else "have"
                        await prep_message.edit(
                            #content=f"Game will not start !\n{A} {B} refused.\n" f"The referee is reset")
                            content=f"Game will not start ! Someone has refused.\n" f"The referee is reset")
                        await prep_message.clear_reactions()
                        await prep_message.add_reaction('🟥')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                    # Reset the referee instance
                    self.bot.referee.reset_end_game()

                else:
                    try:
                        desc = (f"Game will start now !\n"
                                f"**[players]** {','.join(players_involved)}\n"
                                f"**[game name]** {self.bot.referee.game.name.capitalize()}\n"
                                f"{txt if self.bot.referee.game.rules is not None else ''}"
                                #f"**[time per move mode]** {'Activated' if self.bot.referee.time_per_move_activated else 'Deactivated'}\n"
                                f"**[total time]** {self.bot.referee.game.time_per_player}")

                        await prep_message.edit(content=desc)#f"Game will start now !"
                        await prep_message.clear_reactions()
                        await prep_message.add_reaction('🟩')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                    # Starting the referee in the channel where the command start was invoked
                    self.bot.referee.start(ctx.channel)

                    # Displaying the game state if the display mode is activated and if the game supports it
                    await self.bot.referee.display_game(ctx.channel)

                    # Setting the starting time
                    self.bot.referee.set_starting_time(Time.now())

                    # Setting up the log structure using the following format.
                    # {starting_time_game : {id_p1 : [move1, move2 ... moveN], id_p2 : [move1, move2 ... moveM], game_name : "...", time_per_player : nb_seconds, time_per_move_mode : true/false, winner : id_winner, loser : id_loser}, ...}
                    # It will contain each valid move played for each player in a particular game
                    self.bot.bot_ref_log = dict(
                        zip(map(lambda p: p.id, self.bot.referee.players), [[] for _ in self.bot.referee.players]))
                    self.bot.bot_ref_log["game_name"] = self.bot.referee.game.name
                    self.bot.bot_ref_log["time_per_player"] = self.bot.referee.game.time_per_player.to_seconds()
                    self.bot.bot_ref_log["time_per_move_mode"] = self.bot.referee.is_time_per_move_activated()
                    self.bot.bot_ref_log["moves"] = []
                    self.bot.bot_ref_log["players"] = list(map(lambda p: p.id, self.bot.referee.players))
                    self.bot.bot_ref_log["ended"] = False
                    self.bot.bot_ref_log["stopped"] = False

                    # Changing the name of the file
                    self.bot.set_json_file(ctx.guild.name, ctx.channel.name, self.bot.date,
                                           self.bot.referee.starting_time.get_logformated())

                    # Serializes the structure
                    await self.save_bot_ref_log()
                    #print('C')
                    await self.chance_move_treatment()

                    # Displaying next turn info
                    #&#self.instruction_message = InstructionMessage(message=await self.bot.referee.display_next_turn_info(ctx.channel))
                    self.instruction_message = None
                    if self.timeout_player is None:
                        #print('labas:',self.bot.referee.current_turn.mention)
                        self.instruction_message = InstructionMessage(message=await self.display_next_turn_info(ctx.channel))

                    # Set up chronometer limit and start it
                    self.limit = self.bot.referee.time_remaining_player[self.bot.referee.current_turn]

                    self.game_in_progress = True

                    #print('chronometer.start D')
                    try:
                        await self.chronometer.start()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e


            # In all cases, set _timer_cancel to False for next games
            finally:
                self.bot._timer_cancel = False


            #print('-start_game')

        @commands.command(name='pause', description='Pauses the ongoing game.')
        @commands.has_permissions(administrator=True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _pause(self, ctx: Context, *args: None) -> None:
            """Command that pauses the game"""

            try:

                if not self.bot.correct_context(ctx):
                    return

                if not self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model should be in game when invoking the pause command.")

                if self.bot.check_in_pause():
                    raise commands.CheckFailure(message="The model shouldn't be paused when invoking the pause command.")

                if self.bot.check_in_end_game():
                    raise commands.CheckFailure(message="The model shouldn't be in endgame when invoking the pause command.")

                if ctx.author != self.bot.referee.current_turn:
                    return

                # Store the player who paused
                self.bot.referee.pause(ctx.message.author)
                #print('chronometer.stop R')
                self.chronometer.stop()

                self.bot.referee.update_turn(self.instruction_message.message, ctx.message)

                try:
                    await ctx.send(f'Game has been paused by {ctx.message.author.mention}.')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name='resume', description='Resumes the ongoing game.')
        @commands.has_permissions(administrator=True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _resume(self, ctx: Context, *args: None) -> None:
            #print('+ _resume')

            try:

                if not self.bot.correct_context(ctx):
                    return

                if not self.bot.check_resumed_author(ctx):
                    raise commands.CheckFailure(message="The user requesting the game to be resumed should be the same as the one who initiated the pause.")
                if not self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model should be in game when invoking the resume command.")
                if self.bot.check_in_end_game():
                    raise commands.CheckFailure(message="The model shouldn't be in endgame when invoking the resume command.")
                if not self.bot.check_in_pause():
                    raise commands.CheckFailure(message="The model should be in pause when invoking the resume command.")

                try:
                    await ctx.message.channel.send(f'Game has been resumed.')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                self.bot.referee.resume()

                #&#self.instruction_message = InstructionMessage(message=await self.bot.referee.display_next_turn_info(self.bot.referee.channel))
                self.instruction_message = None
                if self.timeout_player is None:
                    #print('ici:',self.bot.referee.current_turn.mention)
                    self.instruction_message = InstructionMessage(message=await self.display_next_turn_info(self.bot.referee.channel))

                self.limit = self.bot.referee.time_remaining_player[self.bot.referee.current_turn]

                #print('chronometer.start E')
                self.ended = False
                try:
                    await self.chronometer.start()

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
            #print('- _resume')

        @commands.command(name='stop', description='Creates a poll to stop the ongoing game. All players have to vote.')#Only works when the game is paused, and all players have to vote.
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _stop(self, ctx: Context, *args: None) -> None:
            """Command that initiates the stopping feature in the ongoing game.

            It will write a message in the channel where the command was invoked, and will wait for all players enrolled
            in the game to react with either a thumb up or a thumb down.
            If there is at least one thumb down, the game continues.
            Otherwise, it will be stopped.
            During this period, the game must be paused.

            PARAMETERS
            ----------
            ctx : discord.Context
                The context in which the command was invoked

            RETURNS
            -------
            None
            """

            try:

                if not self.bot.correct_context(ctx):
                    return

                if not self.bot.check_player_in_game(ctx):
                    raise commands.CheckFailure(message="The user who invoked this command should be a player of the game.")

                if not self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model should be in game when invoking the stop command.")

                if self.bot.check_in_end_game():
                    raise commands.CheckFailure(message="The model shouldn't be in endgame when invoking the stop command.")

                if not self.bot.check_stop_activated():
                    raise commands.CheckFailure(message="The stop command is deactivated. To activate it, see ''!set stop''.")

                '''if not self.bot.check_in_pause():
                    raise commands.CheckFailure(message="The model should be paused to proceed this command")'''

                players_to_react, players_to_react_up, players_to_react_down = [player for player in self.bot.referee.players], [], []

                try:
                    stop_message = await ctx.send('Should the game be stopped ?')
                    await stop_message.add_reaction('👍')
                    await stop_message.add_reaction('👎')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                def check_stop(reaction, user):
                    """Checks if all the players reacted either a 👍 or a 👎"""
                    # Ignore reactions from non-player users
                    if user not in players_to_react:
                        return False

                    # If it is a thumb up... (that is one user who votes in favor of game termination)
                    elif str(reaction.emoji) == '👍':

                        players_to_react_up.append(user)
                        players_to_react.remove(user)

                        # If all players reacted with a thumb up...
                        if len(players_to_react_up) == len(self.bot.referee.players):
                            return True

                        return False

                    # ...else at least one player voted to continue
                    elif str(reaction.emoji) == '👎':

                        players_to_react_down.append(user)
                        players_to_react.remove(user)

                        # If one player voted a thumb down
                        if len(players_to_react_down) >= 1:
                            return True

                        return False

                # While some players did not react, continue to wait for reaction using the check function defined above
                while players_to_react:
                    try:
                        await self.bot.wait_for('reaction_add', check=check_stop)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                # If all players voted for anticipated game termination...
                if len(players_to_react_up) == len(self.bot.referee.players):
                    try:
                        await ctx.send('The game was stopped')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

                    # Delete the entry of this game in the log file
                    # del self.bot.bot_ref_log[self.bot.referee.starting_time.get_logformated()]
                    self.bot.bot_ref_log["stopped"] = True
                    self.bot.bot_ref_log["ended"] = True

                    # Serializes the structure
                    await self.save_bot_ref_log()

                    # Reset the referee instance
                    self.bot.referee.reset_end_game()

                    # Reset the instruction message
                    self.instruction_message.reset()

                # ...Otherwise the game continues
                else:
                    try:
                        await ctx.send('The game continues')

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name='history', description='Shows the history of today\'s recent games')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _history(self, ctx: Context, *args: None) -> None:
            """Coroutine that display the recent games that were played"""

            try:
                if not self.bot.correct_context(ctx):
                    return

                if self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model shouldn't be in game when invoking the history command.")

                # Going through the logs
                desc = ""

                # Iterating through the referee's log files
                #for file in os.listdir("log/bot_ref_log"):
                files = await asyncio.to_thread(os.listdir, 'log/bot_ref_log')
                for file in files:

                    # If it is not a json file with "ref" prefix, then the file is of no use for us
                    if not file.startswith("ref") and not file.endswith(".json"):
                        continue

                    # ...Otherwise, recover the information stored in the filename
                    _, guild_name, channel_name, date, starting_time = file.strip(".json").split("_")

                    # Display only the history for the guild where the command was invoked
                    if guild_name.replace("-", " ") != ctx.guild.name:
                        continue

                    # For each file of use, try to load it and use its information in the final display

                    # with open(os.path.join("log/bot_ref_log", file), mode="r") as fd:
                    async with aiofiles.open(os.path.join("log/bot_ref_log", file), mode="r") as fd:
                        try:
                            content = await fd.read()
                            loading = json.loads(content)
                            #loading = json.load(fd)
                        except JSONDecodeError:
                            print("ERROR JSON")
                        else:
                            if loading['ended']:
                                A = f"Winner <@{loading['winner']}>"
                                B = f"Loser <@{loading['loser']}>"
                                desc += (f"**On {date.replace('-', '.')} at {'h'.join(starting_time.split('-')[0:-1])}** {loading['game_name'].capitalize()} "
                                         #f"- {'time per move mode' if loading['time_per_move_mode'] else 'regular time mode'} "
                                         f"- {loading['time_per_player']}s {'per move' if loading['time_per_move_mode'] else 'per player'} - {'  '.join([A, B])}\n")

                # If the description is empty it means that there were no file that recorded games on the current guild
                if desc != "":
                    embed = Embed(title="Matches History", description=desc, color=Colour.random())
                    try:
                        await ctx.send(embed=embed)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                else:
                    try:
                        await ctx.message.reply(f"The files have no match recorded for the lasts sessions !")

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.group()
        async def set(self, ctx: Context) -> None:
            """Command that will be used to set various parameters in the game"""

            try:

                if not self.bot.correct_context(ctx):
                    return

                if self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model shouldn't be in game when invoking the set command.")

                if ctx.invoked_subcommand is None:
                    try:
                        await ctx.send("Invalid use of set command. Check '!help set' for more details.")

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @set.command(name='time_per_move',
                     description='Command that activates the time per move mode that allows player to have a certain amount of time each round.')
        @commands.has_permissions(administrator=True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _time_per_move_activate(self, ctx: Context, *args: None) -> None:
            """Subcommand of set that allows to flip the value of the time_per_move attribute of the referee instance"""

            try:

                if not self.bot.correct_context(ctx):
                    return


                self.bot.referee.time_per_move_activate()

                if self.bot.referee.is_time_per_move_activated():
                    try:
                        await ctx.send(f'The time per round mode is now activated.')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                else:
                    try:
                        await ctx.send(f'The regular time mode is now activated.')

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @set.command(name='time', description='Change the time allowed to each player')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _time(self, ctx: Context, new_time: str) -> None:
            """Subcommand of set that sets time limit per player to the time in seconds in argument"""

            try:
                if not self.bot.correct_context(ctx):
                    return

                if self.bot.check_in_preparation():
                    raise commands.CheckFailure(message="The model shouldn't be in preparation when invoking the set time command.")

                # Try to set the time for players using the parameter time...
                try:
                    if new_time.isdigit():
                        self.bot.referee.game.set_time_per_player(Time.seconds_to_Time(int(new_time)))
                    else:
                        self.bot.referee.game.set_time_per_player(Time.string_to_Time(new_time))

                    try:
                        await ctx.send(f"Time allowed to each player in a single game of {self.bot.referee.game.name.capitalize()} set to {self.bot.referee.game.time_per_player}")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                # ...Can't set a negative time
                except NegativeError:
                    try:
                        await ctx.send("Not the right format for time")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @set.command(name='game',
                     description='Let a player change the game/gamemode of the referee instance if already listed in the available games.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _game(self, ctx: Context, gamemode: str, *args: None) -> None:
            """Subcommand that allows to change the game that the referee is currently using"""

            try:

                if not self.bot.correct_context(ctx):
                    return


                if self.bot.check_in_preparation():
                    raise commands.CheckFailure(message="The model shouldn't be in preparation when invoking the set game command.")

                # If the gamemode is not referenced in the enumeration EnumGames...
                game = EnumGames.find_game(gamemode)
                if game is None:
                    try:
                        await ctx.send('Game does not exist')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                # ...Otherwise, the game exists
                else:
                    self.bot.referee.set_game(game)
                    try:
                        await ctx.send(f'Game set to {gamemode.capitalize()}')
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise e
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name = 'available_games', description = 'list of available games.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _available_games(self, ctx: Context, *args:None) -> None:
            """Command that resets the current referee instance."""

            try:

                if not self.bot.correct_context(ctx):
                    return

                try:
                    await ctx.send('list of available games: '+', '.join([e.capitalize() for e in EnumGames.__members__]).replace('Gtp_','GTP_'))

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name='add_freegame_moves', description='Add valid actions for the game Free_Game.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _add_freegame_moves(self, ctx: Context, *args: str) -> None:
            """Command that resets the current referee instance."""

            try:

                if not self.bot.correct_context(ctx):
                    return

                if args:
                    FreeGame.EXTRA_MOVE_KEYWORDS.extend(args)
                    message = 'Keywords for Free_game added.'
                else:
                    message = 'No keywords to add.'

                try:
                    await ctx.send(message)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name='show_extra_freegame_moves', description='Show extra valid actions for the game Free_Game.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _show_freegame_moves(self, ctx: Context) -> None:
            """Command that resets the current referee instance."""

            try:

                if not self.bot.correct_context(ctx):
                    return

                try:
                    await ctx.send('Extra Keywords of Free_game: '+', '.join(FreeGame.EXTRA_MOVE_KEYWORDS))

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e


        @commands.command(name='clear_freegame_moves', description='Clear extra valid actions for the game Free_Game.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _clear_freegame_moves(self, ctx: Context) -> None:
            """Command that resets the current referee instance."""

            try:
                if not self.bot.correct_context(ctx):
                    return

                FreeGame.EXTRA_MOVE_KEYWORDS = []

                try:
                    await ctx.send('Free_game now contains only basic moves: "A1", "A2-B3", ... and "end".')

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @commands.command(name='show_move_keywords', description='Show move keywords of the current game ("A1", "B2-A3" excluded).')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _show_move_keywords(self, ctx: Context) -> None:
            """Command that resets the current referee instance."""

            try:

                if not self.bot.correct_context(ctx):
                    return

                try:
                    if self.game_in_progress:
                        await ctx.send('Move keywords of '+self.bot.referee.game.name+': ' + ', '.join(self.bot.referee.game.move_keywords))
                    else:
                        await ctx.send('The game must be started to use this command.')

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @set.command(name='stop', description='Command that activates/deactivates the stop command.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _stop_activate(self, ctx: Context, *args: None) -> None:
            """Subcommand that flips the value of the stop_activated field of the referee instance"""

            try:
                if not self.bot.correct_context(ctx):
                    return

                self.bot.referee.stop_activate()

                try:
                    if self.bot.referee.is_stop_activated():
                        await ctx.send(f'The stop command has been activated.')

                    else:
                        await ctx.send(f'The stop command has been deactivated.')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @set.command(name='display', description='Command that activates the display feature that is displaying the game state at each step.')
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _display_activate(self, ctx: Context, *args: None) -> None:
            """Subcommand that flips the value of the display_activated field of the referee instance"""

            try:
                if not self.bot.correct_context(ctx):
                    return

                self.bot.referee.display_activate()

                try:
                    if self.bot.referee.is_display_activated():
                        await ctx.send(f'The display has been activated.')

                    else:
                        await ctx.send(f'The display has been deactivated.')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
        async def launch_timer(self, message: Message, players_involved: List[str], players_to_react: List[User], duration: int = 0) -> None:
            """Coroutine to launch a timer of specified duration

            PARAMETERS
            ----------
            message : discord.Message
                Original message
            players_involved : list[str]
                List of mentions
            players_to_react : list[discord.User]
                List of players from whom we need a reaction
            duration :
                Duration of the timer in seconds

            RETURNS
            -------
            None
            """
            #print('DEBUG INFO:\t:', 'a')
            try:
                # Transforming the duration in seconds into a Time object for the start_timer time limit
                self.limit = Time.seconds_to_Time(duration)

                # If the start_timer is already running, restart it...
                if self.start_timer.is_running():
                    #print('DEBUG INFO:\t:', 'a1')
                    self.start_timer.restart(message, players_involved, players_to_react)
                    #print('DEBUG INFO:\t:', 'b1')

                # ...Otherwise, start it normally
                else:
                    #print('DEBUG INFO:\t:', 'a2')
                    await self.start_timer.start(message, players_involved, players_to_react)
                    #print('DEBUG INFO:\t:', 'b2')

                # If the timer stopped and error was set to true, it means that the timer reached its limit without all players being ready
                if self.error:
                    raise TimeoutError("Some players did not vote.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

            #print('DEBUG INFO:\t:', 'b')
        @tasks.loop(seconds=1.)
        async def start_timer(self, message: Message, players_involved: List[str], players_to_react: List[User]) -> None:

            try:

                """Task that simulates a timer."""
                self.time_elapsed += Time(second=1)

                # If the time limit is reached, then set error to True and stop the task...
                if self.time_elapsed > self.limit:
                    self.error = True
                    self.start_timer.stop()

                # ...Otherwise, if all players have reacted positively, set _timer_cancel to True...
                if not players_to_react:
                    self.bot._timer_cancel = True

                # ...Otherwise, if _timer_cancel is True, cancel the task.
                # This if statement is added because _timer_cancel can modified by another coroutine executed at the same time this task being executed
                if self.bot.timer_cancel:
                    self.start_timer.stop()

                # Modify the preparation message on the channel
                txt = f'**[rules]** {self.bot.referee.game.rules}\n'
                desc = (f"Game is starting ! \n"
                        f"The game will start when all of you have reacted a 👍 to this message. React 👎 to cancel.\n"
                        f"You have {(self.limit - self.time_elapsed).to_seconds()} seconds to react.\n"
                        f"**[players]** {','.join(players_involved)}\n"
                        f"**[game name]** {self.bot.referee.game.name.capitalize()}\n"
                        f"{txt if self.bot.referee.game.rules is not None else ''}"
                        #f"**[time per move mode]** {'Activated' if self.bot.referee.time_per_move_activated else 'Deactivated'}\n"
                        f"**[total time]** {self.bot.referee.game.time_per_player}")
                try:
                    await message.edit(content=desc)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        @start_timer.before_loop
        async def before_start_timer(self) -> None:
            """Coroutine executed before start_timer task starts"""
            # Set fields to their default value
            self.error = False
            self.time_elapsed = Time()
            self.ended = False
            #print("starting timer...")

        @start_timer.after_loop
        async def after_start_timer(self) -> None:
            """Coroutine executed after start_timer task ends"""
            #print("ending timer...")
            pass

        def cog_unload_start_timer(self) -> None:
            """Function that unload the start_timer task"""
            if self.start_timer.is_running():
                self.start_timer.stop()

        async def wait_react(self, prep_message: Message, duration: int, players_to_react: List[User], player: User) -> None:
            """Coroutine that wait for the players to react

            PARAMETER
            ---------
            prep_message : discord.Message
                Original message
            duration : int
                duration before timing out
            players_to_react : list[discord.User]
                list of players we want to react
            player : discord.User
                player we are looking for

            RETURNS
            -------
            None
            """
            #print('AA')
            def check(reaction: Reaction, user: User) -> bool:
                """Check function that verifies that the reaction added by user is 👍, and that user is enrolled in the game"""
                try:
                    ok = str(reaction.emoji) == '👍' and user == player and reaction.message.id == prep_message.id

                    not_ok = str(reaction.emoji) == '👎' and (user == player or user == self.bot.operator[player] or self.bot.operator[player] == None and user != self.bot.user) and reaction.message.id == prep_message.id


                    #print(user, reaction, ':', ok or not_ok)
                    #print(str(reaction.emoji) == '👍', str(reaction.emoji) == '👎', user == player, user == self.bot.operator[player], self.bot.operator[player] == None , user != self.bot.user)
                    #print()

                    return ok or not_ok
                except:
                    print('DEBUG INFO:\t:',type(self.bot.operator))
                    print('DEBUG INFO:\t:',self.bot.operator)
                    print('DEBUG INFO:\t:',user.mention,':',reaction.emoji, '(',user == player, reaction.message.id == prep_message.id,')')
                    return (str(
                        reaction.emoji) == '👍' or str(
                        reaction.emoji) == '👎') and user == player and reaction.message.id == prep_message.id

            for _ in range(5):
                #print('BB')
                try:
                    prep_message = await prep_message.channel.fetch_message(prep_message.id)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                #print('CC')
                usr = None
                react = None
                ok = False
                for reaction in prep_message.reactions:
                    async for user in reaction.users():
                        if check(reaction, user):
                            usr = user
                            react = reaction
                            ok = True
                            break
                        if ok:
                            break
                if ok:
                    break

            #print('DD')
            # Wait for the proper reaction from a player of the gale
            if not ok:
                try:
                    react, usr = await self.bot.wait_for(f'reaction_add', check=check)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

            #print('EE')
            #print(react, '???')
            # Remove the user from the list
            if usr in players_to_react:
                players_to_react.remove(usr)

            #print('FF')
            # If all players reacted positively, cancel the ongoing timer...
            if players_to_react == [] or str(react.emoji) == '👎':
                self.bot._timer_cancel = True

            #print('GG', react)

            return react

        async def save_bot_ref_log(self):

            self.bot.bot_ref_log['messages_history'] = self.messages_history

            """ with open(self.bot.json_file, 'w') as file:
                                json.dump(self.bot.bot_ref_log, file)"""

            try:
                async with aiofiles.open(self.bot.json_file, 'w') as f:
                    await f.write(json.dumps(self.bot.bot_ref_log))
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        def alarme(self, nom='Glass'):
            try:
                if os.path.exists('/System/Library/Sounds/'):
                    os.system("afplay /System/Library/Sounds/"+nom+".aiff")
            except:
                """"""

            """Basso.aiff	Frog.aiff	Hero.aiff	Pop.aiff	Submarine.aiff
Blow.aiff	Funk.aiff	Morse.aiff	Purr.aiff	Tink.aiff
Bottle.aiff	Glass.aiff	Ping.aiff	Sosumi.aiff"""
            """Basso
    Frog
    Hero
    Pop
    Submarine
    Blow	
    Funk	
    Morse	
    Purr
    Tink
    Bottle	
    Glass	
    Ping 
    Sosumi"""

            """Basso
            Grenouille
            Héros
            Pop
            Sous-marin
            Souffle
            Funk
            Morse
            Ronron
            Fée Clochette
            Bouteille
            Verre
            Ping
            Sosumi"""

        @commands.Cog.listener()
        async def on_message_edit(self, before, after):

            try:

                if before.author == self.bot.user:
                    return

                if before.guild.id != self.bot.guild_id or before.channel.id != self.bot.channel_id:
                    return False

                if self.bot.referee.is_in_game() and before.channel != self.bot.referee.channel:  ### ICI blocage channel que j ai ajouter ###
                    return False

                #print('+on_message_edit', self.bot.referee.game.move_verifier.fullmatch(before.content))
                if self.bot.referee.game.move_verifier.fullmatch(before.content):
                    await after.channel.send("Editing game action messages is **prohibited!** The referee bot ignores this change. Restore the original action. Otherwise the edit can cause system instability. If this occurs, the opponent will be considered the winner of the match.")
                    """try:
                        await after.edit(content=before.content)
                    except
                        import traceback
                        traceback.print_exc()
                        raise e"""
                #print('-on_message_edit')

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        async def save_backup(self):

            #print('S1')
            self_dict = get_attributs(self)
            #print('S2')
            referee_dict = get_attributs(self.bot.referee)
            #print('S3')

            del self_dict['bot']
            del self_dict['message_lock']
            del self_dict['message_lock_beginend']
            del self_dict['instruction_message']
            del self_dict['timeout_player']
            #del self_dict['last_channel']


            self_dict = remove_unpicklable(self_dict)#:operator.id
            #print(self.bot.operator.items())

            operators = {}
            for user, operator in self.bot.operator.items():
                if operator is not None:
                    operators[user.id] = operator.id
                else:
                    operators[user.id] = None


                    #print('S4')
            del referee_dict['player_correspondence']
            del referee_dict['player_anti_correspondence']
            del referee_dict['current_turn']
            del referee_dict['channel']
            del referee_dict['players']

            del referee_dict['time_elapsed_player']
            del referee_dict['time_remaining_player']
            del referee_dict['time_exceeding_player']

            referee_dict = remove_unpicklable(referee_dict)

            """for k, v in self_dict.items():
                print('    ',k)
                await save_obj(v, 'test')

            for k, v in referee_dict.items():
                print('    ',k)
                await save_obj(v, 'test')"""

            #print('S5')
            players = [p.id for p in self.bot.referee.players]

            #print('S6')
            channel = self.bot.referee.channel.id

            #print('S7')
            if self.instruction_message is None:
                instruction_message = None
                #print('S8a')
            else:
                if self.instruction_message.message is None:
                    message = None
                else:
                    message = self.instruction_message.message.id

                #print('S8b')
                deleted = self.instruction_message.deleted
                creation_time = self.instruction_message.creation_time
                last_time = self.instruction_message.last_time
                ended_time = self.instruction_message.ended_time

                instruction_message = (message, deleted, creation_time, last_time, ended_time)

            time_elapsed_player = {}
            for k, v in self.bot.referee.time_elapsed_player.items():
                time_elapsed_player[k.id] = v

            time_remaining_player = {}
            for k, v in self.bot.referee.time_remaining_player.items():
                time_remaining_player[k.id] = v

            time_exceeding_player = {}
            for k, v in self.bot.referee.time_exceeding_player.items():
                time_exceeding_player[k.id] = v


            player_correspondence = {}
            for k, v in self.bot.referee.player_correspondence.items():
                player_correspondence[k.id] = v

            player_anti_correspondence = {}
            for k, v in self.bot.referee.player_anti_correspondence.items():
                if v is None:
                    player_anti_correspondence[k] = v
                else:
                    player_anti_correspondence[k] = v.id

            if self.bot.referee.current_turn is None:
                current_turn = None
            else:
                current_turn = self.bot.referee.current_turn.id


            #print('S9')
            await save_obj((self_dict, referee_dict, players, channel, instruction_message, time_elapsed_player, time_remaining_player, time_exceeding_player, self.bot.json_file, self.bot.bot_ref_log, operators, player_correspondence, player_anti_correspondence, current_turn), 'referee.backup')

            #print('S10')
        async def load_backup(self, context):
            #print('+load_backup')
            try:
                self_dict, referee_dict, players, channel, instruction_message, time_elapsed_player, time_remaining_player, time_exceeding_player, self.bot._json_file, self.bot._bot_ref_log, operators, player_correspondence, player_anti_correspondence, current_turn = await load_obj('referee.backup')

                self.bot.referee.set_referee(self.bot.user)

                if current_turn is None:
                    self.bot.referee.current_turn = None
                else:
                    self.bot.referee.current_turn = await self.bot.fetch_user(current_turn)

                self.bot.referee.player_correspondence = {}
                for k, v in player_correspondence.items():
                    self.bot.referee.player_correspondence[await self.bot.fetch_user(k)] = v

                self.bot.referee.player_anti_correspondence = {}
                for k, v in player_anti_correspondence.items():
                    if v is None:
                        self.bot.referee.player_anti_correspondence[k] = v

                    else:
                        self.bot.referee.player_anti_correspondence[k] = await self.bot.fetch_user(v)

                self.messages_history = self.bot.bot_ref_log['messages_history']

                self.bot.operator = defaultdict(lambda: None)
                for user, operator in operators.items():
                    if operator is not None:
                        self.bot.operator[await self.bot.fetch_user(user)] = await self.bot.fetch_user(operator)
                    else:
                        self.bot.operator[await self.bot.fetch_user(user)] = None



                #print('L1')
                self.__dict__.update(self_dict)
                #print('>1',self.bot.referee.game)
                self.bot.referee.__dict__.update(referee_dict)
                #print(referee_dict)
                #print('>2',self.bot.referee.game)
                #print('L2')
                self.bot.referee.players = [await self.bot.fetch_user(p) for p in players]
                #print('L3')
                self.bot.referee.channel = await self.bot.fetch_channel(channel)
                #print('channel:',self.bot.referee.channel)
                #print('L4')
                self.bot.referee.set_turns()
                #print('L5')
                if instruction_message is None:
                    self.instruction_message = None
                else:
                    message, deleted, creation_time, last_time, ended_time = instruction_message

                    try:#-#
                        if message is not None:
                            message = await context.fetch_message(message)

                        self.instruction_message = InstructionMessage(message=message, deleted=deleted)

                        self.instruction_message.creation_time = creation_time
                        self.instruction_message.last_time = last_time
                        self.instruction_message.ended_time = ended_time
                    except Exception:#-#
                        self.instruction_message= None#-#

                for player in self.bot.referee.players:
                    self.bot.referee.time_elapsed_player[player] = time_elapsed_player[player.id]
                    self.bot.referee.time_remaining_player[player] = time_remaining_player[player.id]
                    self.bot.referee.time_exceeding_player[player] = time_exceeding_player[player.id]
                #print('channel:', self.bot.referee.channel)
            except Exception:
                import traceback
                traceback.print_exc()
            #print('-load_backup')

        async def continue_match(self, context):

            if not self.bot.referee.is_in_game():

                use_chance = False

                await self.load_backup(context)
                #print(self.bot.referee.game.__class__.__name__)
                #print('start reprise')
                await self.reprise_des_messages(continue_reprise=True)
                #print(self.bot.referee.game.__class__.__name__)
                #print('channel reprise:', self.bot.referee.channel)
                if self.bot.referee.game.get_current_player() == -1:
                    #print('B')
                    use_chance = await self.chance_move_treatment()
                
                    if use_chance:
                        self.limit = self.bot.referee.time_remaining_player[self.bot.referee.current_turn]

                if self.need_instruction_message or use_chance:
                    self.need_instruction_message = False

                    self.instruction_message = None
                    if self.timeout_player is None:
                        #print('la:',self.bot.referee.current_turn.mention)
                        self.instruction_message = InstructionMessage(message=await self.display_next_turn_info(self.bot.referee.channel))

                #print(self.chronometer_bugger , not self.chronometer.is_running() , not self.bot.referee.game.ended() , not self.ended)
                if not self.chronometer.is_running() and not self.bot.referee.game.ended() and not self.ended:
                    #print('chronometer.start F')
                    #print('chrono start reprise')
                    self.chronometer.start()

        @commands.command(name='continue', description='Continue the match.', aliases=['c'])
        @commands.has_permissions(administrator=True)
        @commands.guild_only()
        #@RefereeBot.check_guild()
        async def _continue(self, ctx: Context) -> None:
            #print("???")
            """Command that starts the game

            After verifying that the bot is neither in game nor in preparation, invoke the function that starts the game

            PARAMETERS
            ----------
            ctx : discord.Context
                The invocation context.

            RETURNS
            -------
            None
            """

            try:
                if not self.bot.correct_context(ctx):
                    return

                #print('+continue')
                # Checking section
                # To start a game, the model should be in 'in-game' mode

                """if self.bot.check_in_game():
                    raise commands.CheckFailure(message="The model shouldn't be in game when invoking the dump command.")

                if self.bot.check_in_preparation():
                    raise commands.CheckFailure(message="The model shouldn't be in preparation when invoking the dump command.")

                if self.bot.check_in_end_game():
                    raise commands.CheckFailure(message="The model shouldn't be in endgame when invoking the resume command.")"""

                await self.continue_match(ctx)  #
                #print('-continue')
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

async def save_obj(obj, filename):
    #print('+save_obj')
    try:
        data = pickle.dumps(obj)  # objet -> bytes
        async with aiofiles.open(filename, "wb") as f:
            await f.write(data)
    except:
        import traceback
        traceback.print_exc()

    #print('-save_obj')

def remove_unpicklable(d):
    clean = {}
    for k, v in d.items():
        try:
            pickle.dumps(v)
        except Exception:
            # l'objet n'est pas picklable --> on l'ignore
            continue
        clean[k] = v
    return clean
async def load_obj(filename):
    try:
        async with aiofiles.open(filename, "rb") as f:
            data = await f.read()
        return pickle.loads(data)  # bytes -> objet

    except Exception:
        import traceback
        traceback.print_exc()

def get_attributs(Classe):
    d = {}
    for k, v in Classe.__dict__.items():
        if not callable(v) and '_' != k[0] and '_' != k[-1]:
            d[k] = v
    return d
