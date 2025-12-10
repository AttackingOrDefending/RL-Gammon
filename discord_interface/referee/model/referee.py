
from datetime import datetime, timedelta

from discord_interface.games.mygame import Game
from discord_interface.utils.mytime import Time
from discord import User, TextChannel, Message
from random import randint
from typing import Tuple
from typing import Any

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

if __name__ != '__main__':

    class Referee:
        """Class to represent and to offer tools to manipulate a referee model.

        ATTRIBUTES
        ----------
        in_game : bool
            True if a game has started, False otherwise
        in_preparation : bool
            True if the game has not started yet but will start soon (when the players are ready and write in the chat)
        in_end_game : bool
            True if the game is in a final state, that is it will be terminated soon
        paused : bool
            True if the game is currently paused
        paused_by : discord.User
            The last user who paused the game

        stop_activated : bool
            True if the game should support "by-hand" stop command (!stop)
        display_activated : bool
            True if the game should display the game status at each instruction during the game (!set display to (de)activate)
        time_per_move_activated : bool
            True if the game should support time_per_move time mode, that is instead of having a global limit of time each player is granted a certain amount of time per move
        players : tuple[discord.User, discord.User]
            Tuple of the players involved
        game : mygame.Game
            Variant of the game currently set up by the model
        channel : discord.TextChannel
            The channel where the game is currently running, that is the one where "!start" was invoked

        current_turn : discord.User
            Indicates which player is currently in turn

        starting_time : mytime.Time
            The time when the current game has started
        time_elapsed : mytime.Time
            The time elapsed since the game has started
        time_elapsed_player : dict[discord.User, mytime.Time]
            A dictionary containing for each player of players the time he has used
        time_remaining : mytime.Time
            The time remaining until the end of the game
        time_remaining_player : dict[discord.User, mytime.Time]
            A dictionary containing for each player of players the time remaining
        time_exceeding_player : dict[discord.User, mytime.Time]
            a dictionary containing for each player of players how much time has exceeded the limit

        player_correspondence : dict[discord.User, Any]
            Dictionary that maps every player to its name in the current game
        player_anti_correspondence : dict[Any, discord.User]
            Dictionary that maps every name in the current game to its associated player
        """

        def __init__(
                self,
                in_game: bool = False,
                in_preparation: bool = False,
                in_end_game: bool = False,
                paused: bool = False,
                paused_by: User = None,
                stop_activated: bool = True,
                display_activated:bool = False,
                time_per_move_activated:bool = False,
                players: tuple[User, User] = None,
                game: Game = None,
                channel: TextChannel = None,
                current_turn: User = None,
                starting_time: Time = Time(),
                time_elapsed: Time = Time(),
                time_elapsed_player=None,
                time_remaining: Time = Time(),
                time_remaining_player=None,
                time_exceeding_player = None,
                player_correspondence: dict[User, Any] = None,
                player_anti_correspondence: dict[Any, User] = None
        ) -> None:
            if time_elapsed_player is None:
                time_elapsed_player = dict()
            if time_remaining_player is None:
                time_remaining_player = dict()
            if time_exceeding_player is None:
                time_exceeding_player = dict()
            self.in_game = in_game
            self.in_preparation = in_preparation
            self.in_end_game = in_end_game
            self.paused = paused
            self.paused_by = paused_by
            self.stop_activated = stop_activated
            self.display_activated = display_activated
            self.time_per_move_activated = time_per_move_activated
            self.players = players
            self.game = game
            self.channel = channel
            self.current_turn = current_turn
            self.starting_time = starting_time
            self.time_elapsed = time_elapsed
            self.time_elapsed_player = time_elapsed_player
            self.time_remaining = time_remaining
            self.time_remaining_player = time_remaining_player
            self.time_exceeding_player = time_exceeding_player
            if not player_correspondence:
                self.player_correspondence = dict()
            else:
                self.player_correspondence = player_correspondence
            if not player_anti_correspondence:
                self.player_anti_correspondence = dict()
            else:
                self.player_anti_correspondence = player_anti_correspondence

        def __str__(self) -> str:
            return (f"* in_game: {self.in_game}\n"
                    f"* in_preparation: {self.in_preparation}\n"
                    f"* in_end_game: {self.in_end_game}\n"
                    f"* paused: {self.paused}\n"
                    f"* paused_by: {self.paused_by.name if self.paused_by is not None else None}\n"
                    f"* stop_activated: {self.stop_activated}\n"
                    f"* display_activated: {self.display_activated}\n"
                    f"* time_per_move_activated: {self.time_per_move_activated}\n"
                    f"* players: {', '.join(map(lambda x: x.name, self.players)) if self.players is not None else None}\n"
                    f"* game: {self.game.name if self.game is not None else None}\n"
                    f"* channel: {self.channel.name if self.channel is not None else None}\n"
                    f"* current_turn: {self.current_turn.name if self.current_turn is not None else None}\n"
                    f"* starting_time: {self.starting_time}\n"
                    f"* time_elapsed: {self.time_elapsed}\n"
                    f"* time_elapsed_player: {', '.join([f'{player.name}: {self.time_elapsed_player[player]}' for player in self.players]) if self.time_elapsed_player != {} else None}\n"
                    f"* time_remaining: {self.time_remaining}\n"
                    f"* time_remaining_player: {', '.join([f'{player.name}: {self.time_remaining_player[player]}' for player in self.players]) if self.time_remaining_player != {} else None}\n"
                    f"* time_exceeding_player: {', '.join([f'{player.name}: {self.time_exceeding_player[player]}' for player in self.players]) if self.time_exceeding_player != {} else None}\n"
                    f"* player_correspondence: {', '.join([f'{key.name} <-> {value}' for key, value in self.player_correspondence.items()]) if self.player_correspondence != {} else None}\n"
                    f"* player_anti_correspondence: {', '.join([f'{key} <-> {value.name}' for key, value in self.player_anti_correspondence.items()]) if self.player_anti_correspondence != {} is not None else None}\n")

        def get_channel(self) -> TextChannel:
            """getter for the field channel"""
            return self.channel

        def get_paused_by(self) -> User:
            """getter for the field paused_by"""
            return self.paused_by

        def enters_end_game(self) -> None:
            """Method to enter the end game phase"""
            self.in_end_game = True

        def end_game(self) -> None:
            """Method to end the game by setting the field in_game to False"""
            self.in_game = False

        def wins(self, player: User) -> None:
            """Method that sets the winner to player

            PARAMETERS
            ----------
            player : discord.User
                User that won the game
            """
            self.game.winner = self.player_correspondence[player]

        def set_game(self, game: Game) -> None:
            """Method to set up the field game with a particular game instance

            PARAMETERS
            ----------
            game : mygame.Game
                The game instance being associated to the referee
            """
            self.game = game

        def pause(self, author:User) -> None:
            """Method that pauses the current game

            Pauses, stores the author that requested to pause, and updates time fields.

            PARAMETERS
            ----------
            author : discord.User
                The author that requested to pause. Could be of type discord.User, discord.Member or discord.ClientUser
            """
            self.paused = True
            self.paused_by = author

        def resume(self) -> None:
            """Method that resumes the current game"""
            self.paused = False

        def reset_end_game(self) -> None:
            """Resets the game after game end, and reinitializes itself keeping the same game mode and the option to activate the stop command"""
            self.game.reset()
            self.__init__(game = self.game, stop_activated = self.stop_activated, display_activated = self.display_activated, time_per_move_activated = self.time_per_move_activated)

        def reset(self) -> None:
            """Resets the game, and reinitializes itself keeping only the same game mode"""
            self.game.reset()
            self.__init__(game = self.game)

        def prepare(self) -> None:
            """Method setting in_preparation to True"""
            self.in_preparation = True

        def start(self, channel: TextChannel) -> None:
            """Method that puts the referee in game mode

            PARAMETERS
            ----------
            channel : discord.TextChannel
                channel in which the start command was invoked, and the one where the game will take place.
            """
            # Exit the preparation phase
            self.in_preparation = False

            # Fixing the channel for the rest of the game
            self.channel = channel

            # Fixing time-related fields
            self.time_remaining = Time()

            for player in self.players:

                self.time_remaining_player[player] = self.game.get_time_per_player()
                self.time_exceeding_player[player] = Time(0, 0, 0, 0)
                self.time_remaining += self.time_remaining_player[player]
                self.time_elapsed_player[player] = Time()

            # The Bot is now in in-game mode
            self.in_game = True

        def set_players(self, players: Tuple[User, User]) -> None:
            """Setter that sets up the field containing the tuple of players involved in the game

            PARAMETERS
            ----------
            players : Tuple[discord.User, discord.User]
                tuple of players competing in the game
            """
            self.players = players

        def set_referee(self, user:User):
            self.referee_user = user

        def set_turns(self) -> None:
            """Method that initializes the turns

            First, copy the players involved in a list, and start a counter to 0 which will associate each player to its representation in the game considered.
            Then, iterate over the players, and pop randomly a player into the order we are building.
            Finally, use this order in order to create a cycle structure.
            Remark, that we build the player's correspondence and anti-correspondence on the fly
            """

            players = [member for member in self.players]
            order = []
            l = 0

            # While some players are still missing in the order...
            """
            while players:

                # Select uniformly a player to add to the order
                i = randint(0, len(players) - 1)
                member = players.pop(i)
                order.append(member)

                # Associate the i-th player to the i-th player in the internal game order structure
                self.player_correspondence[member] = l
                self.player_anti_correspondence[l] = member

                l+=1"""

            for l, player in enumerate(players):
                # Associate the i-th player to the i-th player in the internal game order structure
                self.player_correspondence[player] = l
                self.player_anti_correspondence[l] = player


            self.player_anti_correspondence[-1] = self.referee_user

            # Initialise the iterator
            self.current_turn = self.player_anti_correspondence[self.game.get_current_player()]

        def set_starting_time(self, time: Time) -> None:
            """Setter for the field starting time"""
            self.starting_time = time

        def update_turn(self, instruction_message: Message, move_message: Message) -> User:
            """Method that updates the current turn, and the time

            This method will update the current turn to the next one while saving and updating the time fields for the appropriate player.
            It will be performed using a disjunction of cases.

            PARAMETERS
            ----------
            instruction_message: Discord.Message
                Instruction message
            move_message: Discord.Message
                Move message

            RETURNS
            -------
            player.Player
                The player that was playing
            """
            time_elapsed_player = Time.timedelta_to_Time( max(move_message.created_at - instruction_message.created_at, timedelta(0)) )
            #print(red(str(time_elapsed_player)))#+' '+str(move_message.created_at - instruction_message.created_at)+'<'+' '+str((move_message.created_at - instruction_message.created_at).seconds)+' ('+str(move_message.created_at) +' '+str(instruction_message.created_at)))

            # If the time_per_move time mode is activated...
            if self.time_per_move_activated:
                return self.time_per_move_update_turn(time_elapsed_player)

            # ...Otherwise ...
            else:
                return self.regular_time_update_turn(time_elapsed_player)

        def time_per_move_update_turn(self, time_elapsed_player: Time) -> User:
            """Method that updates the current turn, and the time in the time_per_move time setting

            PARAMETERS
            ----------
            time_elapsed_player: mytime.Time
                time elapsed between the instruction message and the move message

            RETURNS
            -------
            player.Player
                The player that was playing
            """
            # Stop the timer and retrieve the user currently playing
            player = self.current_turn

            # Update the elapsed, remaining, and exceeding times
            self.time_elapsed += time_elapsed_player

            if self.time_remaining_player[player] - time_elapsed_player == Time(): # If the player exceeds it allowed time
                self.time_exceeding_player[player] += time_elapsed_player - self.time_remaining_player[player]

            try:
                self.time_elapsed_player[self.current_turn] += time_elapsed_player
            except KeyError:
                self.time_elapsed_player[self.current_turn] = time_elapsed_player

            self.time_remaining_player[self.current_turn] -= time_elapsed_player


            return player

        def increase_time(self, player: User, time: int):
            #print('+increase_time')

            if player in self.players:

                if time > 0:

                    time = Time.seconds_to_Time(time)

                    if player == self.current_turn:

                        self.time_elapsed -= time

                    """if self.time_remaining_player[player] - time_elapsed_player == Time(): # If the player exceeds it allowed time
                        self.time_exceeding_player[player] += time_elapsed_player - self.time_remaining_player[player]"""

                    try:
                        self.time_elapsed_player[player] -= time
                    except KeyError:
                        self.time_elapsed_player[player] = Time()

                    #print('\n\n',self.time_remaining_player[self.current_turn],'\n')
                    self.time_remaining_player[player] += time
                    #print('\n',self.time_remaining_player[self.current_turn],'\n\n')

                else:
                    time = Time.seconds_to_Time(-time)

                    if player == self.current_turn:
                        self.time_elapsed -= time

                    """if self.time_remaining_player[player] - time_elapsed_player == Time(): # If the player exceeds it allowed time
                        self.time_exceeding_player[player] += time_elapsed_player - self.time_remaining_player[player]"""

                    try:
                        self.time_elapsed_player[player] += time
                    except KeyError:
                        self.time_elapsed_player[player] = time

                    #print('\n\n',self.time_remaining_player[self.current_turn],'\n')
                    self.time_remaining_player[player] -= time
                    #print('\n',self.time_remaining_player[self.current_turn],'\n\n')

            #print('-increase_time')

        def regular_time_update_turn(self, time_elapsed_player: Time) -> User:
            """Method that updates the current turn, and the time in the regular time setting

            PARAMETERS
            ----------
            time_elapsed_player: mytime.Time
                time elapsed between the instruction message and the move message

            RETURNS
            -------
            player.Player
                The player that was playing
            """
            # Stop the timer and retrieve the user currently playing
            player = self.current_turn

            # Update the elapsed, remaining, and exceeding times
            self.time_elapsed += time_elapsed_player

            if self.time_remaining_player[player] - time_elapsed_player == Time(): # If the player exceed his allowed time
                self.time_exceeding_player[player] += time_elapsed_player - self.time_remaining_player[player]

            self.time_remaining -= time_elapsed_player

            try:
                self.time_elapsed_player[self.current_turn] += time_elapsed_player
            except KeyError:
                self.time_elapsed_player[self.current_turn] = time_elapsed_player

            self.time_remaining_player[self.current_turn] -= time_elapsed_player


            return player


        def next_turn(self) -> None:
            """Method that skips the current turn without updating any fields"""
            self.current_turn = self.player_anti_correspondence[self.game.get_current_player()]

        def opposite(self, player: User) -> User:
            """Method that returns the other player with respect to the one passed to parameter

            PARAMETERS
            ----------
            player : discord.User
                Player from whom we want the opposite.

            RETURNS
            -------
            discord.User
            """
            if player == self.players[0]:
                return self.players[1]
            return self.players[0]

        async def display_game(self, channel: TextChannel):
            """
            Coroutine that shows the game board if the current game is displayable. Otherwise does nothing

            PARAMETERS
            ----------
            channel : discord.TextChannel
                Channel where the game should be displayed.
            """
            if self.is_displayable():
                try:
                    await channel.send(self.game.show_game())
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
            else:
                pass

        '''async def display_next_turn_info(self, channel: TextChannel, custom_elapsed_time: Time=None, message_to_edit: Message=None):
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
            if message_to_edit is None:
                print('X6.1')
                #print('>',self.current_turn)
                #print('>',self.time_remaining_player)
                #print('>',self.current_turn.mention)
                #print('>',self.time_remaining_player[self.current_turn])
                cpt = 0
                while True:
                    print(f'>{cpt}>{self.current_turn.mention} must play (he has {self.time_remaining_player[self.current_turn]} left)')
                    try:
                        M = await channel.send(
                        f'{self.current_turn.mention} must play (he has {self.time_remaining_player[self.current_turn]} left)')
                        break
                    except DiscordServerError as e:
                        if cpt:
                            print(e)
                        await asyncio.sleep(1)
                        #import traceback
                        #traceback.print_exc()
                        cpt+=1

                return M
            else:
                print(f'>>{self.current_turn.mention} must play (he has {self.time_remaining_player[self.current_turn] - custom_elapsed_time} left)')
                try:
                    await message_to_edit.edit(content=
                        f'{self.current_turn.mention} must play (he has {self.time_remaining_player[self.current_turn] - custom_elapsed_time} left)')
                except CancelledError:
                    print('% CancelledError')
                    """"""
                except:# DiscordServerError as e
                    #print(e)
                    await asyncio.sleep(1)
                    import traceback
                    traceback.print_exc()'''

        async def display_time_exceed(self, channel: TextChannel, player: User):
            """Coroutine that displays on the channel passed in Parameter a message to indicate a time exceed from the player passed in parameter.

            PARAMETERS
            ----------
            channel: TextChannel
                Channel where to send the message
            player: discord.User
                Player that exceeded its time alloted
            """
            try:
                #print(f'{player.mention} has exceeded his allowed time:', datetime.now().strftime("%H:%M"))
                await channel.send(f'{player.mention} has exceeded his allowed time')
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e
        def stop_activate(self) -> None:
            """Method that flips the boolean value of the fields stop_activated"""
            self.stop_activated = not self.stop_activated

        def display_activate(self) -> None:
            """Method that flips the boolean value of the fields stop_activated"""
            self.display_activated = not self.display_activated

        def time_per_move_activate(self) -> None:
            """Method that flips the boolean value of the fields time_per_round_activated"""
            self.time_per_move_activated = not self.time_per_move_activated

        def is_in_game(self) -> bool:
            """Getter for the fields in_game"""
            return self.in_game

        def is_in_preparation(self) -> bool:
            """Getter for the fields in_preparation"""
            return self.in_preparation

        def is_in_pause(self) -> bool:
            """Getter for the fields in_pause"""
            return self.paused

        def is_in_end_game(self) -> bool:
            """Getter for the fields in_end_game"""
            return self.in_end_game

        def is_stop_activated(self) -> bool:
            """Getter for the fields stop_activated"""
            return self.stop_activated

        def is_display_activated(self) -> bool:
            return self.display_activated

        def is_time_per_move_activated(self) -> bool:
            return self.time_per_move_activated

        def is_displayable(self):
            return self.display_activated and self.game.show_game() is not None