"""File used for game representation.

This file contains a class called Game that represents a Game used by the RefereeBot as well as the PlayerBot.
"""
from typing import Any, List, Union, Set
from abc import ABCMeta, abstractmethod
from re import compile, Pattern

import numpy as np

from discord_interface.utils.mytime import Time
from discord_interface.utils.pattern_enum import EnumPattern

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

if __name__ != "__main__":

    class Game(metaclass = ABCMeta):
        """Abstract class representing a particular game with its own features and mechanisms.

        ATTRIBUTES
        ----------
        name : str
            the name of the game
        rules : str
            a formated version of its rules used for display purposes
        winner : Any
            the player who wins using the inner representation of a player (=/= discord members listed in the attribute "players" of referee)
        time_per_player : mytime.Time
            time given to each player for a given game. It represents both time per move, or time for a whole game depending on the attribute time_per_move_activated from Referee.
        move_keywords : list[str]
            list of keywords that can be used in a play. (e.g. ["tower", "queen", ...] for chess).
        move_verifier : re.Pattern
            compiled regular expression pattern that contains the pattern for a particular move.
        """
        def __init__(self, name: str='', rules:str='None', winner:Any=None, time_per_player:Time=None, move_keywords:list[str]=None) -> None:

            self.actions_validation_enabled = True
            self.name = name
            self.rules = rules
            self.winner = winner
            if time_per_player:
                self.time_per_player = time_per_player
            else:
                self.time_per_player = Time(minute=30)#Time(second=35)##30)#2)#5) 5)#
            if move_keywords is None:
                self.move_keywords = []
            else:
                self.move_keywords = move_keywords

            #print(self.move_keywords)
            self.move_keywords.append('resign')
            #print(self.move_keywords)

            self.move_verifier = None
            self.build_move_verifier()

        def build_move_verifier(self):
            if self.move_keywords == []:
                move_pattern = EnumPattern.RAW_PATTERN_MOVE.value
            else:
                move_pattern = EnumPattern.RAW_PATTERN_MOVE.value + "|" + "|".join(self.move_keywords)
            new_global_pattern = "(" + move_pattern + ")" + "(-(" + move_pattern + "))*"

            self.move_verifier = compile(new_global_pattern)

            #print('>>',self.move_keywords)



        def __eq__(self, other: Any) -> bool:
            """Redefinition of the equality operator.

            Two games are equal if they share the same name.
            Its main purpose is when command !setgame is invoked. Indeed, to search efficiently in the list of games, we will only compare the names.

            PARAMETERS
            ----------
            other : Object
                Object to compare our instance with

            RAISES
            ------
            TypeError
                Raises a TypeError exception whenever the object we want to compare our instance with has a different type from str or Game.

            RETURNS
            -------
            bool
                True if self has the same name as other (as a Game) or if self its name field has value equals to other (as a str), False otherwise
            """
            if type(other) == str:
                return self.name == other
            if isinstance(other, Game):
                return self.name == other.name
            raise TypeError(f'{other} and {self} have to be of type Game (or str)')

        def __str__(self) -> str:
            desc = (f"* name: {self.name}\n"
                    f"* rules: {self.rules}\n"
                    f"* winner: {self.winner}\n"
                    f"* time_per_player: {self.time_per_player}\n"
                    f"* move_keywords: {self.move_keywords}\n"
                    f"* move_verifier: {self.move_verifier}\n"
                    )
            return desc

        def show_rules(self) -> str:
            """Method that shows the rules of the current game instance.

            RETURNS
            -------
            str
                The rules represented as a string.
            """
            return self.rules

        def get_time_per_player(self) -> Time:
            return self.time_per_player

        def set_time_per_player(self, new_time:Time) -> None:
            self.time_per_player = new_time


        @abstractmethod
        def action_to_string(self, action:Any) -> Union[str, Any]:#[str|Any]:
            """Abstract method to convert an action into a string.

            PARAMETERS
            ----------
            action : Any
                An action in the game language.

            RETURNS
            -------
            str
                A string representation of the action.
            """
            return action

        @abstractmethod
        def string_to_action(self, string:str) -> Union[str, Any]:#[str|Any]:
            """Abstract method that transforms a string into an action for a particular game.

            PARAMETERS
            ----------
            string : str
                The string to transform.

            RETURNS
            -------
            Any
                The translated action.
            """
            return string

        @abstractmethod
        def plays(self, move:Any) -> None:
            """Abstract method that plays the move passed in parameter.

            PARAMETERS
            ----------
            move : Any
                The move to play.

            RETURNS
            -------
            None
            """
            pass


        def textual_plays(self, textual_move) -> None:
            self.plays(self.string_to_action(textual_move))

        @abstractmethod
        def undo(self) -> None:
            """Abstract method that undo the last move."""
            pass

        @abstractmethod
        def valid_actions(self) -> Set[Any]:
            """Abstract method that returns the set of actions that are valid in the current game state.

            RETURNS
            -------
            set
                Set of valid actions
            """
            return set()

        def textual_legal_moves(self):
            return [self.action_to_string(action) for action in self.valid_actions()]

        @abstractmethod
        def get_current_player(self) -> int:
            """Abstract method that returns the index of the current player.

            RETURNS
            -------
            int
                index of the current player
            """
            return 0

        @abstractmethod
        def ended(self) -> bool:
            """Abstract method that tells whether the game has ended.

            RETURNS
            -------
            bool
                True if the game has ended, False otherwise.
            """
            return False

        @abstractmethod
        def show_game(self) -> Any:
            """Abstract method that displays the current game board state.

            RETURNS
            -------
            Any
                A formated description of the current game board state.
            """
            return None

        @abstractmethod
        def reset(self) -> None:
            """Abstract method that reinitializes the game instance.

            RETURNS
            -------
            None
            """
            self.__init__(name=self.name, rules=self.rules, time_per_player=self.time_per_player, move_keywords=self.move_keywords)

        @abstractmethod
        def get_numpy_board(self) -> np.array:
            pass

        @abstractmethod
        def terminate(self, winner):
            """Abstract method that terminate the game instance.

            method "ended" must now return True
            attribute "winner" must be set as winner
            -------
            None
            """