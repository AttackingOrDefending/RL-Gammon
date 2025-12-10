"""File used for pattern enumeration.

This file contains a class called EnumPattern that stores patterns used by the RefereeBot as well as the PlayerBot to recognize particular patterns in a message.
"""
from enum import Enum#, verify, UNIQUE
import re

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"

if __name__ != "__main__":

    #@verify(UNIQUE)
    class EnumPattern(Enum):
        """Enumeration that stores the (raw) patterns for different purposes.

        Note that below
          <text> is any alphanumerical string,
          <mention> is a discord mention that is <@n1...nk> where ni is a integer and k is either 17 or 18,
          ? means could appear once or not at all,
          <text-and-number> is a alphanumerical string of size at least 2,


        RAW_PATTERN_REQUEST matches all expressions of the form !<text> [<text>|<mention>] [<text>|<mention>] ...
        RAW_PATTERN_COMMAND matches all expressions of the form !<text>
        RAW_PATTERN_ARGUMENT matches all expressions of the form <text>
        RAW_PATTERN_INSTRUCTION matches all expressions of the form <mention> must play (he has [<2-digit-number>h]?[<2-digit-number>min]?[<2-digit-number>s]?[<3-digit-number>ms]? left)
        RAW_PATTERN_MOVE matches all expressions of the form <text-and-number>-<text-and-number>
        RAW_PATTERN_MENTION matches all expressions of the form <mention>
        RAW_PATTERN_WIN matches all expressions of the form <mention> won
        RAW_PATTERN_EXCEED_TIME matches all expressions of the form <mention> has exceeded his allowed time
        RAW_PATTERN_HOUR matches all expressions of the form <2-digit-number>h
        RAW_PATTERN_MIN matches all expressions of the form <2-digit-number>min
        RAW_PATTERN_SEC matches all expressions of the form <2-digit-number>s
        RAW_PATTERN_MS matches all expressions of the form <3-digit-number>ms
        """
        RAW_PATTERN_REQUEST = r'!\w+((\w+|<@\d+>))+'  # \w represents any alphanumerical character eg. '!start @player1 @player2' is matched, but not 'start player'. Also be sure to not exceed one blank space between words
        RAW_PATTERN_COMMAND = r'!\w+'
        RAW_PATTERN_ARGUMENT = r'(\w|\d)+'  # an argument can be a mention or any alphanumerical sequence of characters
        RAW_PATTERN_INSTRUCTION = r'<@\d+> must play \(he has (\d{1,2}h)?(\d{1,2}min)?(\d{1,2}s)?(\d{1,3}ms)? left\)'
        #RAW_PATTERN_MOVE = r'(\w|\d+){2}-(\w|\d+){2}'
        RAW_PATTERN_MOVE = r'[a-zA-Z]{0,1}[0-9]{1,2}'
        RAW_PATTERN_MENTION = r'<@\d+>'
        RAW_PATTERN_WIN = r'<@\d+> won'
        RAW_PATTERN_EXCEED_TIME = r'<@\d+> has exceeded his allowed time'
        RAW_PATTERN_START_MESSAGE_INFORMATION = r'\*\*\[([-A-Za-z0-9_ ]+)\]\*\* ([-A-Za-z0-9_ ]+)'

        RAW_PATTERN_HOUR = r'\d{1,2}h'
        RAW_PATTERN_MIN = r'\d{1,2}min'
        RAW_PATTERN_SEC = r'\d{1,2}s'
        RAW_PATTERN_MS = r'\d{1,3}ms'

    #@verify(UNIQUE)
    class EnumCompiledPattern(Enum):
        """Enumeration that stores the (compiled) patterns enumerated in EnumPattern."""
        PATTERN_REQUEST = re.compile(EnumPattern.RAW_PATTERN_REQUEST.value)
        PATTERN_COMMAND = re.compile(EnumPattern.RAW_PATTERN_COMMAND.value)
        PATTERN_ARGUMENT = re.compile(EnumPattern.RAW_PATTERN_ARGUMENT.value)
        PATTERN_INSTRUCTION = re.compile(EnumPattern.RAW_PATTERN_INSTRUCTION.value)
        #PATTERN_MOVE = re.compile(EnumPattern.RAW_PATTERN_MOVE.value) => Now compiled in game instances
        PATTERN_MENTION = re.compile(EnumPattern.RAW_PATTERN_MENTION.value)
        PATTERN_WIN = re.compile(EnumPattern.RAW_PATTERN_WIN.value)
        PATTERN_EXCEED_TIME = re.compile(EnumPattern.RAW_PATTERN_EXCEED_TIME.value)
        PATTERN_START_MESSAGE_INFORMATION = re.compile(EnumPattern.RAW_PATTERN_START_MESSAGE_INFORMATION.value)

        PATTERN_HOUR = re.compile(EnumPattern.RAW_PATTERN_HOUR.value)
        PATTERN_MIN = re.compile(EnumPattern.RAW_PATTERN_MIN.value)
        PATTERN_SEC = re.compile(EnumPattern.RAW_PATTERN_SEC.value)
        PATTERN_MS = re.compile(EnumPattern.RAW_PATTERN_MS.value)

        @classmethod
        def analyse_start_message(cls, message:str):
            """Method that decomposes the start message"""
            return cls.PATTERN_START_MESSAGE_INFORMATION.value.findall(message)

        @classmethod
        def get_hour(cls, message:str):
            """Get the hours from the message"""
            return cls.PATTERN_HOUR.value.findall(message)

        @classmethod
        def get_minute(cls, message: str):
            """Get the minutes from the message"""
            return cls.PATTERN_MIN.value.findall(message)

        @classmethod
        def get_second(cls, message: str):
            """Get the seconds from the message"""
            return cls.PATTERN_SEC.value.findall(message)

        @classmethod
        def get_millisecond(cls, message: str):
            """Get the millisecond from the message"""
            return cls.PATTERN_MS.value.findall(message)

        @classmethod
        def is_instruction_message(cls, message:str):
            """Method that states whether a message contains starting instructions"""
            return cls.PATTERN_INSTRUCTION.value.fullmatch(message)
