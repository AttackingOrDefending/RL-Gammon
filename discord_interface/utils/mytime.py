from discord_interface.utils.pattern_enum import EnumCompiledPattern
from discord.utils import utcnow
from datetime import timedelta, datetime
from typing import Any

#Metadata
__author__ = "Oscar PERIANAYAGASSAMY"
__copyright__ = "Copyright 2025, Interface Computer Olympiad"
__status__ = "Development"
__email__ = "oscar.perianayagassamy@dauphine.eu"



if __name__ != '__main__':
#type TimeRepresentation = Time

    class NegativeError(Exception):
        """Class to represent whenever an entry is supposed to be non-negative, but appears to be negative. """

        def __init__(self, message:str) -> None:
            super().__init__(message)



    class Time:
        """class to represent time with 4 integer attributes (hour, minute, second, millisecond) being displayed using the format h:mm:ss

        ATTRIBUTES
        ----------
        hour : int
        minute : int
        second : int
        millisecond : int
        """
        INF = float("inf") #Quantity that represents the positive infinity, all Time instance are less than INF. Used for comparison purposes.

        def __init__(self, hour:int = 0, minute:int = 0, second:int = 0, millisecond:int = 0) -> None:
            self.hour = hour
            self.minute = minute
            self.second = second
            self.millisecond = millisecond


        def __str__(self) -> str:
            """Displays a Time object using the format h:mm:ss"""
            if self == Time():
                return f'00s'
            #return f'{f'{self.hour}h' if self.hour > 0 else ''}{f'{self.minute:02}min' if self.minute > 0 else ''}{f'{self.second:02}s' if self.second > 0 else ''}{f'{self.millisecond:02}ms' if self.hour == 0 and self.minute == 0 and self.second == 0 else ''}'

            H = f'{self.hour}h' if self.hour > 0 else ''
            M = f'{self.minute:02}min' if self.minute > 0 else ''
            S = f'{self.second:02}s' if self.second > 0 else ''
            m = f'{self.millisecond:02}ms' if self.hour == 0 and self.minute == 0 and self.second == 0 else ''
            return H+M+S+m

        def __add__(self, t: Any):# -> TimeRepresentation:
            """Add two Time objects

            PARAMETERS
            ----------
            t : Object

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            mytime.Time
                new Time object which is the result of adding both time representations
            """
            if type(t) == type(self):

                #Convert the Time instances to milliseconds
                raw_time_self = self.hour*60*60*1000 + self.minute*60*1000 + self.second*1000 + self.millisecond
                raw_time_t = t.hour*60*60*1000 + t.minute*60*1000 + t.second*1000 + t.millisecond

                #Add up the results
                ret_ms = raw_time_self + raw_time_t

                #Convert back to hour, min, sec, millisecond
                new_millisecond = ret_ms%1000
                ret_sec = ret_ms//1000
                new_second = ret_sec%60
                ret_min = ret_sec//60
                new_minute = ret_min%60
                new_hour = ret_min//60

                return Time(new_hour, new_minute, new_second, new_millisecond)

            else:
                raise TypeError(f'{t} and {self} have to be of type Time')

        def __sub__(self, t: Any):# -> TimeRepresentation:
            """Subtract two Time objects

            If the subtraction of both objects yields a negative output, round it up to 00:00:00:000.

            PARAMETERS
            ----------
            t : Object
                The object to be subtracted

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            mytime.Time
                Object which is the result of subtracting both time representions (self - t)
            """
            if type(t) == type(self):

                #Convert the Time instances to milliseconds
                raw_time_self = self.hour * 60 * 60 * 1000 + self.minute * 60 * 1000 + self.second * 1000 + self.millisecond
                raw_time_t = t.hour * 60 * 60 * 1000 + t.minute * 60 * 1000 + t.second * 1000 + t.millisecond

                #Substract the results
                ret_ms = raw_time_self - raw_time_t

                # The result can't be negative in our representation, so return zero-filled Time instance in this setting
                if ret_ms < 0:
                    return Time()

                #Otherwise convert back to hour, min, second, millisecond
                new_millisecond = ret_ms % 1000
                ret_sec = ret_ms // 1000
                new_second = ret_sec % 60
                ret_min = ret_sec // 60
                new_minute = ret_min % 60
                ret_hour = ret_min // 60
                new_hour = ret_hour % 60

                return Time(new_hour, new_minute, new_second, new_millisecond)

            else:
                raise TypeError(f'{t} and {self} have to be of type Time')

        def __ge__(self, t: Any) -> bool:
            """Greater or Equal checking

            PARAMETERS
            ----------
            t : Object
                The object to be compared to.

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            bool
            """
            #If t is Time.INF, it is mandatory that self is less than t...
            if t == Time.INF:
                return False

            #...Otherwise, if t and self have type Time, compare their millisecond time representation...
            elif type(t) == type(self):
                return self.to_milliseconds() >= t.to_milliseconds()

            #...Otherwise, no comparison is possible.
            else:
                raise TypeError(f'{t} and {self} have to be of type Time')


        def __le__(self, t: Any) -> bool:
            """Less or Equal checking

            PARAMETERS
            ----------
            t : Object
                The object to be compared to.

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            bool
            """
            # If t is Time.INF, it is mandatory that self is less than t...
            if t == Time.INF:
                return True

            # ...Otherwise, if t and self have type Time, compare their millisecond time representation...
            elif type(t) == type(self):
                return self.to_milliseconds() <= t.to_milliseconds()

            # ...Otherwise, no comparison is possible.
            else:
                raise TypeError(f'{t} and {self} have to be of type Time')

        def __lt__(self, t: Any) -> bool:
            """Less checking

            PARAMETERS
            ----------
            t : Object
                The object to be compared to.

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            bool
            """
            # If t is Time.INF, it is mandatory that self is less than t...
            if t == Time.INF:
                return True

            # ...Otherwise, if t and self have type Time, compare their millisecond time representation...
            if type(t) == type(self):
                return self.to_milliseconds() < t.to_milliseconds()

            # ...Otherwise, no comparison is possible.
            else:
                raise TypeError(f'{t} and {self} have to be of type Time')

        def __gt__(self, t: Any) -> bool:
            """Greater checking

            PARAMETERS
            ----------
            t : Object
                The object to be compared to.

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            bool
            """
            # If t is Time.INF, it is mandatory that self is less than t...
            if t == Time.INF:
                return False

            # ...Otherwise, if t and self have type Time, compare their millisecond time representation...
            if type(t) == type(self):
                return self.to_milliseconds() > t.to_milliseconds()

            # ...Otherwise, no comparison is possible.
            else:
                raise TypeError(f'{t} and {self} have to be of type Time')



        def __eq__(self, t: Any) -> bool:
            """Equality checking

            PARAMETERS
            ----------
            t : Object
                The object to be compared to.

            RAISES
            ------
            TypeError
                Raises an error whenever the argument t is not of type Time.

            RETURNS
            -------
            bool
            """
            # If t is Time.INF, it is mandatory that self is less than t...
            if t == Time.INF:
                return False

            # ...Otherwise, if t and self have type Time, compare each of their fields one-to-one...
            if type(t) == type(self):
                return t.hour == self.hour and t.minute == self.minute and t.second == self.second and t.millisecond == self.millisecond

            # ...Otherwise, no comparison is possible.
            else:
                raise TypeError(f'{t} and {self} have to be of type Time')

        def set_time(self, date:datetime = None) -> None:
            """Set time using a datetime object

             PARAMETERS
             ----------
            date : datetime.datetime
                Object from which we want to set the time with contains hours, minutes, seconds and microseconds fields

            RETURNS
            -------
            None
            """
            self.hour = date.hour
            self.minute = date.minute
            self.second = date.second
            self.millisecond = date.microsecond//1000

        def to_seconds(self) -> int:
            """Translating to seconds

            RETURNS
            -------
            int
                The representation of the current time representation in seconds.
            """
            return self.hour*60*60 + self.minute*60 + self.second

        def to_milliseconds(self) -> int:
            """Translating to milliseconds

            RETURNS
            -------
            int
                The representation of the current time representation in milliseconds.
            """
            return self.hour*60*60*1000 + self.minute*60*1000 + self.second*1000 + self.millisecond

        # M A L AIR BIZARRE ; pas utiliser alors commenté
        '''def to_timedelta(self) -> timedelta:
            """Translating to timedelta

            RETURNS
            -------
            datetime.timedelta
                The representation of the current time representation in timedelta.
            """
            return timedelta(hours=self.hour, minutes=self.minute, seconds=self.second, microseconds=self.millisecond*1000)'''

        def get_logformated(self) -> str:
            """Get the time representation for logging purposes

            In order to store the moves that are played in a game, we use the following format for the time : hh:mm.
            The reason is that the game state logs are stored in a directory already containing the date.
            Also, no need for further precision because no two games can be instantiated during the same minute.

            RETURNS
            -------
            str
                hh:mm time representation
            """
            if self.hour==0 and self.minute==0 and self.second==0 and self.millisecond > 0:
                return f"{self.hour:02}-{self.minute:02}-{1:02}"
            else:
                return f"{self.hour:02}-{self.minute:02}-{self.second:02}"

        @classmethod
        def now(cls):# -> TimeRepresentation:
            """Static method to get the current time encapsulated in a Time object

            RETURNS
            -------
            mytime.Time
            """
            date = datetime.now()
            return cls(date.hour, date.minute, date.second, date.microsecond//1000)

        @classmethod
        def seconds_to_Time(cls, seconds: int):# -> TimeRepresentation:
            """Get the time representation from seconds

            PARAMETERS
            ----------
            seconds : int
                Number of seconds from which we want to create a Time object.

            RAISES
            ------
            Negative Error
                If the parameter is negative, impossible to create a Time instance in our paradigm.

            RETURNS
            -------
            mytime.Time
                Time representation associated with the input.
            """
            if seconds < 0:
                raise NegativeError("Time should be positive or zero")
            h = seconds//3600
            m = (seconds%3600)//60
            s = (seconds%3600)%60
            return cls(hour=h, minute=m, second=s)

        @classmethod
        def string_to_Time(cls, string: str):# -> TimeRepresentation:
            """Get the time representation from a string

            PARAMETERS
            ----------
            string : str
                String that represents a time object (e.g. 5min)

            RETURNS
            -------
            mytime.Time
                Time representation associated with the input.
            """
            hour = EnumCompiledPattern.get_hour(string)
            hour = int(hour[0][:2].replace('h','')) if hour else 0
            minute = EnumCompiledPattern.get_minute(string)
            minute = int(minute[0][:2].replace('m','')) if minute else 0
            second = EnumCompiledPattern.get_second(string)
            second = int(second[0][:2].replace('s','')) if second else 0
            millisecond = EnumCompiledPattern.get_millisecond(string)
            millisecond = int(millisecond[0][:3].replace('ms','').replace('m','')) if millisecond else 0

            return Time(hour=hour, minute=minute, second=second, millisecond=millisecond)

        @classmethod
        def timedelta_to_Time(cls, period: timedelta) -> Any: #-> TimeRepresentation
            """create a Time object using a timedelta object

             PARAMETERS
             ----------
            date : datetime.timedelta
                Object from which we want to set the time with contains hours, minutes, seconds and microseconds fields

            RETURNS
            -------
            mytime.Time
                The resulting Time object.
            """
            return Time.seconds_to_Time(period.seconds) + Time(millisecond=period.microseconds//1000)

        @classmethod
        def datetime_to_Time(cls, period: datetime) -> Any:  # -> TimeRepresentation
            """create a Time object using a datetime object

             PARAMETERS
             ----------
            date : datetime.datetime
                Object from which we want to set the time with contains hours, minutes, seconds and microseconds fields

            RETURNS
            -------
            mytime.Time
                The resulting Time object.
            """
            return Time(hour=period.hour, minute=period.minute, second=period.second, millisecond=period.microsecond // 1000)

        @classmethod
        def discord_utc_now(cls) -> datetime:
            return utcnow()




    class Timer:
        """Class to represent a timer

        ATTRIBUTES
        ----------
        start_time : mytime.Time
            The time when the timer was started.
        relative_start_time : mytime.Time
            The last time the timer was resumed.
        """

        def __init__(self, start_time: Time=None, relative_start_time: Time=None) -> None:
            self.start_time = start_time
            self.relative_start_time = relative_start_time

        def start(self) -> None:
            """Start the timer

            Function that initializes the timer by setting both start_time and relative_start_time at the same time.

            RETURNS
            -------
            None
            """
            self.start_time = Time.now()
            self.relative_start_time = self.start_time

        def get_time(self) -> Time:
            """Get the Time since the starting time

            RETURNS
            -------
            mytime.Time
                Result of subtracting the start time to the the current time
            """
            return Time.now() - self.start_time

        def stop(self) -> Time:
            """Stop the timer

            Method that stops the timer, i.e. retrieves the time elapsed since the relative starting time (last stop) and updates this quantity.

            RETURNS
            -------
            mytime.Time
                The result of now - last_stop
            """
            time = Time.now()
            ret = time - self.relative_start_time
            self.relative_start_time = time
            return ret

        def resume(self) -> None:
            """Resume the timer

            Method that resumes the timer by updating the relative starting time.

            RETURNS
            -------
            None
            """
            self.relative_start_time = Time.now()

        def __str__(self) -> str:
            """String representation of the timer

            RETURNS
            -------
            str
            """
            return f'starting time : {self.start_time}, last stop : {self.relative_start_time}'
