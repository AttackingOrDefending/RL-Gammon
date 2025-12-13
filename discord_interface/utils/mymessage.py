from time import time

from discord import Message
from discord_interface.utils.mytime import Time

class InstructionMessage:

    def __init__(self, message: Message = None, deleted: bool = False) -> None:
        self.message = message
        self.deleted = deleted
        self.creation_time = time()
        self.last_time = Time(-1)
        self.ended_time = None

    def delete(self):
        self.deleted = True

    def reset(self):
        self.__init__()

class TimedInstructionMessage:

    def __init__(self, message: Message = None, start_time: Time = None, withdraw=True) -> None:
        self.message = message
        self.time = start_time
        self.withdraw = withdraw

    def update_time(self, message: Message=None) -> None:
        if self.withdraw:
            self.time -= Time.timedelta_to_Time(message.created_at - self.message.created_at)
        else:
            self.time += Time.timedelta_to_Time(message.created_at - self.message.created_at)

    def reset(self):
        self.__init__(withdraw=self.withdraw)