import random
from math import inf

from discord_interface.player.model.basic_player import BasicPlayer
from discord_interface.player.model.player import *
from discord_interface.utils.mytime import Time

class TextualRandomAI(BasicPlayer):


    def my_plays(self, game_history, time_left=inf, opponent_time_left=inf):

        """
        this method must return any move of self.textual_legal_moves()

        game_history : contains the list of textual moves (any player) from the begining of the match
        uses it if you need to reconstruct or continue the game from textual moves ; otherwise you can use the methods of the attribute self.game of the as below
        notably, a tensor representing the game state is availiable from self.game.get_numpy_board(). The other useful methods are used below:

        note: time_left & opponent_time_left are in seconds


        """

        if self.game.get_current_player() == 1:
            player = 1
            opponent = 0
        else:
            player = 0
            opponent = 1

        winning = []
        losing = []
        draws = []
        others = []


        for move in self.game.textual_legal_moves():
            self.game.textual_plays(move)

            if self.game.ended():
                if self.game.winner == player:
                    winning.append(move)
                elif self.game.winner == opponent:
                    losing.append(move)
                else:
                    draws.append(move)
            else:
                others.append(move)

            self.game.undo()

        if winning:
            return random.choice(winning)
        if others:
            return random.choice(others)
        if draws:
            return random.choice(draws)
        if losing:
            return random.choice(losing)