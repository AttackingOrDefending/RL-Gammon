
from discord_interface.games.mygame import *


class FreeGame(Game):

    EXTRA_MOVE_KEYWORDS = []

    def __init__(self):
        super().__init__(name = 'Free_game', rules = 'Free moves.', move_keywords=["end"]+FreeGame.EXTRA_MOVE_KEYWORDS)

        self.init()

        self.actions_validation_enabled = False

    def ended(self):
        return self.fini

    def terminate(self, winner):
        self.fini = True

        self.winner = winner

    def init(self):

        self.historique = []

        self.fini = False
        self.winner = None
        self.gagnant = None


    def undo(self):
        self.historique.pop()
        self.fini = False
        self.winner = None

    def reset(self):
        self.init()

    def plays(self, move):
        if move == 'end' and self.historique[-1] == 'end':
            self.fini = True
            self.winner = 'unknown'
        self.historique.append(move)



    def valid_actions(self):
        return  [chr(n + ord('A'))+str(c) for n in range(26) for c in range(25)] + [chr(n1 + ord('A'))+str(c1)+'-'+chr(n2 + ord('A'))+str(c2) for n1 in range(26) for c1 in range(25) for n2 in range(26) for c2 in range(25)] + ['end']
        #return ['A1', 'A2', 'A3', 'A4','end']

    def action_to_string(self, object):
        return object

    def string_to_action(self, string):
        return string

    def get_current_player(self):
        return len(self.historique) % 2

    def show_game(self):
        """"""


    def get_numpy_board(self, copy=True):# copy=False only if you do not modify the numpy board
        raise Exception('Empty Game: no tensor representation!')