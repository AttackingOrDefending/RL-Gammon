import json
import os.path
from json import JSONDecodeError

from discord_interface.games.mygame import Game

class GTP_Game(Game):

    config_filename_gtp = 'gtp.conf'

    def __init__(self, nom=None, rules='standard', move_keywords=['pass',]):


        """config = self.get_config()

        if nom is None:
            nom = config['current_game']

        self.game_program = config['gtp_game_program'][nom.lower().capitalize()]"""
        #raise Exception()
        super().__init__(name=nom, rules=rules, move_keywords=self.modify_move_keywords(move_keywords))


        """boardsize
clear_board

play "move"
genmove > move
undo
time_left "color" "time"

final_score "color"[+"score"]

move = "color" "vertex"
"""


    def modify_move_keywords(self, move_keywords):
        return move_keywords

    def get_config(self):
        if not os.path.exists(GTP_Game.config_filename_gtp):

            #Amazons, Arimaa, Ataxx, Brazilian Draughts, Breakthrough, Canadian Draughts, Chinese Checkers, Chinese Chess, Chinese Dark Chess, Clobber, Connect6, Dice Shogi, Dots and Boxes, EinStein Würfelt Nicht, Go (9×9), Havannah (8×8), Havannah (10×10), Hex (11×11), Hex (13×13), Hex (19×19), HoneyMoon Bridge, International Draughts, Kyoto Shogi, Lines of Action, Mahjong, Mini Shogi, Nonogram, Othello (8×8), Othello (10×10), Othello (16×16), Outer-Open Gomoku, Santorini, Shobu, Surakarta, Sylver Coinage, and Kriegspiel.

            config_dic = {
                'current_game' : 'Clobber',
                'gtp_player_program' : {'Clobber' : 'my_clobber_player_program'},
                'gtp_game_program' : {'Clobber' : 'my_clobber_game_program'},
            }

            with open(GTP_Game.config_filename_gtp, 'w') as f:
                try:
                    f.write(json.dumps(config_dic))
                except JSONDecodeError:
                    print("ERROR JSON")
        else:

            with open(GTP_Game.config_filename_gtp, 'r') as f:
                try:
                    config_dic = json.load(f)
                except JSONDecodeError:
                    print("ERROR JSON")

        return config_dic


    def ended(self):
        return self.jeu.fini

    def actu_winner(self):

        if self.ended():
            raise NotImplementedError()
            #self.winner =
        else:
            self.winner = None



    def reset(self):
        self.history = []


    def get_current_player(self):
        if self.jeu.blancJoue:
            return 1
        else:
            return 0

    def plays(self, move):
        # print(len(self.jeu.historique))
        if move not in self.valid_actions():
            print(move)
            print(self.jeu.coupsLicites())
            print([(a, b) for a, b, *_ in self.jeu.historique])
        assert move in self.valid_actions()
        self.jeu.jouer(*move)
        self.actu_winner()
        # print(len(self.jeu.historique))
        # print(self.jeu.plateau[:,:, 0])
        self.plateau = self.jeu.plateau
        self.fini = self.jeu.fini


    def undo(self):
        self.jeu.undo()
        self.actu_winner()
        self.plateau = self.jeu.plateau
        self.fini = self.jeu.fini

    def valid_actions(self):
        return self.jeu.coupsLicites()

    def action_to_string(self, object):
        return str(object)

    def string_to_action(self, string):
        return string

    def show_game(self):
        """"""

    def terminate(self, winner):
        self.jeu.fini = True
        self.winner = winner

    def get_numpy_board(self, copy=True):# copy=False only if you do not modify the numpy board
        raise Exception('GTP Game: no tensor representation!')