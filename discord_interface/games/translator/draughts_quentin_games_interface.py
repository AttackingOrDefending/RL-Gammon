import numpy as np

from discord_interface.games.mygame import Game
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord
from discord_interface.games.translator.traductor_methods import correspondance_action_python_ludii


class DraughtsInterfaceJeuDiscord(InterfaceJeuDiscord):

    def plays(self, move):

        if move not in self.valid_actions():
            print(move)
            print(self.valid_actions())
            print(self.jeu.coupsLicites())
            print(type(move))
            print(type(self.jeu.coupsLicites()[0]))
            print(type(move[0]))
            print(type(self.jeu.coupsLicites()[0][0]))
            print(move in self.jeu.coupsLicites())
            print('H:',[(a,b) for a, b, *_ in self.jeu.historique])
        assert move in self.valid_actions()

        self.historique_partial_moves.append(list(self.partial_moves))
        self.partial_moves.append(move)

        if self.partial_moves in self.valid_group_actions():

            move = None

            for cp in self.jeu.coupsLicites():
                if self.traduction(cp) == self.partial_moves:
                    move = cp

            self.jeu.jouer(*move)
            self.actu_winner()

            self.plateau = self.jeu.plateau
            self.fini = self.jeu.fini

            self.partial_moves = []


    def __init__(self, nom, Jeu, params=None, rules='standard', move_keywords=[]):
        super().__init__(nom = nom, Jeu=Jeu, params=params, rules = rules, move_keywords=move_keywords)

        self.partial_moves = []
        self.historique_partial_moves = []


    def undo(self):

        if self.partial_moves:
            self.partial_moves.pop()
        else:
            self.jeu.undo()
            self.actu_winner()
            self.plateau = self.jeu.plateau
            self.fini = self.jeu.fini

            self.partial_moves = self.historique_partial_moves.pop()

    def traduction(self, coup):
        if not isinstance(coup[1][0], int):
            L = []
            action = correspondance_action_python_ludii(self.name.lower().replace(' ', '-'), self.jeu, (coup[0], coup[1][0][0]))
            L.append(action)
            for c in range(len(coup[1][0]) - 1):
                action = correspondance_action_python_ludii(self.name.lower().replace(' ', '-'), self.jeu, (coup[1][0][c], coup[1][0][c + 1]))
                L.append(action)

            return L
        else:
            return [correspondance_action_python_ludii(self.name.lower().replace(' ', '-'), self.jeu, coup)]

    def valid_group_actions(self):
        return [self.traduction(cp) for cp in self.jeu.coupsLicites()]

    def valid_actions(self):

        L = []

        for cps in self.valid_group_actions():


            if len(cps) > 1 and cps[:len(self.partial_moves)] == self.partial_moves:
                L.append(cps[len(self.partial_moves)])
            if len(cps) == 1 and len(self.partial_moves) == 0:
                L.append(cps[0])

        return list(set(L))

    def coupsLicites(self):
        return self.valid_actions()


    def action_to_string(self, object):
        return object


    def string_to_action(self, string):
        return string

    def super_string_to_action(self, string):
        return super().string_to_action(string)