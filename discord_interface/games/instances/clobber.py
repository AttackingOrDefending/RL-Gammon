import numpy as np
from discord_interface.games.mygame import *

taille_par_defaut = 10#4#


class Clobber(Game):
    CORRESPONDENCE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', #
                      12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S'}                                    #
                                                                                                                        #
    ANTI_CORRESPONDENCE = {'C': 2, 'H': 7, 'S': 18, 'J': 9, 'E': 4, 'F': 5, 'O': 14, 'B': 1, 'R': 17, 'K': 10, 'I': 8,  #
                           'G': 6, 'L': 11, 'N': 13, 'Q': 16, 'P': 15, 'D': 3, 'A': 0, 'M': 12}                         #
                                                                                                                        #
    def __init__(self, taille=taille_par_defaut):
        super().__init__(name = 'clobber', rules = 'unknown...', move_keywords=["test"])

        self.dtype = 'int8'  # 'float32' # 'float'

        self.longueur_max = taille * taille  # * 4 * 5
        self.longueur_moyenne = self.longueur_max  # /2

        self.taille = taille

        self.init()

    def ended(self):
        return self.fini

    def terminate(self, winner):
        self.fini = True

        self.winner = winner

    def init(self):

        self.coups_licites = {}

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.plateau = np.zeros((self.taille, self.taille, 2), dtype=self.dtype)

        self.historique = []

        for i in range(self.taille):
            for j in range(self.taille):

                if (i + j) % 2 == 0:
                    self.pieces_blanc.add((i, j))
                    self.plateau[i, j, 0] = 1
                else:
                    self.pieces_noir.add((i, j))
                    self.plateau[i, j, 0] = -1

        self.calcul_coups_licites_init()

        self.fini = False
        self.winner = None
        self.gagnant = None

        self.actu_couleur_plateau()

    def get_current_player(self):
        if self.blancJoue:
            return 1
        else:
            return 0

    def undo(self):
        l1, l2, self.blancJoue = self.historique.pop()

        if self.blancJoue:

            for v in self.voisins_a_prendre(*l2, True):
                self.coups_licites[True].discard((l2, v))
                self.coups_licites[False].discard((v, l2))

            self.plateau[(*l1, 0)] = 1
            self.pieces_blanc.add(l1)

            self.pieces_blanc.discard(l2)

            self.plateau[(*l2, 0)] = -1
            self.pieces_noir.add(l2)

            for v in self.voisins_a_prendre(*l1, True):
                self.coups_licites[True].add((l1, v))
                self.coups_licites[False].add((v, l1))

            for v in self.voisins_a_prendre(*l2, False):
                self.coups_licites[False].add((l2, v))
                self.coups_licites[True].add((v, l2))

        else:

            for v in self.voisins_a_prendre(*l2, False):
                self.coups_licites[False].discard((l2, v))
                self.coups_licites[True].discard((v, l2))

            self.plateau[(*l1, 0)] = -1
            self.pieces_noir.add(l1)

            self.pieces_noir.discard(l2)

            self.plateau[(*l2, 0)] = 1
            self.pieces_blanc.add(l2)

            for v in self.voisins_a_prendre(*l1, False):
                self.coups_licites[False].add((l1, v))
                self.coups_licites[True].add((v, l1))

            for v in self.voisins_a_prendre(*l2, True):
                self.coups_licites[True].add((l2, v))
                self.coups_licites[False].add((v, l2))

        self.fini = False
        self.winner = None
        self.gagnant = None

        self.actu_couleur_plateau()

    def reset(self):
        self.init()

    def plays(self, move):
        self.jouer(*move)

    def valid_actions(self):
        return self.coupsLicites()

    def jouer(self, l1, l2):

        if (l1, l2) not in self.coupsLicites():
            print('ouille ...')
            raise Exception('non legal')

        if self.blancJoue:

            for v in self.voisins_a_prendre(*l1, True):
                self.coups_licites[True].discard((l1, v))
                self.coups_licites[False].discard((v, l1))
                # print('b-',(l1, v))

            for v in self.voisins_a_prendre(*l2, False):
                self.coups_licites[False].discard((l2, v))
                self.coups_licites[True].discard((v, l2))
                # print('n-', (l2, v))

            self.plateau[(*l1, 0)] = 0

            self.pieces_blanc.discard(l1)

            self.pieces_blanc.add(l2)

            self.plateau[(*l2, 0)] = 1
            self.pieces_noir.discard(l2)

            for v in self.voisins_a_prendre(*l2, True):
                self.coups_licites[True].add((l2, v))
                self.coups_licites[False].add((v, l2))
                # print('b+', (l2, v))





        else:
            for v in self.voisins_a_prendre(*l1, False):
                self.coups_licites[False].discard((l1, v))
                self.coups_licites[True].discard((v, l1))
                # print('n-', (l1, v))

            for v in self.voisins_a_prendre(*l2, True):
                self.coups_licites[True].discard((l2, v))
                self.coups_licites[False].discard((v, l2))
                # print('b-', (l2, v))

            self.plateau[(*l1, 0)] = 0

            self.pieces_noir.discard(l1)

            self.pieces_noir.add(l2)

            self.plateau[(*l2, 0)] = -1
            self.pieces_blanc.discard(l2)

            for v in self.voisins_a_prendre(*l2, False):
                self.coups_licites[False].add((l2, v))
                self.coups_licites[True].add((v, l2))
                # print('n+', (l2, v))

        self.historique.append((l1, l2, self.blancJoue))

        self.blancJoue = not self.blancJoue

        self.actu_couleur_plateau()

        if not len(self.coups_licites[self.blancJoue]):
            self.fini = True
            if self.blancJoue:
                self.winner = 0
                self.gagnant = 'noir'
            else:
                self.winner = 1
                self.gagnant = 'blanc'

    def voisins_a_prendre(self, i, j, blanc_joue):
        if blanc_joue:
            if i < self.taille - 1 and self.plateau[i + 1, j, 0] == -1:
                yield (i + 1, j)

            if i > 0 and self.plateau[i - 1, j, 0] == -1:
                yield (i - 1, j)

            if j > 0 and self.plateau[i, j - 1, 0] == -1:
                yield (i, j - 1)

            if j < self.taille - 1 and self.plateau[i, j + 1, 0] == -1:
                yield (i, j + 1)
        else:
            if i < self.taille - 1 and self.plateau[i + 1, j, 0] == 1:
                yield (i + 1, j)

            if i > 0 and self.plateau[i - 1, j, 0] == 1:
                yield (i - 1, j)

            if j > 0 and self.plateau[i, j - 1, 0] == 1:
                yield (i, j - 1)

            if j < self.taille - 1 and self.plateau[i, j + 1, 0] == 1:
                yield (i, j + 1)

    def calcul_coups_licites_init(self):

        l = set()

        for i, j in self.pieces_blanc:

            if i < self.taille - 1 and self.plateau[i + 1, j, 0] == -1:
                l.add(((i, j), (i + 1, j)))

            if i > 0 and self.plateau[i - 1, j, 0] == -1:
                l.add(((i, j), (i - 1, j)))

            if j > 0 and self.plateau[i, j - 1, 0] == -1:
                l.add(((i, j), (i, j - 1)))

            if j < self.taille - 1 and self.plateau[i, j + 1, 0] == -1:
                l.add(((i, j), (i, j + 1)))

        self.coups_licites[True] = l

        l = set()

        for i, j in self.pieces_noir:

            if i < self.taille - 1 and self.plateau[i + 1, j, 0] == 1:
                l.add(((i, j), (i + 1, j)))

            if i > 0 and self.plateau[i - 1, j, 0] == 1:
                l.add(((i, j), (i - 1, j)))

            if j > 0 and self.plateau[i, j - 1, 0] == 1:
                l.add(((i, j), (i, j - 1)))

            if j < self.taille - 1 and self.plateau[i, j + 1, 0] == 1:
                l.add(((i, j), (i, j + 1)))

        self.coups_licites[False] = l

    def coupsLicites(self):
        return list(self.coups_licites[self.blancJoue])

    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 1] = -np.ones((self.taille, self.taille), dtype=self.dtype)

    def reorientation(self, coup):

        taille = 10

        lettre = coup[0]
        chiffre = coup[1:]

        lettre = Clobber.ANTI_CORRESPONDENCE[lettre]
        chiffre = int(chiffre)

        return chiffre - 1, lettre #taille - lettre

    def action_to_string(self, object):
        i, j = object

        i1, j1 = i
        i2, j2 = j

        return Clobber.CORRESPONDENCE[j1] + str(i1 + 1) + '-' + Clobber.CORRESPONDENCE[j2] + str(i2 + 1)

    def string_to_action(self, string):

        cp1, cp2 = string.split('-')

        return (self.reorientation(cp1), self.reorientation(cp2))


    def show_game(self):                                                                                                #
        desc = ''                                                                                                       #
        N, D, K = self.plateau.shape                                                                                    #
                                                                                                                        #
                                                                                                                        #
        for n in range(N):                                                                                              #
            for d in range(D):                                                                                          #
                desc = desc + f'{self.plateau[n, d, 0].__str__():^5}' + " "                                             #
            desc += "\n"                                                                                                #                                                                                         #
                                                                                                                        #
        return desc                                                                                                     #


    def get_numpy_board(self, copy=True):# copy=False only if you do not modify the numpy board
        if copy:
            return np.copy(self.plateau)
        else:
            return self.plateau