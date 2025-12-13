import numpy as np


taille_par_defaut = 8

class Lines_of_action_olympiad:

    def actu_nb_repet_plateau(self):
        self.plateau[:, :, 2] = self.nb_repet * np.ones((self.taille, self.taille), dtype=self.dtype)

    def init(self):

        assert self.borne is None

        self.nb_repet = 0
        self.panels = 3

        ###

        self.nb_repet=0
        self.hash_old_positions = []

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.blancJoue = False

        self.tour = 0

        self.plateau = np.zeros((self.taille, self.taille, self.panels), dtype=self.dtype)

        self.init_board()

        self.historique = []

        self.fini = False
        self.gagnant = None

        self.actu_couleur_plateau()
        self.calcul_coups_licites()

        ###

        self.actu_nb_repet_plateau()

    def undo(self):
        self.tour -= 1

        if self.hash_old_positions:
            self.hash_old_positions.pop()

        self.fini = False
        self.gagnant = None

        bj = self.blancJoue

        (i, j), (a, b), self.blancJoue, prise, self.coups_licites, self.nb_repet = self.historique.pop()

        if self.blancJoue:
            self.plateau[i,j,0] = 1
            if prise:
                self.plateau[a,b,0] = -1
                self.pieces_noir.add((a,b))
            else:
                self.plateau[a,b,0] = 0

            self.pieces_blanc.remove((a,b))
            self.pieces_blanc.add((i,j))
        else:
            self.plateau[i,j,0] = -1
            if prise:
                self.plateau[a,b,0] = 1
                self.pieces_blanc.add((a,b))
            else:
                self.plateau[a,b,0] = 0

            self.pieces_noir.remove((a,b))
            self.pieces_noir.add((i,j))

        if not bj == self.blancJoue:
            self.actu_couleur_plateau()
        self.actu_nb_repet_plateau()

        #self.calcul_coups_licites()

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1

    def hash_plateau(self):
        return self.plateau[:,:,0:self.panels-1].tobytes()

    def jouer(self, origine, destination):

        if self.repetition_max is not None:
            self.hash_old_positions.append(self.hash_plateau())

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        i, j = origine
        a, b = destination

        self.tour += 1
        prise=False

        if self.blancJoue:
            self.plateau[i,j,0] = 0
            self.plateau[a,b,0] = 1

            if destination in self.pieces_noir:
                self.pieces_noir.remove(destination)
                prise=True
            self.pieces_blanc.remove(origine)
            self.pieces_blanc.add(destination)
        else:
            self.plateau[i,j,0] = 0
            self.plateau[a,b,0] = -1

            if destination in self.pieces_blanc:
                self.pieces_blanc.remove(destination)
                prise=True
            self.pieces_noir.remove(origine)
            self.pieces_noir.add(destination)


        self.historique.append((origine, destination, self.blancJoue, prise, self.coupsLicites(), self.nb_repet))

        self.test_fini(*destination)

        self.blancJoue = not self.blancJoue

        self.calcul_coups_licites()

        if not len(self.coupsLicites()):
            self.blancJoue = not self.blancJoue
            self.calcul_coups_licites()
        else:
            self.actu_couleur_plateau()
        #print(self.hash_old_positions.count(self.hash_plateau()), self.hash_old_positions.count(self.hash_plateau()) > self.repetition_max, self.repetition_max)

        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())

        self.actu_nb_repet_plateau()

        if not self.fini and (self.borne and self.tour >= self.borne or self.repetition_max is not None and self.nb_repet > self.repetition_max):
            self.fini = True
            self.gagnant = 'nul'
            self.score = 0
            self.score2 = 0
            self.score3 = 0
            self.score4 = 0

        if self.fini:
            self.mobilite_cumule_diff_pour_nul = self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc
            self.mobilite_cumule_frac_pour_nul = (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

            if self.gagnant == 'noir':
                self.mobilite = mobilite
                self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
                self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
                self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

                self.presence_frac = (len(self.pieces_noir) + 1) / (len(self.pieces_blanc) + 1)
            elif self.gagnant == 'blanc':
                self.mobilite = -mobilite
                self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
                self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
                self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)

                self.presence_frac = -(len(self.pieces_blanc) + 1) / (len(self.pieces_noir) + 1)
            else:
                self.mobilite = 0
                self.mobilite_cumule = 0
                self.mobilite_cumule_diff = 0
                self.mobilite_cumule_frac = 0

                self.presence_frac = 0


    ####


    def __init__(self, taille=taille_par_defaut, borne=None, repetition_max=2):# 1 veut dire 3 occurence : fin !

        self.repetition_max = repetition_max

        self.dtype = self.get_board_type()

        self.borne = borne

        if borne:
            self.longueur_max = borne
        else:
            self.longueur_max = 750

        self.longueur_moyenne = self.longueur_max / 2

        self.taille = taille

        self.directions = [(i,j) for i in range(-1,2) for j in range(-1,2) if i!=0 or j!=0]

        self.axes = [(1,0),(0,1),(1,1), (1,-1)]

        self.panels_init = 2
        self.panels = self.get_panels()
        self.init()


    def get_panels(self):
        return self.panels_init

    def get_board_type(self):
        return 'int8'  # 'float32' # 'float'

    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 1] = -np.ones((self.taille, self.taille), dtype=self.dtype)


    def init_board(self):
        for i in range(1, self.taille - 1):
            self.plateau[0, i, 0] = -1
            self.plateau[-1, i, 0] = -1

            self.pieces_noir.add((0, i))
            self.pieces_noir.add((self.taille - 1, i))

            self.plateau[i, 0, 0] = 1
            self.plateau[i, -1, 0] = 1

            self.pieces_blanc.add((i, 0))
            self.pieces_blanc.add((i, self.taille-1))


    def raz(self):
        self.init()





    def test_fini(self, i, j):

        if self.blancJoue:
            blanc_fini = self.connecter(i, j,1, set()) == self.pieces_blanc
            #print('----')
            #print(self.connecter(i, j,1, set()))
            #print('---')
            k, l = list(self.pieces_noir)[0]
            noir_fini = self.connecter(k,l, -1, set()) == self.pieces_noir

        else:
            noir_fini = self.connecter(i,j,-1, set()) == self.pieces_noir
            k,l=list(self.pieces_blanc)[0]
            blanc_fini = self.connecter(k,l, 1, set()) == self.pieces_blanc
        """"""


        if noir_fini or blanc_fini:
            self.fini = True

            if noir_fini and blanc_fini:
                self.gagnant = 'nul'
                self.score = 0
                self.score2 = 0
                self.score3 = 0
                self.score4 = 0

            else:

                if noir_fini:
                    self.gagnant = 'noir'
                    self.score  = 12 + len(self.pieces_noir) - len(self.pieces_blanc)
                    self.score2 = 12 - len(self.pieces_noir) + len(self.pieces_blanc)
                    self.score3 = max(len(self.pieces_noir) - len(self.pieces_blanc), 1)
                    self.score4 = max(- len(self.pieces_noir) + len(self.pieces_blanc), 1)

                elif blanc_fini:
                    self.gagnant = 'blanc'
                    self.score  = -(12 + len(self.pieces_blanc) - len(self.pieces_noir))
                    self.score2 = -(12 - len(self.pieces_blanc) + len(self.pieces_noir))
                    self.score3 = -max(len(self.pieces_blanc) - len(self.pieces_noir), 1)
                    self.score4 = -max(- len(self.pieces_blanc) + len(self.pieces_noir), 1)
                else:
                    raise Exception('????....')

    def connecter(self, i, j, val, s):

        s.add((i,j))

        for di, dj in self.directions:

            a,b = i+di,j+dj

            if self.position_valide(a,b) and (a,b) not in s and self.plateau[a,b,0] == val:
                #print(i,j,self.plateau[i,j])
                #print(a, b, self.plateau[a, b])
                self.connecter(a,b, val, s)


        return s

    def coupsLicites(self):
        return list(self.coups_licites)

    def calcul_coups_licites(self):
        """"""
        l = []
        if self.blancJoue:
            for i,j in self.pieces_blanc:

                for di, dj in self.axes:

                    n = self.nombre_pieces(i,j, di, dj)

                    a = i + n*di
                    b = j + n*dj

                    if self.position_valide(a,b) and self.plateau[a,b,0] != 1 and not self.pieces_adverses_intermediaire(i, j, di, dj, n, 1):
                        l.append(((i,j),(a,b)))


                    a = i - n * di
                    b = j - n * dj

                    if self.position_valide(a, b) and self.plateau[a,b,0] != 1 and not self.pieces_adverses_intermediaire(i, j, -di, -dj, n, 1):
                        l.append(((i, j), (a, b)))
        else:
            for i,j in self.pieces_noir:

                for di, dj in self.axes:

                    n = self.nombre_pieces(i,j, di, dj)

                    a = i + n*di
                    b = j + n*dj

                    if self.position_valide(a,b) and self.plateau[a,b,0] != -1 and not self.pieces_adverses_intermediaire(i, j, di, dj, n, -1):
                        l.append(((i,j),(a,b)))


                    a = i - n * di
                    b = j - n * dj

                    if self.position_valide(a, b) and self.plateau[a,b,0] != -1 and not self.pieces_adverses_intermediaire(i, j, -di, -dj, n, -1):
                        l.append(((i, j), (a, b)))
        self.coups_licites = l

    def nombre_pieces(self, i,j, di, dj):
        n=1

        d = 1
        while self.position_valide(i+di*d, j+dj*d):
            if self.plateau[i + di * d, j + dj * d,0] != 0:
                n += 1
            d+=1

        d = 1
        while self.position_valide(i - di * d, j - dj * d):
            if self.plateau[i - di * d, j - dj * d,0] != 0:
                n += 1
            d += 1

        return n


    def position_valide(self, a, b):
        return 0 <= a < self.taille and 0 <= b < self.taille


    def pieces_adverses_intermediaire(self, i, j, di, dj, n, val):

        for d in range(1, n):

            if self.plateau[i+di*d, j+dj*d,0] == - val:
                return True

        return False
