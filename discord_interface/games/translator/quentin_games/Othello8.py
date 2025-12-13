from time import time

import numpy as np

taille_par_defaut = 8

class Othello8():


    def __init__(self, taille=taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.tps_init = 0
        self.tps_jouer = 0
        self.tps_coups = 0
        self.tps_fini = 0
        self.tps_undo=0



        self.init()

        self.longueur_max = taille * taille
        self.longueur_moyenne = self.longueur_max


    def get_plateau_de_pieces(self):
        return self.plateau[:,:,0]

    def init(self):

        deb = time()


        self.coups_licites = {}

        self.coups_licites[True] = set()
        self.coups_licites[False] = set()

        self.nb_tour_avec_pass=0

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.fini = False

        self.gagnant = None

        self.blancJoue = False

        # self.plateau = [0,]*(taille**2)
        self.plateau = np.zeros((self.taille, self.taille, 2), dtype=self.dtype)

        self.plateau[(self.taille - 1) // 2, (self.taille - 1) // 2, 0] = 1
        self.plateau[(self.taille - 1) // 2 + 1, (self.taille - 1) // 2 + 1, 0] = 1

        self.plateau[(self.taille - 1) // 2, (self.taille - 1) // 2 + 1, 0] = -1
        self.plateau[(self.taille - 1) // 2 + 1, (self.taille - 1) // 2, 0] = -1


        self.actu_couleur_plateau()

        self.historique = []  # deque() # les opérations associées sont plus rapides mais le total est pourtant plus lent ???

        #self.calcul_coupsLicites()
        self.completer_coups_licites((self.taille - 1) // 2, (self.taille - 1) // 2)
        self.completer_coups_licites((self.taille - 1) // 2 + 1, (self.taille - 1) // 2 + 1)
        self.completer_coups_licites((self.taille - 1) // 2, (self.taille - 1) // 2 + 1)
        self.completer_coups_licites((self.taille - 1) // 2 + 1, (self.taille - 1) // 2)

        self.tps_init += time() - deb
        #print(self.plateau[:,:,0])




    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 1] = -np.ones((self.taille, self.taille), dtype=self.dtype)

    def undo(self,):

        deb = time()

        (i,j), self.blancJoue, extremites, coups_noir, coups_blanc, self.nb_tour_avec_pass = self.historique.pop()

        self.coups_licites[True] = coups_blanc
        self.coups_licites[False] = coups_noir

        self.actu_couleur_plateau()

        self.plateau[i,j,0] = 0

        """print(i,j)
        print(extremites)"""

        for k,l in extremites:

            """print(list(range(min(j, l), max(j, l) + 1)))
            print(list(range(min(i, k), max(k, i) + 1)))
            print()"""

            if i != k and j != l:
                if i<k:
                    lx = range(i+1,k+1)
                else:
                    lx = list(range(k, i))
                    lx.reverse()

                if j<l:
                    ly = range(j+1, l + 1)
                else:
                    ly = list(range(l, j))
                    ly.reverse()

                for a, b in zip(lx, ly):
                    self.plateau[a,b,0] = - self.plateau[a,b,0]
                    #print('>',a,b)
            elif i == k:
                for b in range(min(j + 1, l), max(j, l + 1)):
                    self.plateau[i, b, 0] = - self.plateau[i, b, 0]
                    #print('>', i, b)
            elif j == l:
                for a in range(min(i + 1, k), max(k + 1, i)):
                    self.plateau[a, j, 0] = - self.plateau[a, j, 0]
                    #print('>', a, j)

        self.fini = False

        self.gagnant = None

        #self.calcul_coupsLicites()

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1

        self.tps_undo += time() - deb



    def raz(self):

        self.init()

    def jouer(self, i, j):

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        deb = time()

        if not (i,j) in self.coupsLicites():
            raise Exception('ho, du con !')

        if self.blancJoue:
            val = 1
        else:
            val = -1

        extremites = []

        self.plateau[i,j,0] = val

        all_positions = []

        d = 1
        positions = []
        while i + d < self.taille and self.plateau[i + d, j,0] == - val:
            positions.append((i + d, j))
            d += 1

        all_positions.append(positions)
        if d > 1 and i + d < self.taille and self.plateau[i + d, j,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]
            extremites.append((i + d - 1, j))

        d = 1
        positions = []
        while j + d < self.taille and self.plateau[i, j + d,0] == - val:
            positions.append((i, j + d))
            d += 1

        all_positions.append(positions)
        if d > 1 and j + d < self.taille and self.plateau[i, j + d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i, j + d - 1))

        d = 1
        positions = []
        while i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d,0] == - val:
            positions.append((i + d, j + d))
            d += 1

        all_positions.append(positions)
        if d > 1 and i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i + d - 1, j + d - 1))

        d = 1
        positions = []
        while i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d,0] == - val:
            positions.append((i - d, j + d))
            d += 1

        all_positions.append(positions)
        if d > 1 and i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i - d + 1, j + d - 1))

        d = 1
        positions = []
        while i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d,0] == - val:
            positions.append((i + d, j - d))
            d += 1

        all_positions.append(positions)
        if d > 1 and i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i + d - 1, j - d + 1))


        d = 1
        positions = []
        while i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d,0] == - val:
            positions.append((i - d, j - d))
            d += 1

        all_positions.append(positions)
        if d > 1 and i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i - d + 1, j - d + 1))


        d = 1
        positions = []
        while i - d >= 0 and self.plateau[i - d, j,0] == - val:
            positions.append((i - d, j))
            d += 1

        all_positions.append(positions)
        if d > 1 and i - d >= 0 and self.plateau[i - d, j,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i - d + 1, j))


        d = 1
        positions = []
        while j - d >= 0 and self.plateau[i, j - d,0] == - val:
            positions.append((i, j - d))
            d += 1

        all_positions.append(positions)
        if d > 1 and j - d >= 0 and self.plateau[i, j - d,0] == val:
            for k, l in positions:
                self.plateau[k,l,0] = - self.plateau[k,l,0]

            extremites.append((i, j - d + 1))


        self.historique.append(((i,j), self.blancJoue, extremites, set(self.coups_licites[False]), set(self.coups_licites[True]), self.nb_tour_avec_pass))




        self.coups_licites[False].discard((i,j))
        self.coups_licites[True].discard((i,j))

        self.completer_coups_licites(i,j)
        for positions in all_positions:
            for k,l in positions:
                self.completer_coups_licites(k,l)

        self.nb_tour_avec_pass += 1
        self.blancJoue = not self.blancJoue

        #self.calcul_coupsLicites()

        self.test_fini()

        self.actu_couleur_plateau()

        self.tps_jouer += time() - deb

        if self.fini:
            self.mobilite_cumule_diff_pour_nul = self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc
            self.mobilite_cumule_frac_pour_nul = (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

            if self.gagnant == 'noir':
                self.mobilite = mobilite
                self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
                self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
                self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

            elif self.gagnant == 'blanc':
                self.mobilite = -mobilite
                self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
                self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
                self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)
            else:
                self.mobilite = 0
                self.mobilite_cumule = 0
                self.mobilite_cumule_diff = 0
                self.mobilite_cumule_frac = 0

    def completer_coups_licites(self, i, j):


        val0 = self.plateau[i, j, 0]
        if  val0 == 1:
            base_noir = False
            base_blanc = True
        elif val0 == -1:
            base_noir = True
            base_blanc = False

        ######
        blanc = base_blanc
        noir = base_noir
        old_val1 = val0
        old_val2 = val0

        d1 = 1
        if i + d1 < self.taille:
            val1 = self.plateau[i + d1, j, 0]
        while i + d1 < self.taille and val1 != 0:

            if val1 == 1:
                blanc = True
            if val1 == -1:
                noir = True
            d1 += 1
            if i + d1  < self.taille:
                old_val1 = val1
                val1 = self.plateau[i + d1, j, 0]

        d2 = 1
        if i - d2 >= 0:
            val2 = self.plateau[i - d2, j, 0]
        while i - d2 >= 0 and val2 != 0:
            if val2 == 1:
                blanc = True
            if val2 == -1:
                noir = True
            d2 += 1
            if i - d2 >= 0:
                old_val2 = val2
                val2 = self.plateau[i - d2, j, 0]
        #print(noir, blanc)
        if noir and blanc:
            if i + d1 < self.taille:
                if old_val1 == 1:
                    self.coups_licites[False].add((i + d1, j))
                    if (i + d1, j) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d1, j, True):
                        self.coups_licites[True].discard((i + d1, j))
                elif old_val1 == -1:
                    self.coups_licites[True].add((i + d1, j))
                    if (i + d1, j) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d1, j, False):
                        self.coups_licites[False].discard((i + d1, j))

            if i - d2 >= 0:
                if old_val2 == 1:
                    self.coups_licites[False].add((i - d2, j))
                    if (i - d2, j) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d2, j, True):
                        self.coups_licites[True].discard((i - d2, j))
                elif old_val2 == -1:
                    self.coups_licites[True].add((i - d2, j))
                    if (i - d2, j) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d2, j, False):
                        self.coups_licites[False].discard((i - d2, j))
        else:
            #print(i + d1, j)
            #print(i - d2, j)
            if i + d1 < self.taille  :

                if (i + d1, j) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d1, j, True):
                    self.coups_licites[True].discard((i + d1, j))
                if (i + d1, j) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d1, j, False):
                    self.coups_licites[False].discard((i + d1, j))

            if i - d2 >= 0  :

                if (i - d2, j) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d2, j, True):
                    self.coups_licites[True].discard((i - d2, j))
                if (i - d2, j) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d2, j, False):
                    self.coups_licites[False].discard((i - d2, j))

        ##############
        blanc = base_blanc
        noir = base_noir
        old_val1 = val0
        old_val2 = val0

        d1 = 1
        if j + d1 < self.taille:
            val1 = self.plateau[i, j + d1, 0]
        while j + d1 < self.taille and val1 != 0:
            if val1 == 1:
                blanc = True
            if val1 == -1:
                noir = True
            d1 += 1
            if j + d1 < self.taille:
                old_val1 = val1
                val1 = self.plateau[i, j + d1, 0]

        d2 = 1
        if j - d2 >= 0:
            val2 = self.plateau[i, j - d2, 0]
        while j - d2 >= 0 and val2 != 0:
            if val2 == 1:
                blanc = True
            if val2 == -1:
                noir = True
            d2 += 1
            if j - d2 >= 0:
                old_val2 = val2
                val2 = self.plateau[i, j - d2, 0]

        if noir and blanc:
            if j + d1 < self.taille:
                if old_val1 == 1:
                    self.coups_licites[False].add((i, j + d1))
                    if (i, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i, j + d1, True):
                        self.coups_licites[True].discard((i, j + d1))

                elif old_val1 == -1:
                    self.coups_licites[True].add((i, j + d1))
                    if (i, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i, j + d1, False):
                        self.coups_licites[False].discard((i, j + d1))
            if j - d2 >= 0:
                if old_val2 == 1:
                    self.coups_licites[False].add((i, j - d2))
                    if (i, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i, j - d2, True):
                        self.coups_licites[True].discard((i, j - d2))

                elif old_val2 == -1:
                    self.coups_licites[True].add((i, j - d2))
                    if (i, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i, j - d2, False):
                        self.coups_licites[False].discard((i, j - d2))
        else:
            if j + d1 < self.taille  :

                if (i, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i, j + d1, True):
                    self.coups_licites[True].discard((i, j + d1))
                if (i, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i, j + d1, False):
                    self.coups_licites[False].discard((i, j + d1))
            if j - d2 >= 0  :

                if (i, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i, j - d2, True):
                    self.coups_licites[True].discard((i, j - d2))
                if (i, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i, j - d2, False):
                    self.coups_licites[False].discard((i, j - d2))

        ##############
        blanc = base_blanc
        noir = base_noir
        old_val1 = val0
        old_val2 = val0

        d1 = 1
        if i - d1 >= 0 and j + d1 < self.taille:
            val1 = self.plateau[i - d1, j + d1, 0]
        while i - d1 >= 0 and j + d1 < self.taille and val1 != 0:
            if val1 == 1:
                blanc = True
            if val1 == -1:
                noir = True
            d1 += 1
            if i - d1 >= 0 and j + d1 < self.taille:
                old_val1 = val1
                val1 = self.plateau[i - d1, j + d1, 0]

        d2 = 1
        if i + d2 < self.taille and j - d2 >= 0:
            val2 = self.plateau[i + d2, j - d2, 0]
        while i + d2 < self.taille and j - d2 >= 0 and val2 != 0:
            if val2 == 1:
                blanc = True
            if val2 == -1:
                noir = True
            d2 += 1
            if i + d2 < self.taille and j - d2 >= 0:
                old_val2 = val2
                val2 = self.plateau[i + d2, j - d2, 0]

        if noir and blanc:
            if i - d1 >= 0 and j + d1 < self.taille:
                if old_val1 == 1:
                    self.coups_licites[False].add((i - d1, j + d1))
                    if (i - d1, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d1, j + d1,
                                                                                                         True):
                        self.coups_licites[True].discard((i - d1, j + d1))

                elif old_val1 == -1:
                    self.coups_licites[True].add((i - d1, j + d1))
                    if (i - d1, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d1,
                                                                                                          j + d1,
                                                                                                          False):
                        self.coups_licites[False].discard((i - d1, j + d1))
            if i + d2 < self.taille and j - d2 >= 0:
                if old_val2 == 1:
                    self.coups_licites[False].add((i + d2, j - d2))
                    if (i + d2, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d2, j - d2,
                                                                                                         True):
                        self.coups_licites[True].discard((i + d2, j - d2))

                elif old_val2 == -1:
                    self.coups_licites[True].add((i + d2, j - d2))
                    if (i + d2, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d2,
                                                                                                          j - d2,
                                                                                                          False):
                        self.coups_licites[False].discard((i + d2, j - d2))
        else:
            if i - d1 >= 0 and j + d1 < self.taille :

                if (i - d1, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d1, j + d1, True):
                    self.coups_licites[True].discard((i - d1, j + d1))
                if (i - d1, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d1, j + d1, False):
                    self.coups_licites[False].discard((i - d1, j + d1))

            if i + d2 < self.taille and j - d2 >= 0 :

                if (i + d2, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d2, j - d2, True):
                    self.coups_licites[True].discard((i + d2, j - d2))
                if (i + d2, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d2, j - d2, False):
                    self.coups_licites[False].discard((i + d2, j - d2))

        ##############
        blanc = base_blanc
        noir = base_noir
        old_val1 = val0
        old_val2 = val0

        d1 = 1
        if i + d1 < self.taille and j + d1 < self.taille:
            val1 = self.plateau[i + d1, j + d1, 0]
        while i + d1 < self.taille and j + d1 < self.taille and val1 != 0:
            if val1 == 1:
                blanc = True
            if val1 == -1:
                noir = True
            d1 += 1
            if i + d1 < self.taille and j + d1 < self.taille:
                old_val1 = val1
                val1 = self.plateau[i + d1, j + d1, 0]


        d2 = 1
        if i - d2 >= 0 and j - d2 >= 0:
            val2 = self.plateau[i - d2, j - d2, 0]
        while i - d2 >= 0 and j - d2 >= 0 and val2 != 0:
            if val2 == 1:
                blanc = True
            if val2 == -1:
                noir = True
            d2 += 1
            if i - d2 >= 0 and j - d2 >= 0:
                old_val2 = val2
                val2 = self.plateau[i - d2, j - d2, 0]

        if noir and blanc:
            if i + d1 < self.taille and j + d1 < self.taille:
                if old_val1 == 1:
                    self.coups_licites[False].add((i + d1, j + d1))
                    if (i + d1, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d1, j + d1,
                                                                                                         True):
                        self.coups_licites[True].discard((i + d1, j + d1))
                elif old_val1 == -1:
                    self.coups_licites[True].add((i + d1, j + d1))
                    if (i + d1, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d1,
                                                                                                          j + d1,
                                                                                                          False):
                        self.coups_licites[False].discard((i + d1, j + d1))
            if i - d2 >= 0 and j - d2 >= 0:
                if old_val2 == 1:
                    self.coups_licites[False].add((i - d2, j - d2))
                    if (i - d2, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d2, j - d2,
                                                                                                         True):
                        self.coups_licites[True].discard((i - d2, j - d2))
                elif old_val2 == -1:
                    self.coups_licites[True].add((i - d2, j - d2))
                    if (i - d2, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d2,
                                                                                                          j - d2,
                                                                                                          False):
                        self.coups_licites[False].discard((i - d2, j - d2))
        else:
            if i + d1 < self.taille and j + d1 < self.taille :
                if (i + d1, j + d1) in self.coups_licites[True] and not self.est_coups_coups_licites(i + d1, j + d1, True):
                    self.coups_licites[True].discard((i + d1, j + d1))
                if (i + d1, j + d1) in self.coups_licites[False] and not self.est_coups_coups_licites(i + d1, j + d1, False):
                    self.coups_licites[False].discard((i + d1, j + d1))
            if i - d2 >= 0 and j - d2 >= 0 :
                if (i - d2, j - d2) in self.coups_licites[True] and not self.est_coups_coups_licites(i - d2, j - d2, True):
                    self.coups_licites[True].discard((i - d2, j - d2))
                if (i - d2, j - d2) in self.coups_licites[False] and not self.est_coups_coups_licites(i - d2, j - d2, False):
                    self.coups_licites[False].discard((i - d2, j - d2))






    def est_coups_coups_licites(self, i, j, blanc_joue):

            if blanc_joue:
                val = 1
            else:
                val = -1

            d = 1
            while i + d < self.taille and self.plateau[i + d, j, 0] == - val:
                d += 1

            if d > 1 and i + d < self.taille and self.plateau[i + d, j, 0] == val:
                return True

            d = 1
            while j + d < self.taille and self.plateau[i, j + d, 0] == - val:
                d += 1

            if d > 1 and j + d < self.taille and self.plateau[i, j + d, 0] == val:
                return True

            d = 1
            while i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d, 0] == - val:
                d += 1

            if d > 1 and i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d, 0] == val:
                return True

            d = 1
            while i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d, 0] == - val:
                d += 1

            if d > 1 and i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d, 0] == val:
                return True

            d = 1
            while i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d, 0] == - val:
                d += 1

            if d > 1 and i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d, 0] == val:
                return True

            d = 1
            while i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d, 0] == - val:
                d += 1

            if d > 1 and i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d, 0] == val:
                return True

            d = 1
            while i - d >= 0 and self.plateau[i - d, j, 0] == - val:
                d += 1

            if d > 1 and i - d >= 0 and self.plateau[i - d, j, 0] == val:
                return True

            d = 1
            while j - d >= 0 and self.plateau[i, j - d, 0] == - val:
                d += 1

            if d > 1 and j - d >= 0 and self.plateau[i, j - d, 0] == val:
                return True

            return False



    def coupsLicites(self):

        return list(self.coups_licites[self.blancJoue])


    def estFini(self, jeu):
            return self.fini


    def getGagnant(self):
        return self.gagnant


    def calcul_scoring(self):
        self.np_pieces_noir = len(np.argwhere(self.plateau[:, :, 0] == -1))
        self.np_pieces_blanc = len(np.argwhere(self.plateau[:, :, 0] == 1))
        self.score = self.np_pieces_noir - self.np_pieces_blanc


    def test_fini(self):

        deb = time()

        if len(self.historique) == self.taille * self.taille:
            self.fini = True

        if not len(self.coupsLicites()):
            self.blancJoue = not self.blancJoue
            #self.calcul_coupsLicites()
            self.nb_tour_avec_pass += 1
            if not len(self.coupsLicites()):
                self.fini = True
                self.nb_tour_avec_pass += 1

        if self.fini:
            self.calcul_scoring()
            if self.score > 0:
                self.gagnant = 'noir'
                self.presence_frac = (self.np_pieces_noir + 1) / (self.np_pieces_blanc + 1)
            elif self.score < 0:
                self.gagnant = 'blanc'
                self.presence_frac = -(self.np_pieces_blanc + 1) / (self.np_pieces_noir + 1)
            else:
                self.gagnant = 'nul'
                self.presence_frac = 0



        self.tps_fini += time() - deb
