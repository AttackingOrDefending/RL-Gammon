import numpy as np


taille_par_defaut = 7

class Ataxx():

    def __init__(self, taille=taille_par_defaut, borne=None, repetition_max=0, repetition_toujours_nul=False):

        self.repetition_max = repetition_max
        self.repetition_toujours_nul = repetition_toujours_nul

        self.dtype = 'int8'  # 'float32' # 'float'

        if taille > 15 and borne is None:
            borne = 400

        #print('borne :',borne)
        self.tour_max = borne
        if borne:
            self.longueur_max = borne
        else:
            self.longueur_max = 750

        self.longueur_moyenne = 100



        self.taille = taille
        self.panels = 2

        self.init()

    def init(self):

        self.nb_repet=0
        self.hash_old_positions = []

        self.tour = 0

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.plateau = np.zeros((self.taille, self.taille,self.panels), dtype=self.dtype)

        self.historique = []

        self.pieces_blanc.add((0, 0))
        self.plateau[0,0,0]=1

        self.pieces_blanc.add((self.taille-1, self.taille-1))
        self.plateau[self.taille-1, self.taille-1, 0] = 1

        self.pieces_noir.add((self.taille-1, 0))
        self.plateau[self.taille-1,0,0]=-1

        self.pieces_noir.add((0, self.taille - 1))
        self.plateau[0, self.taille - 1, 0] = -1

        self.fini = False
        self.gagnant = None

        self.actu_couleur_plateau()
        self.calcul_coups_licites()

    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 1] = -np.ones((self.taille, self.taille), dtype=self.dtype)

    def undo(self):


        if self.hash_old_positions:
            self.hash_old_positions.pop()

        l1, l2, self.blancJoue, prises, self.nb_repet = self.historique.pop()

        self.tour -= 1

        l1a, l1b = l1
        l2a, l2b = l2

        self.plateau[l2a, l2b, 0] = 0

        if self.blancJoue:
            self.plateau[l1a, l1b, 0]=1
            self.pieces_blanc.add(l1)
            self.pieces_blanc.remove(l2)
            for i,j in  prises:
                self.plateau[i, j, 0] = -1
                self.pieces_noir.add((i,j))
                self.pieces_blanc.remove((i,j))
        else:
            self.plateau[l1a, l1b, 0] = -1
            self.pieces_noir.add(l1)
            self.pieces_noir.remove(l2)
            for i,j in  prises:
                self.plateau[i, j, 0] = 1
                self.pieces_blanc.add((i,j))
                self.pieces_noir.remove((i,j))

        self.fini = False
        self.gagnant = None

        self.actu_couleur_plateau()
        self.calcul_coups_licites()

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1

    def raz(self):
        self.init()


    def hash_plateau(self):
        return self.plateau.tobytes()

    def jouer(self, l1, l2):


        if self.repetition_max is not None:
            self.hash_old_positions.append(self.hash_plateau())

        self.tour += 1

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        """if (l1, l2) not in self.coupsLicites():
            print('ouille ...')"""

        l1a,l1b = l1
        l2a,l2b = l2

        assert self.plateau[l2a,l2b,0] == 0

        if abs(l1a - l2a) > 1 or  abs(l1b - l2b) > 1:
            self.plateau[l1a,l1b,0] = 0
            if self.blancJoue:
                self.pieces_blanc.remove(l1)
            else:
                self.pieces_noir.remove(l1)

        prises = []

        if self.blancJoue:

            for da in [-1,0,1]:
                for db in [-1, 0, 1]:
                    if self.in_bord(l2a+da, l2b+db) and self.plateau[l2a+da, l2b+db,0] == -1:
                        self.plateau[l2a+da, l2b+db,0] = 1
                        prises.append((l2a+da, l2b+db))
                        self.pieces_blanc.add((l2a+da, l2b+db))
                        self.pieces_noir.remove((l2a + da, l2b + db))

            self.pieces_blanc.add(l2)
            self.plateau[l2a,l2b,0] = 1


        else:
            for da in [-1,0,1]:
                for db in [-1, 0, 1]:
                    if self.in_bord(l2a+da, l2b+db) and self.plateau[l2a+da, l2b+db,0] == 1:
                        self.plateau[l2a+da, l2b+db,0] = -1
                        prises.append((l2a+da, l2b+db))
                        self.pieces_noir.add((l2a+da, l2b+db))
                        self.pieces_blanc.remove((l2a + da, l2b + db))

            self.pieces_noir.add(l2)
            self.plateau[l2a,l2b,0] = -1

        self.historique.append((l1, l2, self.blancJoue, prises, self.nb_repet))

        self.blancJoue = not self.blancJoue

        self.calcul_coups_licites()

        if not self.coupsLicites():
            self.blancJoue = not self.blancJoue

            self.calcul_coups_licites()

        self.actu_couleur_plateau()

        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())

        if self.repetition_toujours_nul and self.repetition_max is not None and self.nb_repet > self.repetition_max:
            self.fini = True
            self.gagnant = 'nul'
            self.score = 0
            self.score2 = 0
            self.score3 = 0
            self.score4 = 0

        if len(self.pieces_noir)==0 or len(self.pieces_blanc)==0 or len(self.pieces_noir) + len(self.pieces_blanc) == self.taille **2 or self.tour_max and not self.tour < self.tour_max or (not self.repetition_toujours_nul and self.repetition_max is not None and self.nb_repet > self.repetition_max) :
            self.fini = True
            self.score = len(self.pieces_noir) - len(self.pieces_blanc)
            if len(self.pieces_noir) > len(self.pieces_blanc):
                self.gagnant = 'noir'
                self.mobilite = mobilite
                self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
                self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
                self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

                self.presence_frac = (len(self.pieces_noir) + 1) / (len(self.pieces_blanc) + 1)
                self.presence = max(len(self.pieces_noir) - len(self.pieces_blanc), 1)

            elif len(self.pieces_noir) < len(self.pieces_blanc):
                self.gagnant = 'blanc'
                self.mobilite = -mobilite
                self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
                self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
                self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)

                self.presence_frac = -(len(self.pieces_blanc) + 1) / (len(self.pieces_noir) + 1)
                self.presence = min(len(self.pieces_noir) - len(self.pieces_blanc), -1)
            else:
                self.gagnant = 'nul'
                self.mobilite = 0
                self.mobilite_cumule = 0
                self.mobilite_cumule_diff = 0
                self.mobilite_cumule_frac = 0

                self.presence_frac = 0
                self.presence = 0


    def symetrie_joueur(self, plateau):
        return np.rot90(-1*plateau, 1)

    def in_bord(self, i,j):
        return 0 <= i < self.taille and 0 <= j < self.taille

    def calcul_coups_licites(self):

        l = []

        if self.blancJoue:

            for i,j in self.pieces_blanc:

                for di in [-2, -1, 0, 1, 2]:
                    for dj in [-2, -1, 0, 1, 2]:
                        p = i+di, j+dj
                        if self.in_bord(*p) and p not in self.pieces_blanc and p not in self.pieces_noir:
                            l.append(((i,j),p))


        else:

            for i, j in self.pieces_noir:

                for di in [-2, -1, 0, 1, 2]:
                    for dj in [-2, -1, 0, 1, 2]:
                        p = i+di, j+dj
                        if self.in_bord(*p) and p not in self.pieces_blanc and p not in self.pieces_noir:
                            l.append(((i,j),p))




        self.coups_licites = l

    def coupsLicites(self):
        return self.coups_licites