import numpy as np


taille_par_defaut = 8

class Breakthrough:

    def __init__(self, taille=taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.longueur_max = taille * 4 * 5
        self.longueur_moyenne = 90#taille * 5

        self.taille = taille

        self.init()

    def init(self):

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.plateau = np.zeros((self.taille, self.taille,2), dtype=self.dtype)

        self.historique = []

        for j in range(self.taille):
            for i in range(0,2):
                self.pieces_blanc.add((i, j))
                self.plateau[i,j,0]=1

        for j in range(self.taille):
            for i in range(self.taille-2,self.taille):
                self.pieces_noir.add((i, j))
                self.plateau[i,j,0]=-1

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
        l1, l2, self.blancJoue, prise = self.historique.pop()

        l1a, l1b = l1
        l2a, l2b = l2

        if self.blancJoue:
            self.plateau[l1a, l1b, 0]=1
            self.pieces_blanc.add(l1)
            self.pieces_blanc.remove(l2)
            if prise:
                self.plateau[l2a, l2b, 0] = -1
                self.pieces_noir.add(l2)
            else:
                self.plateau[l2a, l2b, 0] = 0
        else:
            self.plateau[l1a, l1b, 0] = -1
            self.pieces_noir.add(l1)
            self.pieces_noir.remove(l2)
            if prise:
                self.plateau[l2a, l2b, 0] = 1
                self.pieces_blanc.add(l2)
            else:
                self.plateau[l2a, l2b, 0] = 0

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

    def jouer(self, l1, l2):

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        if (l1, l2) not in self.coupsLicites():

            raise Exception('non legal')

        l1a,l1b = l1
        l2a,l2b = l2

        self.plateau[l1a,l1b,0] = 0
        prise = False

        if self.blancJoue:
            self.pieces_blanc.remove(l1)
            self.pieces_blanc.add(l2)
            self.plateau[l2a,l2b,0] = 1
            if l2 in self.pieces_noir:
                self.pieces_noir.remove(l2)
                prise = True
            if l2[0] == self.taille -1 or len(self.pieces_noir) == 0:
                self.fini = True
                self.gagnant = 'blanc'
        else:
            self.pieces_noir.remove(l1)
            self.pieces_noir.add(l2)
            self.plateau[l2a,l2b,0] = -1
            if l2 in self.pieces_blanc:
                self.pieces_blanc.remove(l2)
                prise = True
            if l2[0] == 0 or len(self.pieces_blanc) == 0:
                self.fini = True
                self.gagnant = 'noir'

        self.historique.append((l1, l2, self.blancJoue, prise))

        self.blancJoue = not self.blancJoue

        self.actu_couleur_plateau()
        self.calcul_coups_licites()

        if self.fini:
            if self.gagnant == 'noir':
                self.mobilite = mobilite
                self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
                self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
                self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

                self.presence_frac = (len(self.pieces_noir) + 1) / (len(self.pieces_blanc) + 1)
                self.presence = max(len(self.pieces_noir) - len(self.pieces_blanc), 1)

            elif self.gagnant == 'blanc':
                self.mobilite = -mobilite
                self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
                self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
                self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)

                self.presence_frac = -(len(self.pieces_blanc) + 1) / (len(self.pieces_noir) + 1)
                self.presence = min(len(self.pieces_noir) - len(self.pieces_blanc), -1)
            else:
                self.mobilite = 0
                self.mobilite_cumule = 0
                self.mobilite_cumule_diff = 0
                self.mobilite_cumule_frac = 0

                self.presence_frac = 0
                self.presence = 0



    def symetrie_joueur(self, plateau):
        return np.flip(-1*plateau, 0)

    def calcul_coups_licites(self):

        l = []

        if self.blancJoue:

            for i,j in self.pieces_blanc:

                if i < self.taille -1:
                    if self.plateau[i+1,j,0] == 0:
                        l.append(((i,j),(i + 1, j)))
                    if j > 0 and self.plateau[i + 1, j - 1,0] != 1:
                        l.append(((i,j),(i+1,j-1)))
                    if j < self.taille -1 and self.plateau[i + 1, j + 1,0] != 1:
                        l.append(((i, j), (i + 1, j + 1)))

        else:

            for i, j in self.pieces_noir:

                if i > 0:
                    if self.plateau[i - 1, j,0] == 0:
                        l.append(((i, j), (i - 1, j)))
                    if j > 0 and self.plateau[i - 1, j - 1,0] != -1:
                        l.append(((i, j), (i - 1, j - 1)))
                    if j < self.taille - 1 and self.plateau[i - 1, j + 1,0] != -1:
                        l.append(((i, j), (i - 1, j + 1)))

        self.coups_licites = l

    def coupsLicites(self):
        return self.coups_licites