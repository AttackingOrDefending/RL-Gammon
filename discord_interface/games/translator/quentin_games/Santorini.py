import numpy as np

taille_par_defaut = 5

class Santorini:

    def __init__(self, taille=taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.longueur_max = 2*4*taille*taille
        self.longueur_moyenne = self.longueur_max / 4

        self.positions = set([(i,j) for i in range(self.taille) for j in range(self.taille)])

        self.init()

    def actu_codage_phase_colorer(self):
        self.plateau[:, :, 2] = np.zeros((self.taille, self.taille), dtype=self.dtype)
        if self.phase_un:

            if self.blancJoue:
                for k, l in self.pieces_blanc:
                    self.plateau[k, l, 2] = 1
            else:
                for k, l in self.pieces_noir:
                    self.plateau[k, l, 2] = -1

        else:
            i, j = self.historique[-1][1]

            if self.blancJoue:
                self.plateau[i, j, 2] = 1
            else:
                self.plateau[i, j, 2] = -1


    """"def actu_codage_phase_colorer(self):
        if self.phase_un:
            if self.tour <= 4:
                self.plateau[:,:,2] = np.zeros((self.taille, self.taille), dtype=self.dtype)
            else:
                i, j = self.historique[-2][1]
                self.plateau[i,j,2]=0

            if self.blancJoue:
                for k, l in self.pieces_blanc:
                    self.plateau[k, l, 2] = 1
            else:
                for k, l in self.pieces_noir:
                    self.plateau[k,l,2]=-1

        else:
            i, j = self.historique[-1][1]
            if self.blancJoue:
                self.plateau[i, j, 2] = 1
                for k, l in self.pieces_blanc:
                    if k!=i or l!=j:
                        self.plateau[k,l,2]=0
                for k, l in self.pieces_noir:
                        self.plateau[k,l,2]=0

            else:
                self.plateau[i, j, 2] = -1
                for k, l in self.pieces_noir:
                    if k!=i or l!=j:
                        self.plateau[k,l,2]=0
                for k, l in self.pieces_blanc:
                        self.plateau[k,l,2]=0"""


    """
        def actu_codage_phase_colorer_jouer(self):
        if self.phase_un:
            if len(self.historique) > 1 and len(self.historique[-2]) == 2:
                i, j = self.historique[-2][1]
                self.plateau[i,j,2]=0

            if self.blancJoue:
                for k, l in self.pieces_blanc:
                    self.plateau[k, l, 2] = 1
            else:
                for k, l in self.pieces_noir:
                    self.plateau[k,l,2]=-1

        else:
            i, j = self.historique[-1][1]
            if self.blancJoue:
                for k, l in self.pieces_blanc:
                    if k!=i or l!=j:
                        self.plateau[k,l,2]=0

            else:
                for k, l in self.pieces_noir:
                    if k!=i or l!=j:
                        self.plateau[k,l,2]=0

    def actu_codage_phase_colorer_undo(self):
        if self.phase_un:

            if self.blancJoue:
                for k, l in self.pieces_blanc:
                    self.plateau[k, l, 2] = 1
            else:
                for k, l in self.pieces_noir:
                    self.plateau[k,l,2]= -1

        else:
            i, j = self.historique[-1][1]
            if self.blancJoue:
                self.plateau[i, j, 2] = 1
                for k, l in self.pieces_noir:
                        self.plateau[k,l,2]=0

            else:
                self.plateau[i, j, 2] = -1
                for k, l in self.pieces_blanc:
                        self.plateau[k,l,2]=0
    """
    def init(self):

        #self.cases_vides = set([(i,j) for i in range(self.taille) for j in range(self.taille)])

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.tour = 0
        self.phase_un = True
        self.blancJoue = False
        self.historique = []

        self.fini = False
        self.gagnant = None

        self.plateau = np.zeros((self.taille, self.taille,3), dtype=self.dtype) # CODAGE : 0 : les positions des pieces ; 1 : l appartenance des pieces

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.actu_codage_phase_colorer()
        self.calcul_coups_licite()


        #print(self.plateau)

    def undo(self):
        self.fini = False
        self.gagnant = None

        a,b, self.phase_un, self.blancJoue = self.historique.pop()

        if self.tour <= 4:
            self.plateau[a, b, 0] = 0
            if self.blancJoue:
                self.pieces_blanc.remove((a,b))
            else:
                self.pieces_noir.remove((a,b))
        else:

            if self.phase_un:
                i,j = a
                k,l = b
                self.plateau[k, l, 0] = 0

                if self.blancJoue:
                    self.pieces_blanc.add((i, j))
                    self.pieces_blanc.remove((k, l))
                    self.plateau[i, j, 0] = 1
                else:
                    self.pieces_noir.add((i, j))
                    self.pieces_noir.remove((k, l))
                    self.plateau[i,j,0]= -1
            else:
                self.plateau[a,b,1] -= 1

        self.tour -= 1

        #print('+')
        self.actu_codage_phase_colorer()
        self.calcul_coups_licite()
        #print('-')

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1


    def raz(self):
        self.init()

    def jouer(self, a, b):

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        """print(a,b)
        print(self.plateau[:,:,0])
        print(self.plateau[:, :, 1])
        print(self.coups_licites)
        print(self.phase_un, self.blancJoue)"""

        if self.tour < 4:
            if self.blancJoue:
                self.plateau[a, b, 0] = 1
                self.pieces_blanc.add((a,b))
            else:
                self.plateau[a, b, 0] = -1

                self.pieces_noir.add((a,b))
        else:
            if self.phase_un:
                i,j = a
                k,l = b
                self.plateau[i,j,0] = 0

                if self.blancJoue:
                    self.pieces_blanc.remove((i,j))
                    self.pieces_blanc.add((k,l))
                    self.plateau[k,l,0] = 1

                    if self.plateau[k, l, 1] == 3:
                        self.fini = True
                        self.gagnant = 'blanc'
                else:
                    self.pieces_noir.remove((i, j))
                    self.pieces_noir.add((k, l))
                    self.plateau[k,l,0] = -1

                    if self.plateau[k, l, 1] == 3:
                        self.fini = True
                        self.gagnant = 'noir'

            else:
                self.plateau[a,b,1] += 1


        self.historique.append((a,b, self.phase_un, self.blancJoue))
        #print((a,b, self.phase_un, self.blancJoue, self.tour))
        self.tour += 1

        if self.tour <= 4:
            if self.tour % 2 == 0:
                self.blancJoue = not self.blancJoue
        else:
            self.phase_un = not self.phase_un
            if self.phase_un:
                self.blancJoue = not self.blancJoue


        self.calcul_coups_licite()
        self.actu_codage_phase_colorer()

        if not self.coupsLicites():#self.blancJoue = not self.blancJoue #pour regler buger de ludii
            self.fini = True
            if self.blancJoue:
                self.gagnant = 'noir'
            else:
                self.gagnant = 'blanc'

        if self.fini:
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

    def calcul_coups_licite(self):
        #print(self.phase_un)
        l = []

        if self.tour < 4 :
            self.coups_licites = list(self.positions)

            for i, j in self.pieces_noir:
                self.coups_licites.remove((i,j))

            for i, j in self.pieces_blanc:
                self.coups_licites.remove((i,j))

        else:

            if self.phase_un:
                L = []

                if self.blancJoue:

                    for i, j in self.pieces_blanc:

                        niveau_max = min(3, 1 + self.plateau[i, j, 1])

                        h = self.plateau[i, j, 1]

                        if i < self.taille - 1:
                            if self.plateau[i + 1, j, 0] == 0 and self.plateau[i + 1, j, 1] <= niveau_max:
                                L.append(((i, j), (i + 1, j)))
                            if j > 0 and self.plateau[i + 1, j - 1, 0] == 0 and self.plateau[i + 1, j - 1 ,1] <= niveau_max:
                                L.append(((i, j), (i + 1, j - 1)))
                            if j < self.taille - 1 and self.plateau[i + 1, j + 1, 0] == 0 and self.plateau[i + 1, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i + 1, j + 1)))

                        if i > 0:
                            if self.plateau[i - 1, j, 0] == 0 and self.plateau[i - 1, j, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j)))
                            if j > 0 and self.plateau[i - 1, j - 1, 0] == 0 and self.plateau[i - 1, j - 1, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j - 1)))
                            if j < self.taille - 1 and self.plateau[i - 1, j + 1, 0] == 0 and self.plateau[i - 1, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j + 1)))

                        if j > 0:
                            if self.plateau[i, j - 1, 0] == 0 and self.plateau[i, j - 1, 1] <= niveau_max:
                                L.append(((i, j), (i, j - 1)))

                        if j < self.taille - 1:
                            if self.plateau[i, j + 1, 0] == 0 and self.plateau[i, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i, j + 1)))


                else:

                    for i, j in self.pieces_noir:

                        niveau_max = min(3, 1 + self.plateau[i, j, 1])

                        if i > 0:
                            if self.plateau[i - 1, j, 0] == 0 and self.plateau[i - 1, j, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j)))
                            if j > 0 and self.plateau[i - 1, j - 1, 0] == 0 and self.plateau[i - 1, j - 1, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j - 1)))
                            if j < self.taille - 1 and self.plateau[i - 1, j + 1, 0] == 0 and self.plateau[i - 1, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i - 1, j + 1)))

                        if i < self.taille - 1:
                            if self.plateau[i + 1, j, 0] == 0 and self.plateau[i + 1, j, 1] <= niveau_max:
                                L.append(((i, j), (i + 1, j)))
                            if j > 0 and self.plateau[i + 1, j - 1, 0] == 0 and self.plateau[i + 1, j - 1, 1] <= niveau_max:
                                L.append(((i, j), (i + 1, j - 1)))
                            if j < self.taille - 1 and self.plateau[i + 1, j + 1, 0] == 0 and self.plateau[i + 1, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i + 1, j + 1)))

                        if j > 0:
                            if self.plateau[i, j - 1, 0] == 0 and self.plateau[i, j - 1, 1] <= niveau_max:
                                L.append(((i, j), (i, j - 1)))

                        if j < self.taille - 1:
                            if self.plateau[i, j + 1, 0] == 0 and self.plateau[i, j + 1, 1] <= niveau_max:
                                L.append(((i, j), (i, j + 1)))

            else:
                L = []

                a, b, _, _ = self.historique[-1]
                i, j = b

                if i > 0:
                    if self.plateau[i - 1, j, 0] == 0 and self.plateau[i - 1, j, 1] < 4:
                        L.append((i - 1, j))
                    if j > 0 and self.plateau[i - 1, j - 1, 0] == 0 and self.plateau[i - 1, j - 1, 1] < 4:
                        L.append((i - 1, j - 1))
                    if j < self.taille - 1 and self.plateau[i - 1, j + 1, 0] == 0 and self.plateau[i - 1, j + 1, 1] < 4:
                        L.append((i - 1, j + 1))

                if i < self.taille - 1:
                    if self.plateau[i + 1, j, 0] == 0 and self.plateau[i + 1, j, 1] < 4:
                        L.append((i + 1, j))
                    if j > 0 and self.plateau[i + 1, j - 1, 0] == 0 and self.plateau[i + 1, j - 1, 1] < 4:
                        L.append((i + 1, j - 1))
                    if j < self.taille - 1 and self.plateau[i + 1, j + 1, 0] == 0  and self.plateau[i + 1, j + 1, 1] < 4:
                        L.append((i + 1, j + 1))

                if j > 0:
                    if self.plateau[i, j - 1, 0] == 0 and self.plateau[i, j - 1, 1] < 4 :
                        L.append((i, j - 1))

                if j < self.taille - 1:
                    if self.plateau[i, j + 1, 0] == 0 and self.plateau[i, j + 1, 1] < 4:
                        L.append((i, j + 1))
            self.coups_licites = L




    def coupsLicites(self):
        return self.coups_licites

    def symetrie_joueur(self, plateau):
        raise NotImplemented