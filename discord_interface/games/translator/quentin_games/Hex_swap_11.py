import numpy as np


taille_par_defaut = 11

class Hex_swap():

    def __init__(self, taille=taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.raz()



        self.longueur_max = taille * taille +1
        self.longueur_moyenne = self.longueur_max / 2

        self.coups_licites = set([(i, j) for i in range(self.taille) for j in range(self.taille)])

        self.coup_swap = 0
        self.coup_premier = 0

    def undo(self,):

        (i,j), self.blancJoue = self.historique.pop()

        if (i,j) == self.swapped:
            self.swapped = None
            #self.plateau[i, j,0] = - self.plateau[i,j,0]

            self.plateau[:, :, 1] = np.zeros((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[i,j,0]=0

            if len(self.historique) == 1:
                self.coups_licites.add(self.historique[0][0])




        self.fini = False

        self.gagnant = None

        self.coups_licites.add((i, j))


    def a_swapper(self):
        return self.swapped is not None

    def raz(self):

        self.swapped = None

        self.fini = False

        self.blancJoue = False

        self.plateau = np.zeros((self.taille, self.taille,2), dtype=self.dtype)

        self.historique = []

        self.gagnant = None

        self.coups_licites = set([(i, j) for i in range(self.taille) for j in range(self.taille)])




    def jouer(self, i, j):

        if i == 'swap':
            i, j = self.historique[-1][0]

        #print(i,j,' ',self.historique, self.coups_licites)
        if len(self.historique) == 1:
            self.coup_premier += 1

        if self.plateau[i, j,0] !=0:
            if len(self.historique) == 1 and not self.swapped:

                self.historique.append(((i, j), self.blancJoue))

                self.swapped = (i, j)
                #self.plateau[i, j,0] = - self.plateau[i, j,0]

                self.plateau[:,:,1] = np.ones((self.taille, self.taille), dtype=self.dtype)

                self.test_fini(i, j)

                self.coup_swap += 1

            else:
                print(i,j)
                print(self.plateau[i, j,0], self.blancJoue)
                print(self.coupsLicites())
                print(len(self.historique), self.swapped)
                raise Exception('ho, du con !')
        else:
            self.historique.append(((i,j), self.blancJoue))

            if self.blancJoue and not self.swapped or not self.blancJoue and self.swapped :
                self.plateau[i, j,0] = 1
            else:
                self.plateau[i, j,0] = -1

            self.test_fini(i, j)

        self.blancJoue = not self.blancJoue

        if len(self.historique) != 1:
            self.coups_licites.remove((i, j))
        if len(self.historique) == 2 and not self.swapped:
            #print(self.historique[-2])
            self.coups_licites.remove(self.historique[-2][0])

        #print(self.coupsLicites())

    def coupsLicites(self):
        return list(self.coups_licites)




    # note : famille de classe d'equivalence pour chaque joueur et un classe d eq pour chaque bord (noeud) en plus.


    def test_fini(self, i, j):


        if self.blancJoue and not self.swapped or not self.blancJoue and self.swapped:


            haut_atteint = i == 0
            bas_atteint = i == self.taille - 1


            val = 1

            ouvert = set()
            ouvert.add((i,j))
            fermee = set()

            while ouvert:

                ouvert2 = set([])

                fermee |= ouvert

                for (s, t) in ouvert :

                    for (l, k) in self.voisins(s, t, val):

                        if (l,k) not in fermee :
                            ouvert2.add((l,k))

                            if l == 0 :
                                haut_atteint = True

                            elif l == self.taille -1 :
                                bas_atteint = True

                            if haut_atteint and bas_atteint:
                                self.fini = True
                                if self.swapped:
                                    self.gagnant = "noir"
                                else:
                                    self.gagnant = "blanc"
                                return


                ouvert = ouvert2

        else:


            droite_atteint = j == self.taille -1
            gauche_atteint = j == 0

            val = -1

            ouvert = set()
            ouvert.add((i, j))
            fermee = set()

            while ouvert:

                ouvert2 = set([])

                fermee |= ouvert

                for (s, t) in ouvert:

                    for (l, k) in self.voisins(s, t, val):

                        if (l, k) not in fermee:
                            ouvert2.add((l, k))

                            if k == 0:
                                gauche_atteint = True

                            elif k == self.taille - 1:
                                droite_atteint = True

                            if gauche_atteint and droite_atteint:
                                self.fini = True
                                if self.swapped:
                                    self.gagnant = "blanc"
                                else:
                                    self.gagnant = "noir"
                                return

                ouvert = ouvert2




    def voisins(self, i, j, val):

        v = []

        if i > 0 :
            if self.plateau[i - 1, j,0] == val:
                v.append((i - 1, j))

            if j < self.taille - 1 and self.plateau[i - 1, j + 1,0] == val:
                v.append((i - 1, j + 1))

        if i < self.taille - 1:
            if self.plateau[i + 1, j,0] == val:
                v.append((i + 1, j))

            if j > 0 and self.plateau[i + 1, j - 1,0] == val:
                v.append((i + 1, j - 1))

        if j > 0 and self.plateau[i, j - 1,0] == val:
            v.append((i, j - 1))

        if j < self.taille - 1 and self.plateau[i, j + 1,0] == val:
            v.append((i, j + 1))

        return v
