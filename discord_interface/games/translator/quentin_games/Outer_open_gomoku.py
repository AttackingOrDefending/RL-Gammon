import numpy as np


taille_par_defaut = 15

class Outer_open_gomoku():


    def __init__(self, taille = taille_par_defaut, alignement = 5):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.alignement_victoire = alignement


        self.longueur_max = taille * taille
        self.longueur_moyenne = taille

        self.raz()




    def undo(self):

        (i,j) = self.historique.pop()
        self.plateau[i,j]=0
        self.coups_licites.add((i,j))

        self.blancJoue = not self.blancJoue

        self.fini = False

        self.gagnant = None



    def raz(self):

        self.gagnant = None

        self.fini = False

        self.blancJoue = False

        self.plateau = np.zeros((self.taille, self.taille), dtype=self.dtype)

        self.historique = []

        self.coups_licites = set([(i,j) for i in range(self.taille) for j in range(self.taille)])

        self.coups_licites_start = [(0, i) for i in range(self.taille)]+[(1, i) for i in range(self.taille)]+[(self.taille-1, i) for i in range(self.taille)]+[(self.taille-2, i) for i in range(self.taille)] + \
                       [(i,0) for i in range(2,self.taille-2)]+[(i,1) for i in range(2,self.taille-2)]+[(i,self.taille-1) for i in range(2,self.taille-2)]+[(i,self.taille-2) for i in range(2,self.taille-2)]




    def jouer(self, i, j):

        if self.plateau[i, j] !=0:
            raise Exception('ho, du con !')

        self.historique.append((i,j))

        self.coups_licites.remove((i,j))

        if self.blancJoue:
            self.plateau[i, j] = 1
        else:
            self.plateau[i, j] = -1

        self.test_fini(i, j)

        self.blancJoue = not self.blancJoue




    def coupsLicites(self):
        if self.fini:
            return np.array([])
        else:
            #print(self.plateau)
            #print(len(np.argwhere(self.plateau == 0)))
            #return np.argwhere(self.plateau == 0)
            if self.historique:
                return list(self.coups_licites)
            else:
                return self.coups_licites_start


    def estFini(self, jeu):
        return self.fini


    def getGagnant(self):
        return self.gagnant


    def test_fini(self, i, j):
        if not len(self.coupsLicites()):
            self.fini = True
            self.gagnant = 'nul'
        else:

            v = self.plateau[i, j]

            d1 = 1
            while i + d1 < self.taille and 1 + d1 <= self.alignement_victoire and self.plateau[i + d1, j] == v:
                d1 += 1

            d1 -= 1

            d2 = 1
            while i - d2 >= 0 and 1 + d1 + d2 <= self.alignement_victoire and self.plateau[i - d2, j] == v:
                d2 += 1

            d2 -= 1

            if 1 + d1 + d2 >= self.alignement_victoire:
                self.fini = True

                if self.blancJoue:
                    self.gagnant = 'blanc'
                else:
                    self.gagnant = 'noir'

                return

            d1 = 1
            while j + d1 < self.taille and 1 + d1 <= self.alignement_victoire and self.plateau[i, j + d1] == v:
                d1 += 1

            d1 -= 1

            d2 = 1
            while j - d2 >= 0 and 1 + d1 + d2 <= self.alignement_victoire and self.plateau[i, j - d2] == v:
                d2 += 1

            d2 -= 1

            if 1 + d1 + d2 >= self.alignement_victoire:
                self.fini = True

                if self.blancJoue:
                    self.gagnant = 'blanc'
                else:
                    self.gagnant = 'noir'

                return

            d1 = 1
            while i + d1 < self.taille and j + d1 < self.taille and 1 + d1 <= self.alignement_victoire and self.plateau[
                i + d1, j + d1] == v:
                d1 += 1

            d1 -= 1

            d2 = 1
            while i - d2 >= 0 and j - d2 >= 0 and 1 + d1 + d2 <= self.alignement_victoire and self.plateau[
                i - d2, j - d2] == v:
                d2 += 1

            d2 -= 1

            if 1 + d1 + d2 >= self.alignement_victoire:
                self.fini = True

                if self.blancJoue:
                    self.gagnant = 'blanc'
                else:
                    self.gagnant = 'noir'

                return

            d1 = 1
            while i + d1 < self.taille and j - d1 >= 0 and 1 + d1 <= self.alignement_victoire and self.plateau[
                i + d1, j - d1] == v:
                d1 += 1

            d1 -= 1

            d2 = 1
            while i - d2 >= 0 and j + d2 < self.taille and 1 + d1 + d2 <= self.alignement_victoire and self.plateau[
                i - d2, j + d2] == v:
                d2 += 1

            d2 -= 1

            if 1 + d1 + d2 >= self.alignement_victoire:
                self.fini = True

                if self.blancJoue:
                    self.gagnant = 'blanc'
                else:
                    self.gagnant = 'noir'

                return