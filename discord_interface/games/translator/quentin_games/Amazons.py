from random import choice

import numpy as np

taille_par_defaut = 10

class Amazons():

    def __init__(self, taille=taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.longueur_max = 2*(taille*taille-8)#2*(taille*taille) #> pour compat mes anciennes xp avec prof !
        self.longueur_moyenne = 172 * (taille*taille) / 100

        self.init()


    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 3] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 3] = -np.ones((self.taille, self.taille), dtype=self.dtype)

    def actu_codage_phase_jouer(self):
        if self.phase_un:
            if self.historique:
                #print(self.historique[-2][1])
                i,j = self.historique[-2][1]
                self.plateau[i,j, 2] = 0
        else:
            #print('2',self.historique[-1][1])
            i,j = self.historique[-1][1]
            self.plateau[i,j , 2] = 1

    def actu_codage_phase_undo(self, a, b):

        if self.phase_un:
            i, j = b
            self.plateau[i,j, 2] = 0
        else:
            i,j = self.historique[-1][1]
            self.plateau[i,j, 2] = 1

    def init(self):
        self.tour = 0
        self.phase_un = True
        self.blancJoue = False
        self.historique = []

        self.fini = False
        self.gagnant = None

        if self.taille>6:
            loc = round((self.taille +1) / 4)
        else:
            loc = int((self.taille + 1) / 4)

        self.plateau = np.zeros((self.taille, self.taille,4), dtype=self.dtype) # CODAGE : 0 : les positions des pieces ; 1 : l appartenance des pieces

        self.plateau[0, loc,1] = -1
        self.plateau[loc, 0,1] = -1
        self.plateau[0, -loc-1,1] = -1
        self.plateau[loc, -1,1] = -1

        self.plateau[0, loc, 0] = 1
        self.plateau[loc, 0, 0] = 1
        self.plateau[0, -loc - 1, 0] = 1
        self.plateau[loc, -1, 0] = 1

        self.dames_noir = set()
        self.dames_noir.add((0, loc))
        self.dames_noir.add((loc, 0))
        self.dames_noir.add((0, self.taille-loc-1))
        self.dames_noir.add((loc, self.taille-1))

        self.plateau[-loc - 1, -1, 1] = 1
        self.plateau[-1, -loc - 1, 1] = 1
        self.plateau[-1, loc, 1] = 1
        self.plateau[-loc - 1, 0, 1] = 1

        self.plateau[-loc-1, -1,0] = 1
        self.plateau[-1, -loc-1,0] = 1
        self.plateau[-1, loc,0] = 1
        self.plateau[-loc-1, 0,0] = 1

        self.dames_blanc = set()
        self.dames_blanc.add((self.taille-loc-1, self.taille-1))
        self.dames_blanc.add((self.taille-1, self.taille-loc-1))
        self.dames_blanc.add((self.taille-1, loc))
        self.dames_blanc.add((self.taille-loc-1, 0))

        self.actu_couleur_plateau()
        self.actu_codage_phase_jouer()
        self.calcul_coups_licite()


    def undo(self):
        self.fini = False
        self.gagnant = None

        self.tour -= 1
        if self.phase_un:
            self.blancJoue = not self.blancJoue
        self.phase_un = not self.phase_un

        a,b = self.historique.pop()

        if self.phase_un:
            i,j = a
            k,l = b
            self.plateau[k, l, 0] = 0
            self.plateau[k, l, 1] = 0


            if self.blancJoue:
                self.dames_blanc.add((i, j))
                self.dames_blanc.remove((k, l))
                self.plateau[i, j, 0] = 1
                self.plateau[i, j, 1] = 1
            else:
                self.dames_noir.add((i, j))
                self.dames_noir.remove((k, l))
                self.plateau[i,j,0]= 1
                self.plateau[i, j, 1] = -1
        else:
            self.plateau[a,b,0] = 0

        #print('+')
        self.actu_codage_phase_undo(a,b)
        self.actu_couleur_plateau()
        self.calcul_coups_licite()
        #print('-')


    def raz(self):
        self.init()

    def jouer(self, a, b):

        try:

            if self.phase_un:
                i,j = a
                k,l = b
                self.plateau[i,j,0]=0
                self.plateau[i, j, 1] = 0
                if self.blancJoue:
                    self.dames_blanc.remove((i,j))
                    self.dames_blanc.add((k,l))
                    self.plateau[k,l,0] = 1
                    self.plateau[k, l, 1] = 1
                else:
                    self.dames_noir.remove((i, j))
                    self.dames_noir.add((k, l))
                    self.plateau[k,l,0] = 1
                    self.plateau[k, l, 1] = -1
            else:
                self.plateau[a,b,0] = 1


            self.historique.append((a,b))
            self.tour += 1
            self.phase_un = not self.phase_un
            if self.phase_un:
                self.blancJoue = not self.blancJoue

            self.calcul_coups_licite()
            self.actu_codage_phase_jouer()
            self.actu_couleur_plateau()

            if not self.coupsLicites():
                self.fini = True
                if self.blancJoue:
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
                self.calcul_score()

        except Exception as e:
            print(self.historique)
            raise e

    def calcul_score(self):
        if self.blancJoue:
            self.score2 = self.taille * self.taille - np.count_nonzero(self.plateau[:, :, 0])
            self.score3 = self.score2 / (self.taille * self.taille)

            self.blancJoue = not self.blancJoue
            phase = self.phase_un
            self.phase_un = True
            self.calcul_coups_licite()
            self.score = len(self.coupsLicites())
            self.blancJoue = not self.blancJoue
            self.phase_un = phase
        else:
            self.score2 = np.count_nonzero(self.plateau[:, :, 0]) - self.taille * self.taille
            self.score3 = self.score2 / (self.taille * self.taille)

            self.blancJoue = not self.blancJoue
            phase = self.phase_un
            self.phase_un = True
            self.calcul_coups_licite()
            self.score = -len(self.coupsLicites())
            self.blancJoue = not self.blancJoue
            self.phase_un = phase


    def calcul_coups_licite(self):
        #print(self.phase_un)
        l = []
        if self.phase_un:
            if self.blancJoue:
                for loc in self.dames_blanc:
                    self.ajouter_mouvements_dame(loc, l)
            else:
                for loc in self.dames_noir:
                    self.ajouter_mouvements_dame(loc, l)
        else:
            self.ajouter_mouvements_arrow(self.historique[-1][1],l)

        self.coups_licites = l

    def coupsLicites(self):
        return self.coups_licites

    def ajouter_mouvements_dame(self, loc, l):

        i,j = loc

        d = 1
        while i + d < self.taille and self.plateau[i + d, j, 0] == 0:
            l.append((loc,(i + d, j)))
            d += 1


        d = 1
        while j + d < self.taille and self.plateau[i, j + d, 0] == 0:
            l.append((loc,(i, j + d)))
            d += 1

        d = 1
        while i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d, 0] == 0:
            l.append((loc,(i + d, j + d)))
            d += 1


        d = 1
        while i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d, 0] == 0:
            l.append((loc,(i - d, j + d)))
            d += 1


        d = 1
        while i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d, 0] == 0:
            l.append((loc,(i + d, j - d)))
            d += 1

        d = 1
        while i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d, 0] == 0:
            l.append((loc,(i - d, j - d)))
            d += 1


        d = 1
        while i - d >= 0 and self.plateau[i - d, j, 0] == 0:
            l.append((loc,(i - d, j)))
            d += 1

        d = 1
        while j - d >= 0 and self.plateau[i, j - d, 0] == 0:
            l.append((loc,(i, j - d)))
            d += 1


    def ajouter_mouvements_arrow(self, loc, l):

        i,j = loc

        d = 1
        while i + d < self.taille and self.plateau[i + d, j, 0] == 0:
            l.append((i + d, j))
            d += 1


        d = 1
        while j + d < self.taille and self.plateau[i, j + d, 0] == 0:
            l.append((i, j + d))
            d += 1

        d = 1
        while i + d < self.taille and j + d < self.taille and self.plateau[i + d, j + d, 0] == 0:
            l.append((i + d, j + d))
            d += 1


        d = 1
        while i - d >= 0 and j + d < self.taille and self.plateau[i - d, j + d, 0] == 0:
            l.append((i - d, j + d))
            d += 1


        d = 1
        while i + d < self.taille and j - d >= 0 and self.plateau[i + d, j - d, 0] == 0:
            l.append((i + d, j - d))
            d += 1

        d = 1
        while i - d >= 0 and j - d >= 0 and self.plateau[i - d, j - d, 0] == 0:
            l.append((i - d, j - d))
            d += 1


        d = 1
        while i - d >= 0 and self.plateau[i - d, j, 0] == 0:
            l.append((i - d, j))
            d += 1

        d = 1
        while j - d >= 0 and self.plateau[i, j - d, 0] == 0:
            l.append((i, j - d))
            d += 1

    def bordification(self, p):
        p2 = super().bordification(p)

        p2[:,:,0]+= np.pad(np.zeros((self.taille, self.taille),dtype=self.dtype), ((1,1),(1,1)), constant_values=((1,1),(1,1)), mode='constant')

        return p2

    def symetrie_joueur(self, plateau):
        p =  np.array(plateau, copy=True)

        p[:,:,1] = -p[:,:,1]

        p[:, :, 3] = -p[:, :, 3]

        return p

if __name__  == '__main__':

    jeu = Amazons_couleur()

    i=1

    print(jeu.plateau[:,:,i])
    print(jeu.bordification(jeu.plateau)[:,:,i])
    print()

    jeu.jouer(*choice(jeu.coupsLicites()))

    print(jeu.plateau[:, :, i])
    print(jeu.bordification(jeu.plateau)[:, :, i])
    print()

    jeu.jouer(*choice(jeu.coupsLicites()))

    print(jeu.plateau[:, :, i])
    print(jeu.bordification(jeu.plateau)[:, :, i])
    print()

    jeu.jouer(*choice(jeu.coupsLicites()))

    print(jeu.plateau[:, :, i])
    print(jeu.bordification(jeu.plateau)[:, :, i])
    print()