import numpy as np


taille_par_defaut = 8

class Havannah_swap_8():

    def __init__(self, taille = taille_par_defaut):

        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = 2*taille-1
        self.cote = taille

        self.longueur_max = self.taille * self.taille - self.cote * (self.cote -1)
        self.longueur_moyenne = self.longueur_max / 2

        self.raz()
        #print(self.longueur_max)



    def undo(self,):

        (i, j), self.blancJoue = self.historique.pop()

        if (i, j) == self.swapped:
            self.swapped = None

            self.plateau[:, :, 1] = np.zeros((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[i, j, 0] = 0

            if len(self.historique) == 1:
                self.coups_licites.add(self.historique[0][0])

        self.fini = False

        self.gagnant = None

        self.coups_licites.add((i,j))


    def raz(self):

        self.swapped = None

        self.gagnant = None

        self.fini = False

        self.blancJoue = False

        self.plateau = np.zeros((self.taille, self.taille, 2), dtype=self.dtype)

        self.historique = []

        self.coups_licites = set([(i,j) for i in range(self.taille) for j in range(self.taille) if i + j >= self.cote - 1 and (self.taille - 1 - i) + (self.taille - 1 - j) >= self.cote - 1])
        #print(len(self.coups_licites))

    def a_swapper(self):
        return self.swapped is not None

    def jouer(self, i, j):
        #print(i,j,' ',self.historique, self.coups_licites)

        if self.plateau[i, j,0] !=0:
            if len(self.historique) == 1 and not self.swapped:

                self.historique.append(((i, j), self.blancJoue))

                self.swapped = (i, j)
                #self.plateau[i, j,0] = - self.plateau[i, j,0]

                self.plateau[:,:,1] = np.ones((self.taille, self.taille), dtype=self.dtype)

            else:
                raise Exception('ho, du con !')
        else:
            self.historique.append(((i,j), self.blancJoue))

            if self.blancJoue and not self.swapped or not self.blancJoue and self.swapped :
                self.plateau[i, j,0] = 1
            else:
                self.plateau[i, j,0] = -1


        if len(self.historique) != 1:
            self.coups_licites.remove((i, j))
        if len(self.historique) == 2 and not self.swapped:
            #print(self.historique[-2])
            self.coups_licites.remove(self.historique[-2][0])

        self.test_fini(i, j)


        self.blancJoue = not self.blancJoue



    def coupsLicites(self):
        return list(self.coups_licites)




    # note : famille de classe d'equivalence pour chaque joueur et un classe d eq pour chaque bord (noeud) en plus.


    def test_fini(self, i, j):
        #print(self.plateau)

        if not self.coupsLicites():
            self.fini = True
            self.gagnant = 'nul'
            return

        bords_atteint = set()
        coins_atteint = set()

        vs = {}

        bord = self.est_bord(i, j)
        if bord:
            bords_atteint.add(bord)

        if self.est_coin(i, j):
            coins_atteint.add((i, j))

        if self.blancJoue and not self.swapped or not self.blancJoue and self.swapped:

            val = 1

            ouvert = set()
            ouvert.add((i,j))
            fermee = set()

            while ouvert:

                ouvert2 = set([])

                fermee |= ouvert

                for (s, t) in ouvert :

                    if (s,t) not in vs:
                        vs[s,t]= set()

                    for (l, k) in self.voisins(s, t, val):

                        vs[s,t].add((l, k))

                        if (l,k) not in fermee :
                            ouvert2.add((l,k))

                            bord = self.est_bord(l, k)
                            if bord:
                                bords_atteint.add(bord)

                            if self.est_coin(l, k):
                                coins_atteint.add((l, k))

                            #print(len(bords_atteint), len(coins_atteint))
                            if len(bords_atteint) >= 3 or len(coins_atteint) >= 2:
                                self.fini = True
                                if self.swapped:
                                    self.gagnant = "noir"
                                else:
                                    self.gagnant = "blanc"
                                return
                ouvert = ouvert2

            if self.contient_cycle(fermee, vs):
                self.fini = True
                if self.swapped:
                    self.gagnant = "noir"
                else:
                    self.gagnant = "blanc"


        else:

            val = -1

            ouvert = set()
            ouvert.add((i, j))
            fermee = set()

            while ouvert:

                ouvert2 = set([])

                fermee |= ouvert

                for (s, t) in ouvert:

                    if (s,t) not in vs:
                        vs[s,t]= set()

                    for (l, k) in self.voisins(s, t, val):

                        vs[s,t].add((l, k))

                        if (l, k) not in fermee:
                            ouvert2.add((l, k))

                            bord = self.est_bord(l, k)
                            if bord:
                                bords_atteint.add(bord)

                            if self.est_coin(l, k):
                                coins_atteint.add((l, k))


                            if len(bords_atteint) >= 3 or len(coins_atteint) >= 2:

                                self.fini = True
                                if self.swapped:
                                    self.gagnant = "blanc"
                                else:
                                    self.gagnant = "noir"
                                return

                ouvert = ouvert2

            if self.contient_cycle(fermee, vs):
                self.fini = True
                if self.swapped:
                    self.gagnant = "blanc"
                else:
                    self.gagnant = "noir"

    def contient_cycle(self, ensemble, vs):

        #print(vs)
        #print(ensemble)

        retirer = True
        while retirer:

            retirer = False

            for e in list(ensemble):
                if not self.cycle_partiel(e, vs):
                    ensemble.remove(e)
                    for e2 in vs[e]:
                        vs[e2].remove(e)
                    retirer = True

        if len(ensemble) >=6:
            return True
        else:
            assert not ensemble
            """if ensemble:
                print('!',ensemble)"""
            return False

    def cycle_partiel(self, e, vs):

        if len(vs[e]) >=3:
            return True

        if len(vs[e]) == 2:
            (i,j), (l, k) = vs[e]
            return not (i,j) in vs[l,k]#not ((l == i and (k == j + 1 or k == j - 1)) or (k == j and (l == i + 1 or l == i - 1)) or (k == j + 1 and l == i - 1) or (k == j - 1 and l == i + 1))

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


    def est_coin(self, i, j):
        return (i,j) in [(0, self.taille-1), (self.taille-1, 0), (0, self.cote-1), (self.cote-1, 0), (self.taille-1, self.taille-1-(self.cote-1)), (self.taille-1-(self.cote-1), self.taille-1)]

    def est_bord(self, i, j):
        if self.est_coin(i,j):
            return 0
        if i + j == self.cote - 1:
            return 1
        elif (self.taille - 1 - i) + (self.taille - 1 - j) == self.cote - 1:
            return 2
        elif i == 0:
            return 3
        elif j ==0:
            return 4
        elif i == self.taille-1:
            return 5
        elif j==self.taille-1:
            return 6
        return 0

    def symetrie_joueur(self, plateau):
        raise NotImplemented