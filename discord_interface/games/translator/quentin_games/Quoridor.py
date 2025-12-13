from heapq import heapify, heappop, heappush

import numpy as np

taille_par_defaut = 9

class Quoridor(): # https://www.gigamic.com/files/catalog/products/rules/quoridor-classic-fr.pdf

    def __init__(self, taille = taille_par_defaut, borne = None):


        self.dtype = 'int8'  # 'float32' # 'float'

        self.taille = taille

        self.longueur_max = taille * taille
        self.longueur_moyenne = self.longueur_max / 2

        self.raz()

        self.borne = borne

    def actu_codage_extra(self):
        self.plateau[:, :, -1] = self.extra * np.ones((2 * self.taille - 1, 2 * self.taille - 1), dtype=self.dtype)

    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, -2] = np.ones((2*self.taille-1, 2*self.taille-1), dtype=self.dtype)
        else:
            self.plateau[:, :, -2] = -np.ones((2*self.taille-1, 2*self.taille-1), dtype=self.dtype)



    def raz(self):
        self.score = None

        self.extra = 0

        self.tour = 0
        self.pion_noir = (0, 4*2)
        self.pion_blanc = (8*2, 4*2)

        self.barrieres_noires = 0
        self.barrieres_blanches = 0

        self.gagnant = None

        self.fini = False

        self.blancJoue = False

        self.plateau = np.zeros((2*self.taille-1, 2*self.taille-1, 3+1+2), dtype=self.dtype)

        self.plateau[self.pion_noir[0],self.pion_noir[1],0] = -1
        self.plateau[self.pion_blanc[0],self.pion_blanc[1],0] = 1

        self.historique = [] # deque() # les opérations associées sont plus rapides mais le total est pourtant plus lent ???

        self.coups_licites_mur = set([((i,j),(i+2,j)) for i in range(2*self.taille-3) for j in range(2*self.taille-1) if i % 2 == 0 and j % 2 != 0]+ [((i,j),(i,j+2)) for i in range(2*self.taille-1) for j in range(2*self.taille-3) if i % 2 != 0 and j % 2 == 0]) # i % 2 != 0 and j % 2 == 0 or # and j % 2 == 0 or i % 2 == 0 and j % 2 != 0

        self.calcul_coups_licites()

        self.actu_couleur_plateau()
        self.actu_codage_extra()


    def undo(self,):

        self.score = None

        self.tour -=1
        (i,j), (k,l), self.coups_licites, self.pion_noir, self.pion_blanc, self.coups_licites_mur, self.extra = self.historique.pop()

        if not self.coups_licites:
            self.need_calcul_coups_licites = True
        else:
            self.need_calcul_coups_licites = False

        self.blancJoue = not self.blancJoue
        self.actu_couleur_plateau()
        self.actu_codage_extra()

        self.fini = False

        self.gagnant = None


        if not self.blancJoue:
            pion = self.pion_noir
            v = -1
        else:
            pion = self.pion_blanc
            v = 1

        if (i,j) == pion:
            self.plateau[k, l, 0] = 0
            self.plateau[i, j, 0] = v
        else:
            self.plateau[i, j, 1] = 0
            self.plateau[k, l, 1] = 0

            if self.blancJoue:
                self.barrieres_blanches -= 1
                self.plateau[:,:,-3]-= 1
            else:
                self.barrieres_noires -= 1
                self.plateau[:,:,-4]-= 1


    def jouer(self, a, b):

        #print(self.tour)
        self.tour +=1

        (i,j), (k,l) = a, b

        lm = list(self.coups_licites_mur)
        self.historique.append((a,b, list(self.coups_licites), self.pion_noir, self.pion_blanc, lm, self.extra))

        if not self.blancJoue:
            pion = self.pion_noir
            v = -1
        else:
            pion = self.pion_blanc
            v = 1
        #print(a,b,'>',pion)
        if (i, j) == pion:
            self.plateau[i, j, 0] = 0
            self.plateau[k, l, 0] = v

            if self.blancJoue:
                self.pion_blanc = (k,l)
            else:
                self.pion_noir = (k, l)

            self.extra+=1
        else:

            assert self.plateau[i, j, 1] == 0 and self.plateau[k, l, 1] == 0
            self.plateau[i, j, 1] = 1
            self.plateau[k, l, 1] = 1

            if self.blancJoue:
                self.barrieres_blanches += 1
                self.plateau[:,:,-3] += 1
            else:
                self.barrieres_noires += 1
                self.plateau[:,:,-4] += 1

            for m1, m2 in lm:
                if m1 == a or m1 == b or m2 == a or m2 == b:
                    self.coups_licites_mur.remove((m1, m2))

            dual = self.mur_dual(a, b)
            if dual in self.coups_licites_mur:
                self.coups_licites_mur.remove(dual)

            self.extra = 0


        self.test_fini()

        self.blancJoue = not self.blancJoue
        self.actu_couleur_plateau()

        self.actu_codage_extra()

        #self.calcul_coups_licites()
        self.need_calcul_coups_licites = True
        self.coups_licites = []

    def mur_dual(self, c1, c2):
        (i1, j1), (i2, j2) = c1, c2

        if i1 == i2:
            return (i1 - 1, min(j1, j2)+1), (i1 + 1, min(j1, j2)+1)
        else:
            return (min(i1, i2) + 1, j1 - 1, ), (min(i1, i2) + 1, j1 + 1)



    def in_board(self, i, j):
        return i is not None and j is not None and 0 <= i < 2*self.taille-1 and 0 <= j < 2*self.taille-1

    def calcul_coups_licites(self):

        self.need_calcul_coups_licites = False

        L = []
        if self.blancJoue:
            i, j = self.pion_blanc
            pion_adv = self.pion_noir
            ok_barriere = self.barrieres_blanches < 10
        else:
            i, j = self.pion_noir
            pion_adv = self.pion_blanc
            ok_barriere = self.barrieres_noires < 10

        for p in [(i,j+2),(i,j-2),(i+2,j),(i-2,j)]:
            if self.in_board(*p):
                if not self.barriere_entre((i, j), p):
                    if p != pion_adv:
                            L.append(((i,j),p))
                    else:
                        o = self.opposer((i,j), p)
                        if self.in_board(*o) and not self.barriere_entre(p, o):
                            L.append(((i, j), o))
                        else:
                            for pb in self.perpendiculaires((i, j), p):
                                if  self.in_board(*pb) and not self.barriere_entre(p, pb):
                                    L.append(((i, j), pb))

        L = list(set(L))
        if ok_barriere:
            for m in self.coups_licites_mur:
                if self.existe_chemin(m):
                    L.append(m)
        self.coups_licites = L

    def coordonnee_barriere_entre(self, p1, p2):
        i1, j1 = p1
        i2, j2 = p2

        if i1 == i2:
            if j1 < j2:
                return i1, j1+1
            else:
                return i1, j2 + 1
        else:
            assert j1 == j2
            #print(i1,i2)
            if i1 < i2:
                return i1+1, j1
            else:
                return i2 + 1, j2

    def barriere_entre(self, p1, p2):
        return self.plateau[self.coordonnee_barriere_entre(p1, p2)+ (1,)] == 1


    def opposer(self, p, centre):
        i, j = p
        o1, o2 = centre

        pb = i + 2*(o1 - i), j + 2*(o2 - j)
        if self.in_board(*pb):
            return pb
        else:
            return None, None

    def perpendiculaires(self, p, centre):
        i, j = p
        o1, o2 = centre

        L=[]

        if i == o1:
            for pb in [(i+2, o2), (i-2, o2)]:
                if self.in_board(*pb):
                    L.append(pb)
        else:
            assert j == o2
            for pb in [(o1, j+2), (o1, j-2)]:
                if self.in_board(*pb):
                    L.append(pb)
        return L

    def coupsLicites(self):
        if self.need_calcul_coups_licites: # calcul intime
            self.calcul_coups_licites()
        return list(self.coups_licites)



    # note : famille de classe d'equivalence pour chaque joueur et un classe d eq pour chaque bord (noeud) en plus.

    def seuil_fin(self):
        return  100

    def fin_secondaire(self):
        return self.extra >= self.seuil_fin() or self.borne and self.tour >= self.borne

    def test_fini(self):

        """if self.tour > 1000:
            print(self.plateau[:,:,0])
            print()
            print(self.plateau[:, :, 1])"""


        if not self.blancJoue and self.pion_blanc[0] == 0:
            self.fini = True
            self.gagnant = 'blanc'
            self.score = -10 + self.pion_noir[0]//2 - 8

        if self.blancJoue and self.pion_noir[0] == 8*2:
            self.fini = True
            self.gagnant = 'noir'
            self.score = 10 + self.pion_blanc[0]//2

        if not self.fini:
            if self.fin_secondaire():
                self.fini = True

                self.score = self.pion_blanc[0] // 2 + self.pion_noir[0] // 2 - 8

                if 8*2 - self.pion_noir[0] < self.pion_blanc[0]:
                    self.gagnant = 'noir'


                elif 8*2 - self.pion_noir[0] > self.pion_blanc[0]:
                    self.gagnant = 'blanc'

                else:
                    self.gagnant = 'nul'


    def extremiter_mur_adjacent(self, mur):

        L = []

        (i1, j1), (i2, j2) = mur

        if i1 == i2:

            for p in [min(j1, j2)-2,max(j1, j2)+2]:
                if self.in_board(i1, p):
                    L.append((i1, p))

            for p in [i1 - 1, i1 + 1]:
                for pj in [min(j1, j2)-1, min(j1, j2)+1, max(j1, j2)+1]:
                    if self.in_board(p, pj):
                        L.append((p, pj))
        else:
            assert j1 == j2

            for p in [min(i1, i2)-2,max(i1, i2)+2]:
                if self.in_board(p, j1):
                    L.append((p, j1))

            for p in [j1 - 1, j1 + 1]:
                for pi in [min(i1, i2) - 1, min(i1, i2) + 1, max(i1, i2) + 1]:
                    if self.in_board(pi, p):
                        L.append((pi, p))

        return L

    def a_un_bord_adjacent(self, mur):

        (i1, j1), (i2, j2) = mur

        if i1 == i2:

            return min(j1, j2) == 0 or max(j1, j2) == 2 * self.taille - 2
        else:
            assert j1 == j2

            return min(i1, i2) == 0 or max(i1, i2) == 2 * self.taille - 2

    def murs_adjacent(self, mur):
        L = []

        (i1, j1), (i2, j2) = mur

        if i1 == i2:

            for p1, p2 in [(min(j1, j2) - 4, min(j1, j2) - 2), (max(j1, j2) + 2, max(j1, j2) + 4, )]:
                if self.in_board(i1, p1) and self.in_board(i1, p2) and self.plateau[i1, p1, 1] == 1 and self.plateau[i1, p2, 1] == 1:
                    L.append(((i1, p1), (i1, p2)))

            for p1, p2 in [(i1 - 1, i1 + 1)]:
                for pj in [min(j1, j2) - 1, max(j1, j2) + 1]:
                    if self.in_board(p1, pj) and self.in_board(p2, pj) and self.plateau[p1, pj, 1] == 1 and self.plateau[p2, pj, 1] == 1:
                        L.append(((p1, pj),(p2, pj)))

            for p1, p2 in [(i1 - 3, i1 - 1, ), (i1 + 1, i1 + 3)]:
                for pj in [min(j1, j2) - 1, min(j1, j2) + 1, max(j1, j2) + 1]:
                    if self.in_board(p1, pj) and self.in_board(p2, pj) and self.plateau[p1, pj, 1] == 1 and self.plateau[p2, pj, 1] == 1:
                        L.append(((p1, pj), (p2, pj)))
        else:
            assert j1 == j2


            for p1, p2 in [(min(i1, i2) - 4, min(i1, i2) - 2), (max(i1, i2) + 2, max(i1, i2) + 4, )]:
                if self.in_board(p1, j1) and self.in_board(p2, j1) and self.plateau[p1, j1, 1] == 1 and self.plateau[p2, j1, 1] == 1:
                    L.append(((p1, j1), (p2, j1)))

            for p1, p2 in [(j1 - 1, j1 + 1)]:
                for pi in [min(i1, i2) - 1, max(i1, i2) + 1]:
                    if self.in_board(pi, p1) and self.in_board(pi, p2) and self.plateau[pi, p1, 1] == 1 and self.plateau[pi, p2, 1] == 1:
                        L.append(((pi, p1),(pi, p2)))

            for p1, p2 in [(j1 - 3, j1 - 1, ), (j1 + 1, j1 + 3)]:
                for pi in [min(i1, i2) - 1, min(i1, i2) + 1, max(i1, i2) + 1]:
                    if self.in_board(pi, p1) and self.in_board(pi, p2) and self.plateau[pi, p1, 1] == 1 and self.plateau[pi, p2, 1] == 1:
                        L.append(((pi, p1),(pi, p2)))

        return L


    def existe_chemin(self, mur, opt=True):

        if opt:
            if len([m for m in self.extremiter_mur_adjacent(mur) if self.plateau[m+(1,)] == 1]) + self.a_un_bord_adjacent(mur) < 2:
                return True

            if len([m for m in self.murs_adjacent(mur) if (len([m2 for m2 in self.extremiter_mur_adjacent(m) if self.plateau[m2+(1,)] == 1]) + self.a_un_bord_adjacent(m) >= 1)]) + self.a_un_bord_adjacent(mur) < 2:

                '''
                ok = self.existe_chemin(mur, opt=False)
                if not ok:
                    print(mur, [m for m in self.murs_adjacent(mur) if (len([m2 for m2 in self.extremiter_mur_adjacent(m) if self.plateau[m2 + (1,)] == 1]) + self.a_un_bord_adjacent(m) >= 1)],
                          self.a_un_bord_adjacent(mur),  [(m,self.a_un_bord_adjacent((m))) for m in self.murs_adjacent(mur)])

                    print(self.plateau[:,:,0])
                    print(self.plateau[:,:,1])
                assert ok
                '''

                return True

        ok = {}

        ok[False]=False
        ok[True]=False
        #print()
        for piece, piece_adv, arrivee, blanc in [(self.pion_noir, self.pion_blanc, 16, False), (self.pion_blanc, self.pion_noir, 0, True)]:

            if blanc:
                analyser = set()
                tas = [piece]
                heapify(tas)

                while tas and not ok[blanc]:
                    #print(tas)
                    s, t = heappop(tas)
                    #print(s,t)

                    analyser.add((s, t))

                    for (l, k) in self.voisins(s, t, piece_adv, mur):

                        if (l, k) not in analyser:

                            if l == arrivee:
                                ok[blanc] = True
                                break

                            heappush(tas, (l, k))


            else:
                #print()
                analyser = set()
                tas = [(-piece[0],piece[1])]
                heapify(tas)

                while tas and not ok[blanc]:
                    #print(tas)
                    ns, t = heappop(tas)
                    #print(ns,t)
                    analyser.add((ns, t))

                    for (l, k) in self.voisins(-ns, t, piece_adv, mur):
                        nl = -l
                        if (nl, k) not in analyser:

                            if l == arrivee:
                                ok[blanc] = True
                                break

                            heappush(tas, (nl, k))

        return ok[True] and ok[False]


    def voisins(self, i, j, pion_adv, mur):
        L = []
        for p in [(i, j + 2), (i, j - 2), (i + 2, j), (i - 2, j)]:
            if self.in_board(*p):

                if not self.barriere_entre((i, j), p) and self.coordonnee_barriere_entre((i, j), p) != mur[0] and self.coordonnee_barriere_entre((i, j), p) != mur[1]:
                    L.append(p)
                    """if p != pion_adv:
                            L.append(p)
                    else:
                        o = self.opposer((i, j), p)
                        if self.in_board(*o) and not self.barriere_entre(p, o) and self.coordonnee_barriere_entre(p, o) != mur[0] and self.coordonnee_barriere_entre(p, o) != mur[1]:
                            L.append(o)
                        else:
                            for pb in self.perpendiculaires((i, j), p):
                                if self.in_board(*pb) and not self.barriere_entre(p, pb) and self.coordonnee_barriere_entre(p, pb) != mur[0] and self.coordonnee_barriere_entre(p, pb) != mur[1]:
                                    L.append(pb)"""

        return L

