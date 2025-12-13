import numpy as np


taille_par_defaut = 6

class Surakarta_olympiad():

    def __init__(self, taille=taille_par_defaut, tour_max=None, repetition_max=0):# 1 veut dire 3 occurence : fin !

        self.dtype = 'int8'  # 'float32' # 'float'

        self.longueur_max = tour_max
        self.longueur_moyenne = 4*taille*taille
        self.tour_max = tour_max

        self.repetition_max = repetition_max


        self.taille = taille
        self.panels = 2
        self.init()

    def init(self):
        self.nb_repet=0

        self.tour_sans_prise=0

        self.hash_old_positions = []

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.tour = 0

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.plateau = np.zeros((self.taille, self.taille,self.panels), dtype=self.dtype)

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

    def hash_plateau(self):
        return self.plateau.tobytes()

    def undo(self):

        if self.hash_old_positions:
            self.hash_old_positions.pop()

        l1, l2, self.blancJoue, prise, self.coups_licites, self.tour_sans_prise, self.nb_repet = self.historique.pop()

        l1a, l1b = l1
        l2a, l2b = l2

        self.tour -= 1

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
        #self.calcul_coups_licites()

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
            print('ouille ... : ',l1,l2,'n est pas dans',self.coupsLicites())
            print(self.blancJoue)
            print(self.plateau[:,:, 0])

            raise Exception('non legal')

        if self.repetition_max is not None:
            self.hash_old_positions.append(self.hash_plateau())

        self.tour += 1

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

        else:
            self.pieces_noir.remove(l1)
            self.pieces_noir.add(l2)
            self.plateau[l2a,l2b,0] = -1
            if l2 in self.pieces_blanc:
                self.pieces_blanc.remove(l2)
                prise = True


        self.historique.append((l1, l2, self.blancJoue, prise, self.coupsLicites(), self.tour_sans_prise, self.nb_repet))

        self.blancJoue = not self.blancJoue

        self.actu_couleur_plateau()
        self.calcul_coups_licites()

        if prise:
            self.tour_sans_prise=0
        else:
            self.tour_sans_prise+=1

        self.test_gagnant()

        if self.fini:
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


    def calcul_coups_licites(self):

        L = []

        if self.blancJoue:

            for i,j in self.pieces_blanc:

                if i < self.taille -1:
                    if self.plateau[i+1,j,0] == 0:
                        L.append(((i,j),(i + 1, j)))
                    if j > 0 and self.plateau[i + 1, j - 1,0] == 0:
                        L.append(((i,j),(i+1,j-1)))
                    if j < self.taille -1 and self.plateau[i + 1, j + 1,0] == 0:
                        L.append(((i, j), (i + 1, j + 1)))

                if i > 0:
                    if self.plateau[i - 1, j,0] == 0:
                        L.append(((i, j), (i - 1, j)))
                    if j > 0 and self.plateau[i - 1, j - 1,0] == 0:
                        L.append(((i, j), (i - 1, j - 1)))
                    if j < self.taille - 1 and self.plateau[i - 1, j + 1,0] == 0:
                        L.append(((i, j), (i - 1, j + 1)))

                if j > 0:
                    if self.plateau[i, j - 1,0] == 0:
                        L.append(((i, j), (i, j - 1)))

                if j < self.taille -1:
                    if self.plateau[i,j+1,0] == 0:
                        L.append(((i, j), (i, j + 1)))

                if (i,j) not in [(0,0),(0,self.taille-1),(self.taille-1,0),(self.taille-1,self.taille-1)]:
                    self.plateau[i, j, 0] = 0
                    boucle, adv, k, l = self.analyse_boucle(i, j, 1, 0, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, -1, 0, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, 0, 1, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, 0, -1, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))
                    self.plateau[i, j, 0] = 1

        else:

            for i, j in self.pieces_noir:

                if i > 0:
                    if self.plateau[i - 1, j,0] == 0:
                        L.append(((i, j), (i - 1, j)))
                    if j > 0 and self.plateau[i - 1, j - 1,0] == 0:
                        L.append(((i, j), (i - 1, j - 1)))
                    if j < self.taille - 1 and self.plateau[i - 1, j + 1,0] == 0:
                        L.append(((i, j), (i - 1, j + 1)))

                if i < self.taille -1:
                    if self.plateau[i+1,j,0] == 0:
                        L.append(((i,j),(i + 1, j)))
                    if j > 0 and self.plateau[i + 1, j - 1,0] == 0:
                        L.append(((i,j),(i+1,j-1)))
                    if j < self.taille -1 and self.plateau[i + 1, j + 1,0] == 0:
                        L.append(((i, j), (i + 1, j + 1)))

                if j > 0:
                    if self.plateau[i, j - 1, 0] == 0:
                        L.append(((i, j), (i, j - 1)))

                if j < self.taille - 1:
                    if self.plateau[i, j + 1, 0] == 0:
                        L.append(((i, j), (i, j + 1)))

                if (i, j) not in [(0, 0), (0, self.taille - 1), (self.taille - 1, 0), (self.taille - 1, self.taille - 1)]:
                    self.plateau[i, j, 0] = 0
                    boucle, adv, k, l = self.analyse_boucle(i, j, 1, 0, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, -1, 0, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, 0, 1, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))

                    boucle, adv, k, l = self.analyse_boucle(i, j, 0, -1, self.blancJoue, i, j)
                    if boucle and adv:
                        L.append(((i, j), (k, l)))
                    self.plateau[i, j, 0] = -1

        self.coups_licites = set(L)

    def in_board(self, i, j):
        return 0 <= i < self.taille and 0 <= j < self.taille

    """
      def analyse_boucle(self, i, j, di, dj, blancJoue, i0, j0, boucle=False, passage_init=-1):

        if i == i0 and j == j0:
            passage_init += 1

            if passage_init == 2:
                assert boucle
                return boucle, False, i, j

        if self.in_board(i+di, j+dj):
            k = i + di
            l = j + dj

            dk = di
            dl = dj
        else:
            if i == 0 and j ==0 or i == 0 and j == self.taille -1 or i == self.taille -1 and j == 0 or i == self.taille - 1 and j == self.taille -1:
                return boucle, False, i, j
            else:
                k, l, dk, dl = self.tour_de_boucle(i, j)

                boucle = True

        v = self.plateau[k, l, 0]
        if v == 0:
            return self.analyse_boucle(k, l, dk, dl, blancJoue, i0, j0, boucle=boucle, passage_init=passage_init)
        else:
            if blancJoue and v == -1 or not blancJoue and v == 1:
                return boucle, True, k, l
            else:
                return False, True, k, l
    """


    def analyse_boucle(self, i, j, di, dj, blancJoue, i0, j0, boucle=False, ):

        assert i == i0 and j == j0

        passage_init = -1

        k, l = i, j

        dk = di
        dl = dj

        p=0
        while passage_init < 2:
            #print(p)
            p+=1
            if p>10:
                print(i0 , j0, k, l, dk, dl, blancJoue, boucle)
                print(self.plateau[:,:,0])

            while self.in_board(k+dk, l+dl):

                if k == i0 and l == j0:
                    passage_init += 1

                k = k + dk
                l = l + dl

                v = self.plateau[k, l, 0]
                if v != 0:
                    if blancJoue and v == -1 or not blancJoue and v == 1:
                        return boucle, True, k, l
                    else:
                        return False, True, k, l

            if k == i0 and l == j0:
                passage_init += 1

            if k == 0 and l ==0 or k == 0 and l == self.taille -1 or k == self.taille -1 and l == 0 or k == self.taille - 1 and l == self.taille -1:
                return boucle, False, k, l
            else:
                k, l, dk, dl = self.tour_de_boucle(k, l)

                boucle = True

            v = self.plateau[k, l, 0]
            if not v == 0:
                if blancJoue and v == -1 or not blancJoue and v == 1:
                    return boucle, True, k, l
                else:
                    return False, True, k, l

        assert boucle
        return boucle, False, k, l

    def tour_de_boucle(self, i, j):

        if i == 0 and j < self.taille / 2:
            return j, i, 0, 1

        if i == 0 and j >= self.taille / 2:
            return self.taille - 1 - j, self.taille - 1 - i, 0 , -1

        if j == 0 and i < self.taille / 2:
            return j, i, 1, 0

        if j == 0 and i >= self.taille / 2:
            return self.taille -1 - j, self.taille - 1 - i,  -1, 0

        if i == self.taille - 1 and j < self.taille / 2:
            return self.taille - 1 - j, self.taille - 1 - i, 0, 1

        if i == self.taille - 1 and j >= self.taille / 2:
            return j, i, 0, -1

        if j == self.taille - 1 and i < self.taille / 2:
            return self.taille - 1 - j, self.taille - 1 - i, 1, 0

        if j == self.taille - 1 and i >= self.taille / 2:
            return j, i, -1, 0



    def coupsLicites(self):
        return list(self.coups_licites)


    def test_gagnant(self):
        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())
        if (self.tour_max is not None and not self.tour < self.tour_max) or not(self.coupsLicites()) or (self.repetition_max is not None and self.nb_repet > self.repetition_max) or self.tour_sans_prise >= 50:
            self.fini = True
            self.score = len(self.pieces_noir) - len(self.pieces_blanc)
            if len(self.pieces_noir) > len(self.pieces_blanc):
                self.gagnant = 'noir'
            elif len(self.pieces_noir) < len(self.pieces_blanc):
                self.gagnant = 'blanc'

            else:
                self.gagnant = 'nul'

                #nb_pieces_init_joueur = 2*self.taille

                #self.score2 = len(self.pieces_noir) / (nb_pieces_init_joueur + 1) - len(self.pieces_blanc) / (nb_pieces_init_joueur + 1) # DEBILE CAR SI NUL meme nb de piece, on peut evaluer par 1 / longueur de partie mais faut-il preferer des parties longues pour avoir plus de chance de gagner ou courte pour remonter plus vite le signal et donner les points au premier ou au second joueur ? En superviser de celui qui met fin à la partie ?! Si doit favoriser partie courte alors les points en faveur de celui qui termine.

        if not self.pieces_noir:
            self.fini = True
            self.gagnant = 'blanc'
            self.score = len(self.pieces_noir) - len(self.pieces_blanc)

        if not self.pieces_blanc:
            self.fini = True
            self.gagnant = 'noir'
            self.score = len(self.pieces_noir) - len(self.pieces_blanc)

    def symetrie_joueur(self, plateau):
        return np.flip(-1*plateau, 0)


if __name__ == '__main__':
    s = Surakarta_olympiad()
    s.plateau = np.zeros(s.plateau.shape)
    S=set()
    for oi, oj in [(i,j) for i in range(6) for j in range(6)]:
        #oi, oj = 2, 5
        #di, dj = 0, 1

        for vi, vj in [(1,0),(-1,0),(0,1),(0,-1)]:
            print()

            i=oi
            j=oj

            while  not (i == 0 and j ==0 or i == 0 and j == s.taille -1 or i == s.taille -1 and j == 0 or i == s.taille - 1 and j == s.taille -1):
                if s.in_board(i+vi,j+vj):
                    i = i+vi
                    j = j + vj
                else:
                    i,j, vi, vj = s.tour_de_boucle(i,j)

                S.add((i,j))
                print(i,j)
                if (i == oi and j == oj):
                    break

        print()
        #print((di,dj)in S)
        print(S)
