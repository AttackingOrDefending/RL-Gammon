import numpy as np


class Arimaa_vrai():

    def __init__(self, borne=900, defaite_si_repet=True, repetition_max=2, num_type_hash=0, codage_piege=True):#

        self.codage_extras = 0

        self.taille = 8

        self.dtype = 'int8'  # 'float32' # 'float'

        self.borne = borne
        self.defaite_si_repet = defaite_si_repet
        self.repetition_max = repetition_max
        self.num_type_hash = num_type_hash

        assert self.repetition_max is None or self.repetition_max >= 0

        if borne:
            self.longueur_max = borne
            self.longueur_moyenne = self.longueur_max/10
        else:
            self.longueur_max = 500
            self.longueur_moyenne = self.longueur_max / 10

        self.positions_initiales_noir = [(self.taille-1,i) for i in range(self.taille)] + [(self.taille-2,i) for i in range(self.taille)]
        self.positions_initiales_blanc = [(0,i) for i in range(self.taille)] + [(1,i) for i in range(self.taille)]


        self.codage_ultra_compact = False

        self.codage_piege = codage_piege
        self.init()

        self.valeur_noir_max = 37
        self.valeur_blanc_max = 37
        self.nb_pieces_blanc_init = 16
        self.nb_pieces_noir_init = 16

        self.valeur_noir_max_5 = 69
        self.valeur_blanc_max_5 = 69




    def init(self):
        self.nb_repet = 0

        self.lapins_blanc_pris = 0

        self.lapins_noir_pris = 0

        self.positions_libre_noir = list(self.positions_initiales_noir)
        self.positions_libre_blanc = list(self.positions_initiales_blanc)

        self.pas = 0
        self.phase = 'placement'

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.liste_positions_placement_blanc = set()
        self.pieces_a_poser_blanc = [1,]*8+[2,]*2+[3,]*2+[4,]*2+[5,]+[6,]
        self.liste_positions_placement_noir  = set()
        self.pieces_a_poser_noir = [-1,]*8+[-2,]*2+[-3,]*2+[-4,]*2+[-5,]+[-6,]


        self.hash_old_positions_noir = []
        self.hash_old_positions_blanc = []

        self.positions_vides = set([(i,j) for i in range(self.taille) for j in range(self.taille)])

        self.tour = 0

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.type_pieces = {}


        self.init_plateau()

        self.historique = []



        self.fini = False
        self.gagnant = None

        self.calcul_coups_licites()
        #print(self.coups_licites)
        self.actu_couleur_plateau()


    def repriorisation(self, i):
        return (i-2) % 6 + 1

    def init_plateau(self):
        if self.codage_ultra_compact:
            raise
            self.plateau = np.zeros((self.taille, self.taille, 3 + self.codage_piege + self.codage_extras), dtype=self.dtype)
        else:
            self.plateau = np.zeros((self.taille, self.taille, 6 + 3 + self.codage_piege + self.codage_extras), dtype=self.dtype)

        self.encodage_pieges()

    def encodage_pieges(self):
        if self.codage_piege:
            self.plateau[2, 2, -4-self.codage_extras] = 1
            self.plateau[2, self.taille-3, -4-self.codage_extras] = 1
            self.plateau[self.taille-3, 2, -4-self.codage_extras] = 1
            self.plateau[self.taille-3, self.taille-3, -4-self.codage_extras] = 1


    def placer(self, piece, i, j, blanc):

        if blanc:
            self.pieces_blanc.add((i, j))
        else:
            self.pieces_noir.add((i, j))

        self.type_pieces[i, j] = piece

        self.positions_vides.remove((i, j))

        if self.codage_ultra_compact:
            if blanc:
                self.plateau[i, j, 0] = abs(piece)
            else:
                self.plateau[i, j, 0] = -abs(piece)
        else:
            if blanc:
                self.plateau[i, j, abs(piece)-1] = 1
            else:
                self.plateau[i, j, abs(piece)-1] = -1
        #print( piece, i, j, blanc)
        return 'deplacement', not self.blancJoue

    def retirer(self, i, j):
        self.pieces_noir.discard((i,j))
        self.pieces_blanc.discard((i, j))

        self.positions_vides.add((i,j))

        if self.codage_ultra_compact:
            self.plateau[i,j,0] = 0
        else:
            self.plateau[i, j, abs(self.type_pieces[i,j])-1] = 0

        del self.type_pieces[i, j]


    """def pousser(self):


    def tirer(self):"""

    def deplacer(self, i, j, k, l, blancJoue):
        piece = self.type_pieces[i, j]

        self.retirer(i, j)
        self.placer(piece, k, l, blancJoue)



    def actu_couleur_plateau(self):


        self.plateau[:, :, -2-self.codage_extras] = self.pas * np.ones((self.taille, self.taille), dtype=self.dtype)

        if self.blancJoue:
            self.plateau[:, :, -1-self.codage_extras] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, -1-self.codage_extras] = - np.ones((self.taille, self.taille), dtype=self.dtype)

        """if self.phase == 'placement':
            self.plateau[:, :, -1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, -] = np.zeros((self.taille, self.taille), dtype=self.dtype)"""



    def raz(self):
        self.init()

    def hash_plateau(self):
        return self.plateau.tobytes()

    def hash_plateau_separer(self):
        if self.num_type_hash == 0:
            #print('ndim', self.plateau[:,:,:-3-self.codage_piege].shape)
            return self.plateau[:,:,:6].tobytes()
        elif self.num_type_hash == 1: # risque de collision mais peut-être un gain de perf d'environ 2%
            raise
            return hash(self.plateau[:,:,:6].tobytes())
        """elif self.num_type_hash == 2:# risque de collision et 100 fois plus lent ...
            return hash(str(self.plateau))"""


    def jouer(self, a, b):

        bj = self.blancJoue

        if not (a,b) in self.coupsLicites():
            print()
            for h in self.historique:
                print(h[0:2])
            raise Exception(str(a)+','+str(b)+' pas dans coups licites : '+str(self.coupsLicites()))

        self.tour += 1

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1


        cases_noires =  list(self.positions_libre_noir)
        cases_blanches =  list(self.positions_libre_blanc)

        if self.phase == 'placement':

            if len(self.positions_libre_noir):
                #print('>',self.pieces_a_poser_noir[self.tour-1])
                self.placer(self.pieces_a_poser_noir[self.tour-1], a, b, self.blancJoue)

                #print(self.positions_libre_noir, (a,b))
                #print(self.coups_licites)
                #print()
                self.positions_libre_noir.remove((a,b))
                assert not self.blancJoue

            else:
                if len(self.positions_libre_blanc):
                    self.placer(self.pieces_a_poser_blanc[self.tour-1-16], a, b, self.blancJoue)
                    self.positions_libre_blanc.remove((a, b))
                    assert self.blancJoue
                else:
                    raise


            self.historique.append((a, b, self.blancJoue, self.phase, self.coups_licites, cases_noires, cases_blanches, self.lapins_noir_pris, self.lapins_blanc_pris,  self.pas, [], None, self.nb_repet))
            if not len(self.positions_libre_blanc):
                self.phase = 'deplacement'
                self.pas = 1
                self.blancJoue = not self.blancJoue

                self.actu_couleur_plateau()
                self.first_state = self.get_first_state()
                self.actu_codage_first_state()


            if not len(self.positions_libre_noir) and len(self.positions_libre_blanc) == 16:
                self.blancJoue = not self.blancJoue

                self.actu_couleur_plateau()

            self.calcul_coups_licites()

            self.nb_repet = 0

        else:

            if self.pas == 1 and self.repetition_max is not None and a is not None:
                if self.blancJoue:
                    self.hash_old_positions_blanc.append(self.hash_plateau_separer())
                else:
                    self.hash_old_positions_noir.append(self.hash_plateau_separer())

            if a == None and b == None:
                cout = 5-self.pas
                i1, j1 = None, None

            elif isinstance(b[0], tuple):

                (i, j) = a
                (i0, j0), (i1, j1) = b
                cout = 2


                self.deplacer(i0, j0, i1, j1, self.blancJoue)
                self.deplacer(i, j, i0, j0, not self.blancJoue)

            elif isinstance(a[0], tuple):

                (i, j) = b
                (i0, j0), (i1, j1) = a
                cout = 2

                self.deplacer(i1, j1, i, j, not self.blancJoue)
                self.deplacer(i0, j0, i1, j1, self.blancJoue)

            else:
                cout = 1

                i0, j0 = a
                i1, j1 = b

                assert self.est_vide(*b)

                self.deplacer(i0,j0,i1,j1, self.blancJoue)

            lapins_noir_pris, lapins_blanc_pris = self.lapins_noir_pris, self.lapins_blanc_pris
            capturer = self.captures()
            self.historique.append((a, b, self.blancJoue, self.phase, self.coups_licites, list(self.positions_libre_noir), list(self.positions_libre_blanc), lapins_noir_pris, lapins_blanc_pris, self.pas, capturer, self.first_state, self.nb_repet))


            self.pas += cout

            assert self.pas < 6

            if self.pas == 5:
                self.blancJoue = not self.blancJoue
                self.pas = 1

                self.actu_couleur_plateau()

                if self.blancJoue:# et pas bj car on la liste correspond à celui qui va jouer et pas celui qui vient de jouer
                    self.nb_repet = self.hash_old_positions_blanc.count(self.hash_plateau_separer())
                else:
                    self.nb_repet = self.hash_old_positions_noir.count(self.hash_plateau_separer())

            else:

                self.actu_couleur_plateau()

            self.calcul_coups_licites()

            self.test_fini(bj)

            if self.pas == 1:
                self.first_state = self.get_first_state()
                self.actu_codage_first_state()


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
        #print(self.plateau[:, :, 0] + self.plateau[:, :, 1] + self.plateau[:, :, 2] + self.plateau[:, :, 3] + self.plateau[:, :, 4] + self.plateau[:, :, 5])
        #print(self.pas)
        #print(self.blancJoue)

        #print(self.coupsLicites())

    def get_first_state(self):
        return  np.copy(self.plateau[:,:,:6])

    def actu_codage_first_state(self):
        if self.first_state is not None:
            self.plateau[:, :, -3-self.codage_extras] = self.first_state[:, :, 0] + 2 * self.first_state[:, :, 1] + 3 * self.first_state[:, :, 2] + 4 * self.first_state[:, :, 3] + 5 * self.first_state[:, :, 4] + 6 * self.first_state[:, :, 5]
        else:
            self.plateau[:, :, -3-self.codage_extras] = np.zeros((self.taille, self.taille))

    def undo(self, ):
        self.tour -= 1


        pas = self.pas

        a, b, self.blancJoue, self.phase, self.coups_licites, self.positions_libre_noir, self.positions_libre_blanc, self.lapins_noir_pris, self.lapins_blanc_pris, self.pas, capturer, self.first_state, self.nb_repet = self.historique.pop()

        if pas == 1:
            self.actu_codage_first_state()

        if self.pas ==1:
            if self.blancJoue  and self.hash_old_positions_blanc:
                self.hash_old_positions_blanc.pop()
            if not self.blancJoue and self.hash_old_positions_noir:
                self.hash_old_positions_noir.pop()

        #print(self.coups_licites)
        for p, i, j in capturer:
            self.placer(p, i, j, p > 0)


        if self.phase == 'deplacement':

            if a == None and b == None:
                """"""
            elif isinstance(b[0], tuple):

                (i, j) = a
                (i0, j0), (i1, j1) = b

                self.deplacer(i0, j0, i, j, not self.blancJoue)
                self.deplacer(i1, j1, i0, j0, self.blancJoue)

            elif isinstance(a[0], tuple):

                (i, j) = b
                (i0, j0), (i1, j1) = a

                self.deplacer(i1, j1, i0, j0, self.blancJoue)

                self.deplacer(i, j, i1, j1, not self.blancJoue)


            else:

                i0, j0 = a
                i1, j1 = b

                self.deplacer(i1, j1, i0, j0, self.blancJoue)


        elif self.phase == 'placement':

            self.retirer(a,b)




        self.fini = False
        self.gagnant = None

        self.actu_couleur_plateau()

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1

    def est_vide(self, i, j):
        return (i, j) in self.positions_vides

    def est_piece_blanc(self, i, j):
        return (i,j) in self.pieces_blanc

    def est_piece_noir(self, i, j):
        return (i, j) in self.pieces_noir


    def appliquer_deplacement(self, state, coup):

        (i, j), (k, l) = coup

        state[k,l] = state[i, j]
        state[i, j] = 0

        return state

    def appliquer_tirage(self, state, coup):

        (I, J), ((i, j), (k, l)) = coup

        state[k, l] = state[i, j]
        state[i, j] = state[I, J]
        state[I, J] = 0

        return state

    def appliquer_poussage(self, state, coup):

        ((i, j), (k, l)), (I, J) = coup

        state[I, J] = state[k, l]
        state[k, l] = state[i, j]
        state[i, j] = 0

        return state

    def calcul_coups_licites(self):
        L = []

        if self.phase == 'deplacement':


            if self.pas > 1:
                if self.pas > 2:
                    if not np.array_equal(self.first_state, self.get_first_state()):
                        L.append((None, None))
                else:
                    L.append((None, None))

            if self.blancJoue:
                for i, j in self.pieces_blanc:
                    if not self.est_geler(i,j):
                        for k, l in self.voisinage_deplacement(self.type_pieces[i, j], i, j):# # if not self.hors_jeu(k,l) and self.est_vide(k,l):
                            if  self.est_vide(k, l):
                                if self.pas == 4:
                                    new_state = self.appliquer_deplacement(self.get_first_state(), ((i,j),(k, l)))
                                    if not np.array_equal(self.first_state, new_state):
                                        L.append(((i,j),(k, l)))
                                else:
                                    L.append(((i, j), (k, l)))

                                if self.pas < 4:
                                    for I, J in self.voisinage(i, j):
                                        if (I, J) in self.pieces_noir and abs(self.type_pieces[i,j]) > abs(self.type_pieces[I,J]):
                                            if self.pas == 3:
                                                new_state = self.appliquer_tirage(self.get_first_state(), ((I,J), ((i,j),(k, l))))
                                                if not np.array_equal(self.first_state, new_state):
                                                    L.append(((I,J), ((i,j),(k, l))))
                                            else:
                                                L.append(((I,J), ((i,j),(k, l))))

                            if self.pas < 4 and (k, l) in self.pieces_noir and abs(self.type_pieces[i,j]) > abs(self.type_pieces[k,l]):
                                for I, J in self.voisinage(k, l):
                                    if self.est_vide(I,J):
                                        if self.pas == 3:
                                            new_state = self.appliquer_poussage(self.get_first_state(), (((i, j), (k, l)), (I, J)))
                                            if not np.array_equal(self.first_state, new_state):
                                                L.append((((i, j), (k, l)), (I, J)))
                                        else:
                                            L.append((((i, j), (k, l)), (I, J)))


            else:
                for i, j in self.pieces_noir:

                    if not self.est_geler(i, j):
                        for k, l in self.voisinage_deplacement(self.type_pieces[i, j], i, j):
                            #print(self.hors_jeu(k, l), self.est_piece_noir(k, l))
                            if self.est_vide(k, l):
                                if self.pas == 4:
                                    new_state = self.appliquer_deplacement(self.get_first_state(), ((i, j), (k, l)))
                                    if not np.array_equal(self.first_state, new_state):
                                        L.append(((i, j), (k, l)))
                                else:
                                    L.append(((i, j), (k, l)))

                                if self.pas < 4:
                                    for I, J in self.voisinage(i, j):
                                        if (I, J) in self.pieces_blanc and abs(self.type_pieces[i, j]) > abs(self.type_pieces[I, J]):
                                            if self.pas == 3:
                                                new_state = self.appliquer_tirage(self.get_first_state(), ((I, J), ((i, j), (k, l))))
                                                if not np.array_equal(self.first_state, new_state):
                                                    L.append(((I, J), ((i, j), (k, l))))
                                            else:
                                                L.append(((I, J), ((i, j), (k, l))))

                            if self.pas < 4 and (k, l) in self.pieces_blanc and abs(self.type_pieces[i, j]) > abs(self.type_pieces[k, l]):
                                for I, J in self.voisinage(k, l):
                                    if self.est_vide(I, J):
                                        if self.pas == 3:
                                            new_state = self.appliquer_poussage(self.get_first_state(), (((i, j), (k, l)), (I, J)))
                                            if not np.array_equal(self.first_state, new_state):
                                                L.append((((i, j), (k, l)), (I, J)))

                                        else:
                                            L.append((((i, j), (k, l)), (I, J)))


        elif self.phase == 'placement':
            if self.blancJoue:
                for i, j in self.positions_libre_blanc:
                            L.append((i,j))
            else:
                for i, j in self.positions_libre_noir:
                            L.append((i, j))

        #print(len(L))
        self.coups_licites = L

    def captures(self):
        L = []

        for i,j in [(2, 2),(2, self.taille-3),(self.taille-3, 2),(self.taille-3, self.taille-3)]:

            if not self.est_vide(i,j) and not self.est_soutenu(i,j):
                p=self.type_pieces[i, j]
                L.append((p, i,j))
                self.retirer(i,j)

                if p == 1:
                    self.lapins_blanc_pris += 1

                if p == -1:
                    self.lapins_noir_pris += 1
        return L


    def est_soutenu(self, i, j):
        p = self.type_pieces[i, j]
        if j > 0 and not self.est_vide(i, j - 1):
            p2 = self.type_pieces[i, j - 1]
            if signe(p) == signe(p2):
                return True

        if j < self.taille - 1 and not self.est_vide(i, j + 1):
            p2 = self.type_pieces[i, j + 1]
            if signe(p) == signe(p2):
                return True

        if i > 0 and not self.est_vide(i - 1, j):
            p2 = self.type_pieces[i - 1, j]
            if signe(p) == signe(p2):
                return True

        if i < self.taille - 1 and not self.est_vide(i + 1, j):
            p2 = self.type_pieces[i + 1, j]
            if signe(p) == signe(p2):
                return True

    def est_geler(self, i,j):

        if self.est_soutenu(i, j):
            return False

        p = self.type_pieces[i, j]

        if j > 0 and not self.est_vide(i, j - 1):
            p2 = self.type_pieces[i, j - 1]
            if signe(p) != signe(p2) and abs(p) < abs(p2):
                return True


        if j < self.taille - 1 and not self.est_vide(i, j + 1):
            p2 = self.type_pieces[i, j + 1]
            if signe(p) != signe(p2) and abs(p) < abs(p2):
                return True

        if i > 0 and not self.est_vide(i - 1, j):
            p2 = self.type_pieces[i - 1, j]
            if signe(p) != signe(p2) and abs(p) < abs(p2):
                return True

        if i < self.taille - 1 and not self.est_vide(i + 1, j):
            p2 = self.type_pieces[i + 1, j]
            if signe(p) != signe(p2) and abs(p) < abs(p2):
                return True

        return False

    def voisinage(self, i, j):
        L = []
        if j > 0:
            L.append((i, j - 1))

        if j < self.taille - 1:
            L.append((i, j + 1))

        if i > 0:
            L.append((i - 1, j))

        if i < self.taille - 1:
            L.append((i + 1, j))
        return L

    def voisinage_deplacement(self, p, i, j):
        L = []
        if abs(p) == 1:
            if p == 1:

                if j > 0:
                    L.append((i, j - 1))

                if j < self.taille - 1:
                    L.append((i, j + 1))

                if i < self.taille - 1:
                    L.append((i + 1, j))

            else:

                if j > 0:
                    L.append((i, j - 1))

                if j < self.taille - 1:
                    L.append((i, j + 1))

                if i > 0:
                    L.append((i - 1, j))

        else:
            return self.voisinage(i, j)


        return L

    def coupsLicites(self):
        return self.coups_licites


    def hors_jeu(self, i, j):
        return not (0 <= i < self.taille and 0 <= j < self.taille)

    def test_fini(self, blanc_joue):

        if self.pas == 1:

            lapin_blanc_arriver = False
            lapin_noir_arriver = False

            for j in range(self.taille):
                if (self.taille - 1, j) in self.pieces_blanc:
                    p = self.type_pieces[self.taille - 1, j]
                    if p == 1:
                        lapin_blanc_arriver = True

                if (0, j) in self.pieces_noir:
                    p = self.type_pieces[0, j]
                    if p == -1:
                        lapin_noir_arriver = True

            if blanc_joue:

                if lapin_blanc_arriver:
                    self.fini = True
                    self.gagnant = 'blanc'

                elif lapin_noir_arriver:
                    self.fini = True
                    self.gagnant = 'noir'

                elif self.lapins_noir_pris == 8:
                    self.fini = True
                    self.gagnant = 'blanc'

                elif self.lapins_blanc_pris == 8:
                    self.fini = True
                    self.gagnant = 'noir'

            if not blanc_joue:

                if lapin_noir_arriver:
                    self.fini = True
                    self.gagnant = 'noir'

                elif lapin_blanc_arriver:
                    self.fini = True
                    self.gagnant = 'blanc'

                elif self.lapins_blanc_pris == 8:
                    self.fini = True
                    self.gagnant = 'noir'

                elif self.lapins_noir_pris == 8:
                    self.fini = True
                    self.gagnant = 'blanc'

            if self.fini:
                self.calcul_score()
                return

            if not self.coupsLicites():
                self.fini = True
                if not blanc_joue:
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
                #self.hash_old_positions_blanc.count(self.hash_plateau_separer()) # self.hash_old_positions_noir.count(self.hash_plateau_separer()) # attention peut etre inverser car il faut utiliser self.blancJoue avec self.hash_old_positions_blanc.count(self.hash_plateau_separer()) et non blanc_joue qui est l ancien joueur et donc le joueur inverse de self.blancJoue qui est le joueur qui va jouer et comme hash_old_positions_blanc stock les états ou blanc va jouer
            elif self.repetition_max is not None and (blanc_joue and self.nb_repet > self.repetition_max or not blanc_joue and self.nb_repet > self.repetition_max):
                self.fini = True

                if blanc_joue:
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'

            if self.fini:
                self.calcul_score()
                return

        if self.borne and self.tour >= self.borne:
            self.fini = True
            self.gagnant = 'nul'
            self.calcul_score()


    def calcul_score(self):

        if self.gagnant == 'nul':
            self.presence_frac = 0
            self.score = 0
            self.score2 = 0
            self.score3 = 0
            self.score4 = 0
            self.score5 = 0
            self.score6 = 0


            v_noir = 0
            v_noir_5 = 0
            for i, j in self.pieces_noir:
                v_noir += abs(self.type_pieces[i, j])
                v_noir_5 += self.repriorisation(abs(self.type_pieces[i, j]))

            v_blanc = 0
            v_blanc_5 = 0
            for i, j in self.pieces_blanc:
                v_blanc += abs(self.type_pieces[i, j])
                v_blanc_5 += self.repriorisation(abs(self.type_pieces[i, j]))


            max_p_noir = max(self.nb_pieces_noir_init, len(self.pieces_noir))
            max_p_blanc = max(self.nb_pieces_blanc_init, len(self.pieces_blanc))

            self.score8 = len(self.pieces_noir) / (max_p_noir + 1) - len(self.pieces_blanc) / (max_p_blanc + 1)

            self.score7 = v_noir_5 / self.valeur_noir_max_5  - v_blanc_5 / self.valeur_blanc_max_5
        else:

            v_noir = 0
            v_noir_5 = 0
            for i, j in self.pieces_noir:
                v_noir += abs(self.type_pieces[i, j])
                v_noir_5 += self.repriorisation(abs(self.type_pieces[i, j]))

            v_blanc = 0
            v_blanc_5 = 0
            for i, j in self.pieces_blanc:
                v_blanc += abs(self.type_pieces[i, j])
                v_blanc_5 += self.repriorisation(abs(self.type_pieces[i, j]))


            if self.gagnant == 'noir':
                self.score2 = max(v_noir-v_blanc, 1)
                self.score4 = max(len(self.pieces_noir) - len(self.pieces_blanc), 1)
                self.score3 = len(self.pieces_noir) + self.nb_pieces_blanc_init - len(self.pieces_blanc)
                self.score = v_noir + self.valeur_blanc_max - v_blanc
                self.presence_frac = (len(self.pieces_noir)+1) / (len(self.pieces_blanc)+1)

                self.score6 = max(v_noir_5 - v_blanc_5, 1)
                self.score5 = v_noir_5 + self.valeur_blanc_max_5 - v_blanc_5

            elif self.gagnant == 'blanc':
                self.score2  = -max(v_blanc - v_noir, 1)
                self.score4 = -max(len(self.pieces_blanc) - len(self.pieces_noir), 1)
                self.score3 = - (len(self.pieces_blanc) + self.nb_pieces_noir_init  - len(self.pieces_noir))
                self.score = - (v_blanc + self.valeur_noir_max - v_noir)
                self.presence_frac = -(len(self.pieces_blanc)+1) / (len(self.pieces_noir)+1)

                self.score5 = - (v_blanc_5 + self.valeur_noir_max_5 - v_noir_5)
                self.score6 = -max(v_blanc_5 - v_noir_5, 1)
            else:
                raise

            self.score7=self.score5
            self.score8 = self.score4
        #print(self.plateau[:, :, 0] + 2*self.plateau[:, :, 1] + 3*self.plateau[:, :, 2] + 4*self.plateau[:, :, 3] + 5*self.plateau[:, :, 4] + +6*self.plateau[:, :, 5])
        #print(self.score, self.score4)

    def symetrie_joueur(self, plateau):
        return np.flip(-1*plateau, 0)

def signe(x):
        return (x > 0) - (x < 0)

if '__main__' == __name__:
    a = Arimaa_vrai()
    print(a.valeur_blanc_max_5)
    print(a.valeur_noir_max_5)