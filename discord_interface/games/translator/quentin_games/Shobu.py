from discord_interface.games.translator.quentin_games.JeuSimpleAbstrait import JeuSimpleAbstrait


class Shobu(JeuSimpleAbstrait):

    """ VARIANTES : """


    def __init__(self, taille=4, repetition_max=0, regle_50c=False, borne_max=400, coder_la_repetition=True):

        self.repetition_max = repetition_max
        self.regle_50c = regle_50c
        self.borne_max=borne_max
        self.coder_la_repetition = coder_la_repetition

        self.calcul_codage_repetitions()
        self.calcul_codage_regle_50c()

        self.positions_init_noir = [(3, i) for i in range(4)]
        self.positions_init_blanc = [(0, i) for i in range(4)]

        self.directions = [(i,j) for i in range(-1,2) for j in range(-1,2) if i != 0 or j != 0]

        self.zones_blanc = [0, 1]
        self.zones_noir = [2, 3]

        super().__init__(taille=taille)

        self.longueur_max = 350
        self.longueur_moyenne = 100


    def get_type_board(self):
        return 'int8' #'int8'  # 'float32' # 'float'



    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 4] = 1
        else:
            self.plateau[:, :, 4] = -1


    def zones_opposes(self, z):
        if z % 2 == 0:
            return [1, 3]
        else:
            return [0, 2]

    def in_board(self, i, j):
        return 0 <= i < self.taille and 0 <= j < self.taille#((1 <= i <= self.taille and 1 <= j < self.taille) or (1 <= i <= self.taille and 1 <= j < self.taille and self.taille+2 <= i <= 2*self.taille+1) or (self.taille+2 <= i <= 2*self.taille+1 and 1 <= j < self.taille) or (self.taille+2 <= i <= 2*self.taille+1 and self.taille+2 <= i <= 2*self.taille+1))

    def opposer(self, p, centre):
        i, j = p
        o1, o2 = centre

        pb = i + 2*(o1 - i), j + 2*(o2 - j)

        return pb




    def deplacer(self, z, i, j, k, l, blanc):
        #print('-',z,(i, j),'->',(k, l))
        if self.in_board(i,j):
            self.retirer(z, i,j)
        if self.in_board(k,l):
            self.ajouter(z, k,l,blanc)

    def pousser(self, z, i, j, k, l, blanc):
        self.retirer(z, i,j)
        self.retirer(z, k,l)

        self.ajouter(z, k,l,blanc)
        o = self.opposer((i,j),(k,l))
        if self.in_board(*o):
            self.ajouter(z, *o, not blanc)



    def ajouter(self, z, i, j, blanc):
        if blanc:
            self.plateau[i,j,z] = 1
            self.pieces_blanc[z].add((i, j))
        else:
            self.plateau[i, j, z] = -1
            self.pieces_noir[z].add((i, j))


    def retirer(self, z, i, j):
        v = self.plateau[i,j,z]
        self.plateau[i,j,z]=0

        if v == 1:
            self.pieces_blanc[z].remove((i, j))
        else:
            if not v == -1:
                for z in range(4):
                    print(self.plateau[:,:,z])
            assert v == -1
            self.pieces_noir[z].remove((i, j))


    def init_plateau(self):
        for z in range(4):
            for i,j in self.positions_init_noir:
                self.ajouter(z, i,j, blanc=False)

            for i,j in self.positions_init_blanc:
                self.ajouter(z, i,j, blanc=True)

    def use_repetitions(self):
        if self.repetition_max is not None:
            return  1
        else:
            return 0

    def codage_repetitions(self):
        return self.code_repetition

    def codage_regle_50c(self):
        return self.code_regle_50c

    def calcul_codage_repetitions(self):
        if self.repetition_max is not None and self.coder_la_repetition:
            self.code_repetition =  1
        else:
            self.code_repetition = 0

    def calcul_codage_regle_50c(self):
        if self.regle_50c:
            self.code_regle_50c = 1
        else:
            self.code_regle_50c = 0

    def get_panel(self):
        return 4 + 1 + self.codage_regle_50c() + self.codage_repetitions()


    def init(self):

        self.nb_repet = 0
        self.tour=0
        self.tour_sans_prise = 0

        self.panels = self.get_panel()

        self.pieces_blanc = {}
        self.pieces_noir = {}

        for p in range(4):
            self.pieces_blanc[p] = set()
            self.pieces_noir[p] = set()

        super().init()


        self.hash_old_positions = []
        self.actu_codage_extras()






    def actu_codage_extras(self):
        if self.codage_regle_50c():
            self.plateau[:, :, -self.codage_regle_50c()-self.codage_repetitions()] = self.tour_sans_prise
        if self.codage_repetitions():
            self.plateau[:, :, -1] = self.nb_repet

    def application_coup_elementaire(self, coup, agressif):
        z, (i, j), *b = coup
        # print(b)

        if len(b) == 1:
            k, l = b[0]
            self.deplacer(z, i, j, k, l, self.blancJoue)

            if agressif:
                self.tour_sans_prise += 1
        else:
            (k, l), (m, n), (p, q) = b

            self.deplacer(z, m, n, p, q, not self.blancJoue)
            self.deplacer(z, i, j, k, l, self.blancJoue)

            if agressif:
                if not self.in_board(p, q):
                    self.tour_sans_prise = 0
                else:
                    self.tour_sans_prise += 1

    def jouer(self, phase1,phase2):

        if self.use_repetitions():
            self.hash_old_positions.append(self.hash_plateau())

        self.historique.append(((phase1,phase2), self.blancJoue, list(self.coups_licites), self.tour, self.tour_sans_prise, self.nb_repet))

        self.tour+=1

        for coup, agressif in [(phase1, False), (phase2, True)]:
           self.application_coup_elementaire(coup, agressif)


        self.blancJoue = not self.blancJoue
        self.actu_couleur_plateau()
        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())
        self.actu_codage_extras()

        self.test_fini_A()

        self.calcul_coups_licites()

        self.test_fini_B()

        if self.fini:
            self.calcul_score()


    def unapplication_coup_elementaire(self, coup):
        #print(coup)
        z, (i, j), *b = coup

        if len(b) == 1:
            k, l = b[0]
            self.deplacer(z, k, l, i, j, self.blancJoue)
        else:
            (k, l), (m, n), (p, q) = b

            self.deplacer(z, k, l, i, j, self.blancJoue)
            self.deplacer(z, p, q, m, n, not self.blancJoue)

    def undo(self, ):

        if self.hash_old_positions:
            self.hash_old_positions.pop()

        (phase1,phase2), self.blancJoue, self.coups_licites, self.tour, self.tour_sans_prise, self.nb_repet = self.historique.pop()

        for coup in [phase1, phase2]:
            self.unapplication_coup_elementaire(coup)

        self.actu_couleur_plateau()
        self.actu_codage_extras()

        self.fini = False

        self.gagnant = None




    def est_vide(self, z, p):
        return p not in self.pieces_noir[z] and p not in self.pieces_blanc[z] and self.in_board(*p)

    def calcul_coups_licites(self):
        if self.fini:
            return []

        if self.blancJoue:
            pieces_soi = self.pieces_blanc
            pieces_adv = self.pieces_noir
            ses_zones = self.zones_blanc
        else:
            pieces_adv = self.pieces_blanc
            pieces_soi = self.pieces_noir
            ses_zones = self.zones_noir

        L = []
        """for z in range(4):
            print(self.plateau[:, :, z])
            print(self.pieces_noir[z], self.pieces_blanc[z])
        print('<<<<')"""
        for z in ses_zones:
            for p in pieces_soi[z]:
                for d in self.directions:
                    p1, _ = self.position_distance_un(p, d)
                    if self.est_vide(z, p1):

                        for zb in self.zones_opposes(z):
                            for pb in pieces_soi[zb]:
                                #ok = False
                                p1b, p1b_post = self.position_distance_un(pb, d)
                                #print(z, p, p1, zb, pb, p1b, p1b not in self.pieces_noir[zb], p1b not in self.pieces_blanc[zb], self.in_board(*p1b))
                                if self.est_vide(zb, p1b):
                                    L.append((((z, p,p1), (zb, pb, p1b))))
                                    #ok = True
                                if p1b in pieces_adv[zb] and (self.est_vide(zb, p1b_post) or not self.in_board(*p1b_post)):
                                    L.append(((z, p,p1), (zb, pb, p1b, p1b, p1b_post)))
                                """else:
                                    if not ok:
                                        print(((z, p,p1), (zb, pb, p1b, p1b, p1b_post)), p1b_post not in self.pieces_noir[zb] and p1b_post not in self.pieces_blanc[zb] and self.in_board(*p), not self.in_board(*p1b_post))
"""

                for d in self.directions:
                    p2, p2_inter, _ = self.positions_distance_deux(p, d)
                    #print(p2, p2 not in self.pieces_noir[z] , p2 not in self.pieces_blanc[z] , self.in_board(*p), p2_inter, z, self.pieces_noir[z], self.pieces_blanc[z], "/", self.est_vide(z, p2_inter))
                    if self.est_vide(z, p2) and self.est_vide(z, p2_inter):

                        for zb in self.zones_opposes(z):
                            #print('>>>>>>>',pieces_soi[zb])
                            for pb in pieces_soi[zb]:
                                p2b, p2_inter_b, p2b_post = self.positions_distance_deux(pb, d)

                                if self.est_vide(zb, p2b) and self.est_vide(zb, p2_inter_b):
                                    L.append(((z, p, p2), (zb, pb, p2b)))
                                if self.est_vide(zb, p2b) and (self.est_vide(zb, p2b_post) or not self.in_board(*p2b_post)) and p2_inter_b in pieces_adv[zb]:
                                    L.append(((z, p, p2), (zb, pb, p2b, p2_inter_b, p2b_post)))
                                if self.est_vide(zb, p2_inter_b) and (self.est_vide(zb, p2b_post) or not self.in_board(*p2b_post)) and p2b in pieces_adv[zb]:
                                    L.append(((z, p, p2), (zb, pb, p2b, p2b, p2b_post)))
                                #print(((z, p, p2), (zb, pb, p2b, p2b, p2b_post)), p2b_post not in self.pieces_noir[zb] and p2b_post not in self.pieces_blanc[zb], not self.in_board(*p2b_post), p2b in pieces_adv[zb])

        self.coups_licites = L

    def position_distance_un(self, p, d):
        i, j = p
        di, dj = d

        return (i+di, j+dj), (i+2*di, j+2*dj),

    def positions_distance_deux(self, p, d):
        i, j = p
        di, dj = d

        return (i + 2 * di, j + 2 * dj), (i + di, j + dj), (i + 3 * di, j + 3 * dj),

    def calcul_score(self):

        if self.gagnant == 'nul':
            self.score = 0
            self.score2 = 0
        else:
            self.score2= 0

            cn = [len(self.pieces_noir[0]), len(self.pieces_noir[1]), len(self.pieces_noir[2]), len(self.pieces_noir[3])]
            cb = [len(self.pieces_blanc[0]), len(self.pieces_blanc[1]), len(self.pieces_blanc[2]), len(self.pieces_blanc[3])]

            cn.sort()
            cb.sort()

            cn.reverse()
            cb.reverse()

            for i, c in enumerate(cn):
                self.score2 += c * 5**i

            for i, c in enumerate(cb):
                self.score2 -= c * 5**i

            if self.gagnant == 'noir':

                self.score = min(len(self.pieces_noir[0]), len(self.pieces_noir[1]), len(self.pieces_noir[2]), len(self.pieces_noir[3]))
            else:
                assert self.gagnant == 'blanc'
                self.score = -min(len(self.pieces_blanc[0]), len(self.pieces_blanc[1]), len(self.pieces_blanc[2]), len(self.pieces_blanc[3]))

    def test_fini_A(self):

        if self.blancJoue:
            if not self.pieces_blanc[0] or not self.pieces_blanc[1] or not self.pieces_blanc[2] or not self.pieces_blanc[3]:
                self.fini = True
                self.gagnant = 'noir'
        else:
            if not self.pieces_noir[0] or not self.pieces_noir[1] or not self.pieces_noir[2] or not self.pieces_noir[3]:
                self.fini = True
                self.gagnant = 'blanc'

    def test_fini_B(self):

        if not self.fini:
            if not self.coups_licites:

                self.fini = True
                if self.blancJoue:
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'

        if not self.fini:
            if self.regle_50c and (self.tour_sans_prise >= 100) or self.tour >= self.borne_max or self.use_repetitions() and self.nb_repet > self.repetition_max:
                self.fini = True
                #print('+>',self.regle_50c and (self.tour_sans_prise >= 100), self.tour >= self.borne_max, self.use_repetitions() and self.nb_repet > self.repetition_max)
                self.gagnant = 'nul'




    def hash_plateau(self):
        return self.plateau[:,:,:5].tobytes()