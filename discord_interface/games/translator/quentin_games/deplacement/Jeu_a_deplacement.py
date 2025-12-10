from collections import defaultdict

import numpy as np



class Jeu_a_deplacement():

    def __init__(self, longeur, largeur, liste_pieces, positions_init_pieces, piece_a_prendre, borne=750, codage_ultra_compact=False, phase_init='deplacement', defaite_si_repet=True, repetition_max=0, num_type_hash=0, valeurs_pieces=None, mode_pat=False, regle_50_coups=False, mode_echec = False, true_50_coups=False, defaite_si_pat=False, defaite_si_borne=False, defaite_si_50_coups=False, codage_nb_repet=False, codage_bord=False, pieces_placables_noir_initiales=[], pieces_placables_blanc_initiales=[], codage_retirer=False, codage_historique=0, extra_panels=0):#

        self.extra_panels = extra_panels
        self.codage_retirer = codage_retirer
        self.codage_historique=codage_historique

        self.pieces_placables_blanc_initiales = pieces_placables_blanc_initiales
        self.pieces_placables_noir_initiales = pieces_placables_noir_initiales

        self.codage_bord=codage_bord

        self.codage_nb_repet=codage_nb_repet

        self.dtype = self.get_dtype()

        self.mode_echec = mode_echec

        self.mode_pat=mode_pat
        self.regle_50_coups=regle_50_coups
        self.true_50_coups = true_50_coups

        self.borne = borne
        self.defaite_si_repet = defaite_si_repet
        self.defaite_si_pat = defaite_si_pat
        self.defaite_si_borne =defaite_si_borne
        self.defaite_si_50_coups=defaite_si_50_coups

        self.repetition_max = repetition_max
        self.num_type_hash = num_type_hash

        assert self.repetition_max is None or self.repetition_max >= 0

        if borne:
            self.longueur_max = borne
            self.longueur_moyenne = self.longueur_max/10
        else:
            self.longueur_max = 500
            self.longueur_moyenne = self.longueur_max / 10

        self.nb_pieces = len(liste_pieces)

        self.piece_a_prendre = piece_a_prendre

        self.correspondance_pieces_num = {liste_pieces[i]:i for i in range(len(liste_pieces))}

        self.liste_pieces = liste_pieces
        self.positions_init_pieces = positions_init_pieces

        self.codage_ultra_compact = codage_ultra_compact

        self.longueur = longeur
        self.largeur = largeur

        self.phase_init = phase_init

        self.valeurs_pieces = valeurs_pieces

        self.init()

        self.valeur_noir_max = 0
        self.valeur_blanc_max = 0
        self.nb_pieces_blanc_init = 0
        self.nb_pieces_noir_init = 0
        for (i, j), (piece, blanc) in self.positions_init_pieces:
            if blanc:
                self.nb_pieces_blanc_init+=1
                self.valeur_blanc_max+= self.valeurs_pieces[piece]
            else:
                self.nb_pieces_noir_init += 1
                self.valeur_noir_max+= self.valeurs_pieces[piece]

    def get_dtype(self):
        return 'int8'  # 'float32' # 'float'

    def index_first_extra_panel(self):
        return self.index_codage_historique() + 2*self.codage_historique

    def index(self, nom_piece, blanc_joue):
        assert not self.codage_ultra_compact
        return self.num_piece(self.dic_pieces[nom_piece])


    def positions_placement_blanc_init(self):
        L = []

        return L

    def positions_placement_noir_init(self):
        L = []

        return L

    def init(self):
        self.cause_fin = None

        self.nb_repet = 0


        self.echec = False

        self.tour_sans_prise_et_autres_evenements_speciaux = 0
        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.liste_positions_placement_blanc = set(self.positions_placement_blanc_init())
        self.pieces_placables_blanc = list(self.pieces_placables_blanc_initiales)
        self.liste_positions_placement_noir  = set(self.positions_placement_noir_init())
        self.pieces_placables_noir = list(self.pieces_placables_noir_initiales)

        self.hash_old_positions = []

        self.positions_vides = set([(i,j) for i in range(self.longueur) for j in range(self.largeur)])
        #print(self.longueur, self.largeur)

        self.pieces_blanc_a_promouvoir = []
        self.pieces_noir_a_promouvoir = []

        self.phase = self.phase_init

        self.tour = 0

        self.blancJoue = False

        self.pieces_blanc = set()
        self.pieces_noir = set()

        self.type_pieces = {}

        self.init_plateau()

        self.historique = []

        self.placement_init()

        self.fini = False
        self.gagnant = None

        self.calcul_coups_licites()
        self.actu_couleur_plateau()

        if self.codage_nb_repet:
            self.actu_nb_repet_plateau()


        self.init_nb_pieces_par_type()



    def init_nb_pieces_par_type(self):
        self.nb_pieces_noir = defaultdict(zero)
        self.nb_pieces_blanc = defaultdict(zero)

    def placement_init(self):
        for (i, j), (piece, blanc) in self.positions_init_pieces:
            self.placer(piece, i, j, blanc)

    def actu_nb_repet_plateau(self):
        self.plateau[:, :, -1] = self.nb_repet * np.ones((self.longueur, self.largeur), dtype=self.dtype)

    def get_extra_extra_panels(self):
        return 0


    def init_plateau(self):

        if self.codage_ultra_compact:
            assert not self.codage_retirer
            self.plateau = np.zeros((self.longueur, self.largeur, 2+self.codage_nb_repet+self.codage_bord + 2*self.codage_historique + self.extra_panels + self.get_extra_extra_panels()), dtype=self.dtype)
        else:
            if self.codage_retirer:
                self.plateau = np.zeros((self.longueur, self.largeur, 3*self.nb_pieces + 1 + self.codage_nb_repet + self.codage_bord + 2*self.codage_historique + self.extra_panels + self.get_extra_extra_panels()), dtype=self.dtype)
            else:
                self.plateau = np.zeros((self.longueur, self.largeur, self.nb_pieces + 1 + self.codage_nb_repet + self.codage_bord + 2*self.codage_historique + self.extra_panels + self.get_extra_extra_panels()), dtype=self.dtype)

        if self.codage_bord:
            self.plateau[:, :, -1 - self.codage_nb_repet - 1] = self.get_code_bord()

    def get_code_bord(self):
        p = np.zeros((self.longueur-2, self.largeur-2))

        board_vertical = np.array([[1] for i in range(self.longueur-2)], dtype=self.dtype)
        bord_horizontal = np.array([[1 for i in range(self.largeur)]], dtype=self.dtype)

        return np.vstack((bord_horizontal, np.hstack((board_vertical, p, board_vertical)), bord_horizontal))

    def num_piece(self, piece):
        return self.correspondance_pieces_num[piece]

    def placer(self, piece, i, j, blanc):
        #print(piece, i, j, blanc, self.num_piece(piece))
        if blanc:
            self.pieces_blanc.add((i, j))
            if self.pieces_placables_blanc and self.phase == 'placement':
                self.pieces_placables_blanc.remove(piece)
                self.liste_positions_placement_blanc.remove((i,j))
        else:
            self.pieces_noir.add((i, j))
            if self.pieces_placables_blanc and self.phase == 'placement':
                self.pieces_placables_noir.remove(piece)
                self.liste_positions_placement_noir.remove((i,j))

        self.type_pieces[i, j] = piece

        self.positions_vides.remove((i, j))

        if self.codage_ultra_compact:
            if blanc:
                self.plateau[i, j, 0] = self.num_piece(piece)+1
            else:
                self.plateau[i, j, 0] = -(self.num_piece(piece)+1)
        else:
            if blanc:
                self.plateau[i, j, self.num_piece(piece)] = 1
            else:
                self.plateau[i, j, self.num_piece(piece)] = -1

        if isinstance(piece, self.piece_a_prendre.__class__):
            if blanc:
                self.piece_a_prendre_blanc = i, j
            else:
                self.piece_a_prendre_noir = i, j


        if self.codage_retirer and self.phase == 'deplacement':
            self.retirer_codage_piece_retirer(piece, blanc)

        return self.phase_apres_un_placement(), not self.blancJoue

    def phase_apres_un_placement(self):
        return 'deplacement'

    def retirer(self, i, j):

        blanc = None
        if (i,j) in self.pieces_noir:
            self.pieces_noir.remove((i, j))
            blanc = False
        if (i,j) in self.pieces_blanc:
            self.pieces_blanc.remove((i, j))
            assert blanc is None
            blanc = True


        self.positions_vides.add((i,j))

        piece = self.type_pieces[i, j]


        if self.phase == 'placement':
            if blanc:
                self.pieces_placables_blanc.append(piece)
                self.liste_positions_placement_blanc.add((i, j))
            else:
                self.pieces_placables_noir.append(piece)
                self.liste_positions_placement_noir.add((i, j))

        if self.codage_ultra_compact:
            self.plateau[i,j,0] = 0
        else:
            self.plateau[i, j, self.num_piece(self.type_pieces[i,j])] = 0

        if self.codage_retirer and self.phase == 'deplacement':
            self.ajouter_codage_piece_retirer(self.type_pieces[i, j], blanc)

        del self.type_pieces[i, j]

    """
    # code que si chaque piece est unique
    def ajouter_codage_piece_retirer(self, piece, blanc):
        if blanc:
            self.plateau[:, :, self.num_piece(piece)+self.nb_pieces] = 1
        else:
            self.plateau[:, :, self.num_piece(piece)+self.nb_pieces] = -1

    def retirer_codage_piece_retirer(self, piece, blanc):
        self.plateau[:, :, self.num_piece(piece) + self.nb_pieces] = 0"""

    def index_codage_piece_retirer(self):
        return 2 * self.nb_pieces

    def ajouter_codage_piece_retirer(self, piece, blanc):
        self.plateau[:, :, 2 * self.num_piece(piece) + blanc + self.index_codage_piece_retirer()] += 1

        if blanc:
            self.nb_pieces_blanc[piece] -= 1
        else:
            self.nb_pieces_noir[piece] -= 1


    def retirer_codage_piece_retirer(self, piece, blanc):
        self.plateau[:, :, 2 * self.num_piece(piece) + blanc + self.index_codage_piece_retirer()] -= 1

        if blanc:
            self.nb_pieces_blanc[piece] += 1
        else:
            self.nb_pieces_noir[piece] += 1



    def deplacer(self, i, j, k, l):
        #assert self.plateau[i,j,self.num_piece(self.type_pieces[i, j])]==1 and (i,j) in self.pieces_blanc and self.blancJoue or self.plateau[i,j,self.num_piece(self.type_pieces[i, j])]==-1 and (i,j) in self.pieces_noir and not self.blancJoue

        if (k,l) in self.type_pieces and self.type_pieces[i, j].__class__.__name__ != self.type_pieces[k, l].__class__.__name__:
            """assert self.plateau[k, l, self.num_piece(self.type_pieces[k, l])] == 1 and (k, l) in self.pieces_blanc or self.plateau[k, l, self.num_piece(self.type_pieces[k, l])] == -1 and (k, l) in self.pieces_noir
            
            if self.plateau[i,j,self.num_piece(self.type_pieces[i, j])] == self.plateau[k,l,self.num_piece(self.type_pieces[k, l])]:
                print(self.type_pieces[i, j].__class__.__name__, self.type_pieces[k, l].__class__.__name__)
                print((i,j),(k,l))
                assert self.plateau[i, j, self.num_piece(self.type_pieces[i, j])] != self.plateau[k, l, self.num_piece(self.type_pieces[k, l])]"""
        piece = self.type_pieces[i, j]
        #print('+')
        #assert ((k, l) in self.type_pieces) == ((k, l) in self.pieces_blanc or (k, l) in self.pieces_noir)
        #assert ((k,l) in self.type_pieces) == ((k,l) not in self.positions_vides)
        """if  ((k,l) in self.type_pieces) != ((k,l) not in self.positions_vides):
            print((k,l))
            print(self.type_pieces)
            print(self.positions_vides)
            print((k,l) in self.type_pieces)
            print((k,l) not in self.positions_vides)
        #print()
        
        if ((k,l) in self.type_pieces) != ((k,l) in self.pieces_blanc or (k,l) in self.pieces_noir):
            print(k,l)
            print(self.type_pieces)
            print(self.pieces_blanc)
            print(self.pieces_noir)
            print((k,l) in self.type_pieces)
            print((k,l) in self.pieces_blanc)
            print((k,l) in self.pieces_noir)
            raise"""

        if (k, l) in self.type_pieces:
            piece_retirer = self.type_pieces[k, l]
            self.retirer(k, l)
        else:
            piece_retirer = None
        self.retirer(i, j)
        self.placer(piece, k, l, self.blancJoue)

        self.type_pieces[k, l] = piece

        return (piece_retirer, *self.new_phase_couleur_post_deplacement(piece, piece_retirer, (i,j), (k, l)))

    def promouvoir(self, i, j, new_piece):

        piece_retirer = self.type_pieces[i, j]

        #assert self.plateau[i,j,self.num_piece(piece_retirer)] == 1 and self.blancJoue or self.plateau[i,j,self.num_piece(piece_retirer)] == -1 and not self.blancJoue
        self.retirer(i, j)
        self.placer(new_piece, i, j, self.blancJoue)
        """self.type_pieces[i, j] = new_piece

        if self.codage_ultra_compact:
            if self.blancJoue:
                self.plateau[i,j, 0] = self.num_piece(new_piece)
            else:
                self.plateau[i, j, 0] = -self.num_piece(new_piece)
        else:
            if self.blancJoue:
                self.plateau[i, j, self.num_piece(new_piece)] = 1
            else:
                self.plateau[i, j, self.num_piece(new_piece)] = -1"""

        return piece_retirer, 'deplacement', not self.blancJoue

    def new_phase_couleur_post_deplacement(self, piece, piece_retirer, pos_init, pos_fin):
        return 'deplacement', not self.blancJoue

    def actu_couleur_plateau(self):

        if self.phase == 'deplacement':
            k = 1
        elif self.phase == 'placement':
            k = 2
        elif self.phase == 'promotion':
            k = 3

        if self.blancJoue:
            self.plateau[:, :, -1-self.codage_nb_repet] = k * np.ones((self.longueur, self.largeur), dtype=self.dtype)
        else:
            self.plateau[:, :, -1-self.codage_nb_repet] = - k * np.ones((self.longueur, self.largeur), dtype=self.dtype)

        if self.codage_historique:
            self.actu_codage_historique()

    def raz(self):
        self.init()

    def hash_plateau(self):
        if self.codage_nb_repet:
            if self.num_type_hash == 0:
                return self.plateau[:,:,:-1].tobytes()
            elif self.num_type_hash == 1: # risque de collision mais peut-être un gain de perf d'environ 2%
                return hash(self.plateau[:,:,:-1].tobytes())
        else:

            if self.num_type_hash == 0:
                return self.plateau.tobytes()
            elif self.num_type_hash == 1: # risque de collision mais peut-être un gain de perf d'environ 2%
                return hash(self.plateau.tobytes())
            """elif self.num_type_hash == 2:# risque de collision et 100 fois plus lent ...
                return hash(str(self.plateau))"""

    def regles_speciales_jouer(self, a, b):
        """"""

    def jouer(self, a, b):

        #print('?')
        if not (a,b) in self.coupsLicites():
            print('cp : ',a,b)
            print('cps :',self.coupsLicites())
            print('H : ',[(a2,b2) for a2,b2, *_ in self.historique])
        assert (a,b) in self.coupsLicites()

        mobilite= self.actualisation_jeu_debut()

        prise = False
        if self.phase == 'deplacement':
            piece_retirer, next_phase, next_blancJoue = self.deplacer(*(*a, *b))
            prise = piece_retirer is not None

        elif self.phase == 'placement':

            next_phase, next_blancJoue = self.placer(a, *b, self.blancJoue)
            piece_retirer = None

        elif self.phase == 'promotion':

            piece_retirer, next_phase, next_blancJoue = self.promouvoir(*a, b)

        else:
            raise

        self.actualisation_jeu_fin(a, b, piece_retirer, prise, next_phase, next_blancJoue, mobilite)

    def actualisation_jeu_debut(self):
        self.tour += 1
        #print(self.tour, self.borne)

        mobilite = self.calcul_mobilite()

        if self.repetition_max is not None:
            self.hash_old_positions.append(self.hash_plateau())

        return mobilite


    def actualisation_jeu_fin(self, a, b, piece_retirer, prise, next_phase, next_blancJoue, mobilite):


        self.historique.append(
                (a, b, self.blancJoue, piece_retirer, self.phase, list(self.coups_licites), self.pieces_noir_a_promouvoir, self.pieces_blanc_a_promouvoir, self.tour_sans_prise_et_autres_evenements_speciaux, self.echec, self.nb_repet))

        if not next_phase == 'promotion':
            self.tour_sans_prise_et_autres_evenements_speciaux += 1
        if prise or self.phase == 'promotion':# and self.historique[-2][3] is not None:
            self.tour_sans_prise_et_autres_evenements_speciaux = 0

        self.pieces_noir_a_promouvoir = []
        self.pieces_blanc_a_promouvoir = []

        if next_phase == 'promotion':
            if self.blancJoue:
                self.pieces_blanc_a_promouvoir.append(b)
            else:
                self.pieces_noir_a_promouvoir.append(b)

        self.phase = next_phase

        self.blancJoue = next_blancJoue

        if self.mode_echec:
            self.echec = self.test_echec()
            #print(self.echec, self.blancJoue)


        self.regles_speciales_jouer(a, b)

        self.calcul_coups_licites()

        self.actu_couleur_plateau()

        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())
        if self.codage_nb_repet:
            self.actu_nb_repet_plateau()

        self.test_fini(piece_retirer)

        if self.fini:
            self.mise_a_jour_mobilite(mobilite)

    def calcul_mobilite(self):
        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1
        #print('<', self.nb_coup_noir, self.nb_coup_blanc, len(self.historique), self.fini)
        return mobilite


    def mise_a_jour_mobilite(self, mobilite):
        #print('fini :', self.fini, len(self.historique), self.cause_fin )
        self.mobilite_cumule_diff_pour_nul = self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc
        self.mobilite_cumule_frac_pour_nul = (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

        if self.gagnant == 'noir':
            self.mobilite = mobilite
            self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
            self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
            self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

            self.mobilite_cumule_diff_0dist = self.mobilite_cumule_diff

        elif self.gagnant == 'blanc':
            self.mobilite = -mobilite
            self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
            self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
            self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)

            self.mobilite_cumule_diff_0dist = self.mobilite_cumule_diff
        else:
            self.mobilite = 0
            self.mobilite_cumule = 0
            self.mobilite_cumule_diff = 0
            self.mobilite_cumule_frac = 0

            self.mobilite_cumule_diff_0dist =  self.nb_coup_blanc / self.mobilite_cumule_blanc - self.nb_coup_noir / self.mobilite_cumule_noir

    def regles_speciales_undo(self, a, b):
        """"""


    def undo(self, ):

        self.tour -= 1

        if self.hash_old_positions:
            self.hash_old_positions.pop()


        a, b, self.blancJoue, piece_retirer, self.phase, self.coups_licites, self.pieces_noir_a_promouvoir, self.pieces_blanc_a_promouvoir, self.tour_sans_prise_et_autres_evenements_speciaux, self.echec, self.nb_repet = self.historique.pop()

        self.regles_speciales_undo(a, b)


        if self.phase == 'deplacement':

            self.deplacer(*b, *a)
            if piece_retirer:
                self.placer(piece_retirer, *b, not self.blancJoue)

        elif self.phase == 'placement':

            self.retirer(*b)

        elif self.phase == 'promotion':

            self.promouvoir(*a, piece_retirer)

        self.fini = False
        self.gagnant = None
        self.cause_fin = None
        self.actu_couleur_plateau()

        self.actu_mobilite_undo()

        if self.codage_nb_repet:
            self.actu_nb_repet_plateau()

    def actu_mobilite_undo(self):
        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1
        #print('>', self.nb_coup_noir, self.nb_coup_blanc, len(self.historique), self.fini)

    def est_vide(self, i, j):
        return (i, j) in self.positions_vides

    def est_piece_blanc(self, i, j):
        return (i,j) in self.pieces_blanc

    def est_piece_noir(self, i, j):
        return (i, j) in self.pieces_noir

    def test_echec(self):
        raise NotImplemented
        if self.blancJoue:
            I, J = self.piece_a_prendre_blanc

            for i, j in self.pieces_noir:
                for k, l in self.type_pieces[i, j].deplacements(i, j, self):
                    if (I, J) == (k, l):
                        self.echec = True
                        return
        else:
            I, J = self.piece_a_prendre_noir

            for i, j in self.pieces_blanc:
                for k, l in self.type_pieces[i, j].deplacements(i, j, self):
                    if (I, J) == (k, l):
                        self.echec = True
                        return


    def calcul_coups_licites(self):
        L = []


        if self.phase == 'deplacement':

            if self.mode_echec:

                if self.blancJoue:
                    for i, j in self.pieces_blanc:
                        for k, l in self.type_pieces[i, j].deplacements(i, j, self):# # if not self.hors_jeu(k,l) and self.est_vide(k,l):
                            if not self.hors_jeu(k, l) and not self.est_piece_blanc(k, l) and not self.clouer(i, j, k, l) and (not self.echec or self.solve_echec(i, j, k, l)):
                                L.append(((i,j),(k, l)))

                else:
                    for i, j in self.pieces_noir:
                        for k, l in self.type_pieces[i, j].deplacements(i, j, self):
                            if not self.hors_jeu(k, l) and not self.est_piece_noir(k, l) and not self.clouer(i, j, k, l) and (not self.echec or self.solve_echec(i, j, k, l)):
                                L.append(((i,j),(k, l)))

            else:

                if self.blancJoue:
                    for i, j in self.pieces_blanc:
                        for k, l in self.type_pieces[i, j].deplacements(i, j, self):# # if not self.hors_jeu(k,l) and self.est_vide(k,l):
                            if not self.hors_jeu(k, l) and not self.est_piece_blanc(k, l):
                                L.append(((i,j),(k, l)))

                else:
                    for i, j in self.pieces_noir:
                        for k, l in self.type_pieces[i, j].deplacements(i, j, self):
                            if not self.hors_jeu(k, l) and not self.est_piece_noir(k, l):
                                L.append(((i,j),(k, l)))


        elif self.phase == 'placement':
            if self.blancJoue:
                for i, j in self.liste_positions_placement_blanc:
                        for p in set(self.pieces_placables_blanc):
                            L.append((p,(i,j)))
            else:
                for i, j in self.liste_positions_placement_noir:
                        for p in set(self.pieces_placables_noir):
                            L.append((p, (i, j)))



        elif self.phase == 'promotion':
            #print('promo',self.pieces_blanc_a_promouvoir, self.pieces_noir_a_promouvoir)
            if self.blancJoue:
                for i, j in self.pieces_blanc_a_promouvoir:
                    for p in self.type_pieces[i,j].possibles_promotions():
                        L.append(((i,j),p))
            else:
                for i, j in self.pieces_noir_a_promouvoir:
                    for p in self.type_pieces[i,j].possibles_promotions():
                        L.append(((i,j),p))

        self.coups_licites = L

    def coupsLicites(self):
        #print(self.coups_licites)
        return self.coups_licites


    def hors_jeu(self, i, j):
        return not (0 <= i < self.longueur and 0 <= j < self.largeur)


    def use_fin_borne_special(self):
        return False

    def fin_borne_special(self):
        raise NotImplementedError()


    def test_fini(self, prise):
        #print(self.hash_old_positions)
        #print(self.repetition_max, self.hash_old_positions.count(self.hash_plateau()), self.hash_plateau())
        #print(self.hash_old_positions.count(self.hash_plateau()), self.repetition_max)
        if self.borne and self.tour >= self.borne:
            self.fini = True
            self.cause_fin = 'borne'
            #print('borne  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if self.defaite_si_borne:

                if self.use_fin_borne_special():
                    self.fin_borne_special()

                else:
                    if not self.blancJoue:  # on a changer de jouer (joueur actuel joueur du tour d apres)
                        self.gagnant = 'noir'
                    else:
                        self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()


        elif self.repetition_max is not None and self.nb_repet > self.repetition_max:
            self.fini = True
            self.cause_fin = 'répétition'
            #print('répétition !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #print(self.hash_old_positions.count(self.hash_plateau()), self.repetition_max)

            if self.defaite_si_repet:
                if not self.blancJoue: # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if not self.coupsLicites() and self.mode_pat and not self.echec:
            #print('pat !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.cause_fin = 'pat'
            self.fini = True
            if self.defaite_si_pat:
                if self.blancJoue:  # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if not self.coupsLicites() and self.mode_echec and self.echec:
            #print('echec et mat !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.cause_fin = 'echec et mat'
            self.fini = True
            if self.blancJoue:
                self.gagnant = 'noir'
            else:
                self.gagnant = 'blanc'
            self.calcul_score()

        if self.regle_50_coups and (self.tour_sans_prise_et_autres_evenements_speciaux >= 50 and not self.true_50_coups or self.tour_sans_prise_et_autres_evenements_speciaux >= 100 and self.true_50_coups):
            #print('50c !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.cause_fin = '50c'
            self.fini = True
            if self.defaite_si_50_coups:
                if not self.blancJoue:  # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if prise == self.piece_a_prendre or (not self.coupsLicites() and not self.mode_pat):
            self.fini = True
            self.cause_fin = 'prise royal'
            if self.blancJoue:
                self.gagnant = 'noir'
            else:
                self.gagnant = 'blanc'
            self.calcul_score()



    def test_fini_debug(self, prise):
        #print(self.hash_old_positions)
        #print(self.repetition_max, self.hash_old_positions.count(self.hash_plateau()), self.hash_plateau())
        if self.borne and self.tour >= self.borne:
            self.fini = True
            print('borne')
            if self.defaite_si_borne:
                if not self.blancJoue: # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()
        elif self.repetition_max is not None and self.hash_old_positions.count(self.hash_plateau()) > self.repetition_max:
            self.fini = True
            print('rep')
            #print(self.hash_old_positions.count(self.hash_plateau()), self.repetition_max)

            if self.defaite_si_repet:
                if not self.blancJoue: # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if not self.coupsLicites() and self.mode_pat and not self.echec:
            print('pat')
            self.fini = True
            if self.defaite_si_pat:
                if not self.blancJoue: # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if not self.coupsLicites() and self.mode_echec and self.echec:
            print('echec et mat')
            self.fini = True
            if self.blancJoue:
                self.gagnant = 'noir'
            else:
                self.gagnant = 'blanc'
            self.calcul_score()

        if self.regle_50_coups and (self.tour_sans_prise_et_autres_evenements_speciaux >= 50 and not self.true_50_coups or self.tour_sans_prise_et_autres_evenements_speciaux >= 100 and self.true_50_coups):
            print('50c')
            self.fini = True
            if self.defaite_si_50_coups:
                if not self.blancJoue: # on a changer de jouer (joueur actuel joueur du tour d apres)
                    self.gagnant = 'noir'
                else:
                    self.gagnant = 'blanc'
            else:
                self.gagnant = 'nul'
            self.calcul_score()

        if prise == self.piece_a_prendre or (not self.coupsLicites() and not self.mode_pat):
            self.fini = True
            if self.blancJoue:
                self.gagnant = 'noir'
            else:
                self.gagnant = 'blanc'
            self.calcul_score()


    def calcul_score(self):

        v_noir = 0
        for i, j in self.pieces_noir:
            v_noir += self.valeurs_pieces[self.type_pieces[i, j]]

        v_blanc = 0
        for i, j in self.pieces_blanc:
            v_blanc += self.valeurs_pieces[self.type_pieces[i, j]]

        if self.gagnant == 'nul':
            self.presence_frac = 0
            self.score = 0
            self.score2 = 0
            self.score3 = 0
            self.score4 = 0

            max_p_noir = max(self.nb_pieces_noir_init, len(self.pieces_noir))
            max_p_blanc = max(self.nb_pieces_blanc_init, len(self.pieces_blanc))

            self.score5 = len(self.pieces_noir) / (max_p_noir + 1) - len(self.pieces_blanc) / (max_p_blanc + 1)

            v_noir = 0
            for i, j in self.pieces_noir:
                v_noir += self.valeurs_pieces[self.type_pieces[i, j]]

            v_blanc = 0
            for i, j in self.pieces_blanc:
                v_blanc += self.valeurs_pieces[self.type_pieces[i, j]]

            self.score6 = v_noir / (self.valeur_noir_max + 1) - v_blanc / (self.valeur_blanc_max+1)
            self.score7 = 0
            self.score8 = v_noir / (3*self.valeur_noir_max + 1) - v_blanc / (3*self.valeur_blanc_max+1)
        else:

            if self.gagnant == 'noir':
                self.score2 = max(v_noir-v_blanc, 1)
                self.score4 = max(len(self.pieces_noir) - len(self.pieces_blanc), 1)
                self.score3 = len(self.pieces_noir) + self.nb_pieces_blanc_init - len(self.pieces_blanc)
                self.score = max(v_noir + self.valeur_blanc_max - v_blanc, 1)
                self.presence_frac = (len(self.pieces_noir)+1) / (len(self.pieces_blanc)+1)
                self.score7 = max(v_noir + 3*self.valeur_blanc_max - v_blanc, 1)

            elif self.gagnant == 'blanc':
                self.score2  = -max(v_blanc - v_noir, 1)
                self.score4 = -max(len(self.pieces_blanc) - len(self.pieces_noir), 1)
                self.score3 = - (len(self.pieces_blanc) + self.nb_pieces_noir_init  - len(self.pieces_noir))
                self.score = min(- (v_blanc + self.valeur_noir_max - v_noir), -1)
                self.presence_frac = -(len(self.pieces_blanc)+1) / (len(self.pieces_noir)+1)
                self.score7 = min(- (v_blanc + 3*self.valeur_noir_max - v_noir), -1)

            else:
                raise

            self.score5 = self.score4
            self.score6 = self.score
            self.score8 = self.score7

        if self.fini:
            self.score_pour_nul = v_noir - v_blanc

    def symetrie_joueur(self, plateau):
        return np.flip(-1*plateau, 0)

    def index_codage_historique(self):
        if self.codage_retirer:
            return 3 * self.nb_pieces
        else:
            return 1 * self.nb_pieces

    def actu_codage_historique(self):

        idx = self.index_codage_historique()

        self.plateau[:,:,idx:idx+2*self.codage_historique] = 0

        for c in range(self.codage_historique):
            if len(self.historique)-(c+1) >=0:
                a,b, _, __, phase, *___ = self.historique[-(c+1)]
                if phase == 'deplacement':
                    self.plateau[a+(idx + 2 * c,  )] = 1
                    self.plateau[b+(idx + 2 * c+1,)] = 1

def zero():
    return 0