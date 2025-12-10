import sys
from math import inf

import numpy as np


taille_par_defaut = 10

class Draughts():
    def __init__(self, taille=taille_par_defaut, repetition_max=2, codage_bord=False, codage_50c=False, codage_repet=False):# 1 veut dire 3 occurence : fin !

        self.codage_bord=codage_bord
        self.codage_50c=codage_50c
        self.codage_repet=codage_repet

        self.dtype = self.get_type_board()

        self.longueur_max = 2*taille*taille
        self.longueur_moyenne = taille*taille/2

        self.repetition_max = repetition_max


        self.taille = taille
        self.panels = self.get_panels()
        self.init()
        self.first_init()

        if repetition_max > 0 or taille >= 18:
            sys.setrecursionlimit(2500)

    def get_type_board(self):
        return 'int8'  # 'float32' # 'float'

    def get_panels(self):
        return 3

    def get_code_bord(self):
        p = np.zeros((self.taille-2, self.taille-2))

        board_vertical = np.array([[1] for i in range(self.taille-2)], dtype=self.dtype)
        bord_horizontal = np.array([[1 for i in range(self.taille)]], dtype=self.dtype)

        return np.vstack((bord_horizontal, np.hstack((board_vertical, p, board_vertical)), bord_horizontal))

    def first_init(self):
        """"""
        #sys.setrecursionlimit(1500)

    def init(self):
        #if self.codage_repet:
        self.nb_repet = 0

        self.pre_nul = False
        self.tour_restant = inf

        self.pions_blanc = set()
        self.pions_noir = set()

        self.dames_blanc = set()
        self.dames_noir = set()


        self.tour_sans_prise_ni_deplacement_pions=0

        self.hash_old_positions = []

        self.mobilite_cumule_blanc = 0
        self.mobilite_cumule_noir = 0
        self.nb_coup_blanc = 0
        self.nb_coup_noir = 0

        self.tour = 0

        self.blancJoue = False



        self.plateau = np.zeros((self.taille, self.taille, self.panels+self.codage_50c+self.codage_bord+self.codage_repet), dtype=self.dtype)

        if self.codage_bord:
            self.plateau[:, :, -1] = self.get_code_bord()

        self.historique = []

        self.fini = False
        self.gagnant = None

        nb_ranger = int(self.taille/2-1)
        for i in range(nb_ranger):
            for j in range(self.taille):

                if (i + j) % 2 == 1:
                    self.ajouter_pion((i,j), True)

        for i in range(self.taille-nb_ranger, self.taille):
            for j in range(self.taille):

                if (i + j) % 2 == 1:
                    self.ajouter_pion((i, j), False)
        #print(self.pions_blanc)
        #print(self.pions_noir)
        self.actu_couleur_plateau()
        self.actu_nb_repet()
        self.calcul_coups_licites()


    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, 2] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, 2] = -np.ones((self.taille, self.taille), dtype=self.dtype)

        if self.codage_50c:
            self.plateau[:, :, -1-self.codage_bord] = self.tour_sans_prise_ni_deplacement_pions * np.ones((self.taille, self.taille), dtype=self.dtype)

    def actu_nb_repet(self):
        if self.codage_repet:
            self.plateau[:, :, -1-self.codage_50c-self.codage_bord] = self.nb_repet * np.ones((self.taille, self.taille), dtype=self.dtype)

    def hash_plateau(self):
        """if not self.codage_50c and not self.codage_repet:
            return self.plateau.tobytes()
        else:"""
        return self.plateau[:, :, :3].tobytes()

    def undo(self):
        #print('u')

        if self.hash_old_positions:
            self.hash_old_positions.pop()

        self.tour -= 1

        self.fini = False
        self.gagnant = None

        #if self.codage_repet:
        (piece, action), estpions, self.blancJoue, self.pre_nul, self.tour_restant, self.tour_sans_prise_ni_deplacement_pions, promotion, self.coups_licites, self.nb_repet = self.historique.pop()
        #else:
        #    (piece, action), estpions, self.blancJoue, self.pre_nul, self.tour_restant, self.tour_sans_prise_ni_deplacement_pions, promotion, self.coups_licites = self.historique.pop()

        if isinstance(action[0], int):
            final = action

        else:
            mouvement, prises = action
            final = mouvement[-1]

            for prise, estpion in zip(prises, estpions):
                if estpion:
                    self.ajouter_pion(prise, not self.blancJoue)
                else:
                    self.ajouter_dame(prise, not self.blancJoue)

        if promotion:
            self.retirer_dame(final, self.blancJoue)
            self.ajouter_pion(piece, self.blancJoue)
        elif (self.blancJoue and final in self.pions_blanc) or (not self.blancJoue and final in self.pions_noir) :
            self.retirer_pion(final, self.blancJoue)
            self.ajouter_pion(piece, self.blancJoue)
        elif (self.blancJoue and final in self.dames_blanc) or (not self.blancJoue and final in self.dames_noir):
            self.retirer_dame(final, self.blancJoue)
            self.ajouter_dame(piece, self.blancJoue)
        else:
            raise


        self.actu_couleur_plateau()
        self.actu_nb_repet()

        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc -= mobilite
            self.nb_coup_blanc -= 1
        else:
            self.mobilite_cumule_noir -= mobilite
            self.nb_coup_noir -= 1

    def raz(self):
        self.init()

    def jouer(self, piece, action):
        #print(piece, action)
        mobilite = len(self.coupsLicites())
        if self.blancJoue:
            self.mobilite_cumule_blanc += mobilite
            self.nb_coup_blanc += 1
        else:
            self.mobilite_cumule_noir += mobilite
            self.nb_coup_noir += 1

        if (piece, action) not in self.coupsLicites():
            print('ouille ... : ', piece, action, 'n est pas dans', self.coupsLicites())
            print(self.blancJoue)
            print(self.plateau[:, :, 0])

            raise Exception('non legal')

        if self.repetition_max is not None:
            self.hash_old_positions.append(self.hash_plateau())

        self.tour += 1

        """if self.tour > 900:
            f=open('dames_debug_histo_'+str(self.tour),'wb')
            dump([e[0] for e in self.historique],f)
            f.close()"""

        tour_sans_prise_ni_deplacement_pions = self.tour_sans_prise_ni_deplacement_pions
        self.tour_sans_prise_ni_deplacement_pions+=1

        if self.blancJoue:
            pions = self.pions_blanc
            dames = self.dames_blanc

            pions_adv = self.pions_noir

        else:
            pions = self.pions_noir
            dames = self.dames_noir

            pions_adv = self.pions_blanc

        if piece in pions:
            self.tour_sans_prise_ni_deplacement_pions = 0
            self.retirer_pion(piece, self.blancJoue)
            est_dame = False
        elif piece in dames:
            self.retirer_dame(piece, self.blancJoue)
            est_dame = True
        else:
            raise

        estpions = []
        if isinstance(action[0], int):
            if est_dame:
                self.ajouter_dame(action, self.blancJoue)
                promotion = False
            else:
                self.ajouter_pion(action, self.blancJoue)
                #self.coup = piece, action
                promotion = self.test_promotion(action)
        else:
            mouvement, prises = action

            self.tour_sans_prise_ni_deplacement_pions = 0

            if est_dame:
                self.ajouter_dame(mouvement[-1], self.blancJoue)
            else:
                self.ajouter_pion(mouvement[-1], self.blancJoue)


            assert len(prises) # si crash alors "self.tour_sans_prise_ni_deplacement_pions" ne doit être mis à 0 que si len(prises) > 0

            for prise in prises:
                if prise in pions_adv:
                    self.retirer_pion(prise, not self.blancJoue)
                    estpions.append(True)
                else:
                    self.retirer_dame(prise, not self.blancJoue)
                    estpions.append(False)

            if est_dame:
                promotion = False
            else:
                promotion = self.test_promotion(mouvement[-1])

        #if self.codage_repet:
        self.historique.append(((piece, action), estpions, self.blancJoue, self.pre_nul, self.tour_restant, tour_sans_prise_ni_deplacement_pions, promotion, list(self.coups_licites), self.nb_repet))
        #else:
        #    self.historique.append(((piece, action), estpions, self.blancJoue, self.pre_nul, self.tour_restant, tour_sans_prise_ni_deplacement_pions, promotion, list(self.coups_licites)))

        if self.pre_nul:
            self.tour_restant -= 1

        self.blancJoue = not self.blancJoue

        self.actu_couleur_plateau()
        self.nb_repet = self.hash_old_positions.count(self.hash_plateau())
        if self.codage_repet:
            self.actu_nb_repet()
        self.calcul_coups_licites()


        self.test_fini()


        if self.fini:
            self.mobilite_cumule_diff_pour_nul = self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc
            self.mobilite_cumule_frac_pour_nul = (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

            if self.gagnant == 'noir':
                self.mobilite = mobilite
                self.mobilite_cumule = self.mobilite_cumule_noir / self.nb_coup_noir
                self.mobilite_cumule_diff = max(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, 1)
                self.mobilite_cumule_frac = 1 * (self.mobilite_cumule_noir / self.nb_coup_noir) / (self.mobilite_cumule_blanc / self.nb_coup_blanc)

                self.presence_frac = (len(self.pions_noir) + 1) / (len(self.pions_blanc) + 1)
            elif self.gagnant == 'blanc':
                self.mobilite = -mobilite
                self.mobilite_cumule = -self.mobilite_cumule_blanc / self.nb_coup_blanc
                self.mobilite_cumule_diff = min(self.mobilite_cumule_noir / self.nb_coup_noir - self.mobilite_cumule_blanc / self.nb_coup_blanc, -1)
                self.mobilite_cumule_frac = -1 * (self.mobilite_cumule_blanc / self.nb_coup_blanc) / (self.mobilite_cumule_noir / self.nb_coup_noir)

                self.presence_frac = -(len(self.pions_blanc) + 1) / (len(self.pions_noir) + 1)
            else:
                self.mobilite = 0
                self.mobilite_cumule = 0
                self.mobilite_cumule_diff = 0
                self.mobilite_cumule_frac = 0

                self.presence_frac = 0


    def calcul_coups_licites(self):

        #print(self.plateau[:,:,0])

        L = []

        if self.blancJoue:
            pions = self.pions_blanc
            pions_adv = self.pions_noir
            dames = self.dames_blanc
            dames_adv = self.dames_noir
        else:
            pions = self.pions_noir
            pions_adv = self.pions_blanc
            dames =  self.dames_noir
            dames_adv =self.dames_blanc

        L = []

        for p in pions:
            L.extend(self.prises_possibles_pions(p, pions_adv, dames_adv))

        for d in dames:
            L.extend(self.prises_possibles_dames(d, pions_adv, dames_adv))

        if L:

            lmax = max([len(prises[1][1]) for prises in L])
            self.coups_licites = [prises for prises in L if len(prises[1][1]) == lmax]
        else:
            self.coups_licites =  self.get_deplacements(pions, dames, self.blancJoue)



    def in_board(self, i, j):
        return 0 <= i < self.taille and 0 <= j < self.taille


    def coupsLicites(self):
        return list(self.coups_licites)


    def test_fini(self):

        if not self.coupsLicites():

            self.fini = True
            self.score = 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc))
            self.score2 = 30 * len(self.dames_noir) + len(self.pions_noir) - (30 * len(self.dames_blanc) + len(self.pions_blanc))
            if self.blancJoue:
                self.gagnant = 'noir'

                self.score3 = max(1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)))
                self.score4 = max(1, 30 * len(self.dames_noir) + len(self.pions_noir) - (30 * len(self.dames_blanc) + len(self.pions_blanc)))
                self.score5 = abs(self.score) + 1
                self.score6 = self.score
                if self.score6 <= 0:
                    self.score6 = 20
                self.score7 = max(1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)) + 1)
                self.score8 = max(1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)) + 1) + 1
            else:
                self.gagnant = 'blanc'

                self.score3 = min(-1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)))
                self.score4 = min(-1, 30 * len(self.dames_noir) + len(self.pions_noir) - (30 * len(self.dames_blanc) + len(self.pions_blanc)))
                self.score5 = -abs(self.score) -1
                self.score6 = self.score
                if self.score6 >= 0:
                    self.score6 = -20
                self.score7 = min(-1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)) -1)
                self.score8 = min(-1, 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc)) + 1) -1
        else:

            est_nul = False

            if self.tour_sans_prise_ni_deplacement_pions >= 50 or self.tour_restant == 0:
                #print('r50 :',self.tour_sans_prise_ni_deplacement_pions,self.tour_restant)
                est_nul = True
            elif not self.pions_blanc and not self.pions_noir and ( (len(self.dames_blanc) == 1 and len(self.dames_noir) <= 2) or (len(self.dames_noir) == 1 and len(self.dames_blanc) <= 2) ):
                #print('que dames')
                est_nul = True

            elif self.repetition_max is not None and ((not self.codage_repet and self.nb_repet > self.repetition_max) or (self.codage_repet and self.nb_repet > self.repetition_max)):#self.hash_old_positions.count(self.hash_plateau())
                #print('repet')
                est_nul = True
            else:
                if not self.pre_nul:
                    if len(self.dames_blanc) == 1 and len(self.pions_blanc) == 0 and (len(self.dames_noir)==3 and len(self.pions_noir) == 0 or len(self.dames_noir) == 2 and len(self.pions_noir) == 1 or len(self.dames_noir) == 1 and len(self.pions_noir) == 2):
                        self.pre_nul = True
                        self.tour_restant = 32

                    if len(self.dames_noir) == 1 and len(self.pions_noir) == 0 and (len(self.dames_blanc)==3 and len(self.pions_blanc) == 0 or len(self.dames_blanc) == 2 and len(self.pions_blanc) == 1 or len(self.dames_blanc) == 1 and len(self.pions_blanc) == 2):
                        self.pre_nul = True
                        self.tour_restant = 32

            if est_nul:
                self.fini = True
                self.score = 0
                self.gagnant = 'nul'
                self.score2 = (30*len(self.dames_noir) + len(self.pions_noir) - 30*len(self.dames_blanc) - len(self.pions_blanc)) / 31*31
                self.score5 = 0
                self.score6 = 0
                self.score3 = 0
                self.score4 = (30 * len(self.dames_noir) + len(self.pions_noir) - 30 * len(self.dames_blanc) - len(self.pions_blanc)) / 31 * 31
                self.score7 = 0
                self.score8 = 0

        if self.fini:
            self.score_pour_nul = 20 * len(self.dames_noir) + len(self.pions_noir) - (20 * len(self.dames_blanc) + len(self.pions_blanc))



    def symetrie_joueur(self, plateau):
        return np.flip(-1*plateau, 0)


    def get_deplacements(self, pions, dames, blanc_joue):

        L = []

        for p in pions:

            if blanc_joue:
                p1 = p[0]+1, p[1]-1
                p2 = p[0]+1, p[1]+1
            else:
                p1 = p[0] - 1, p[1] - 1
                p2 = p[0] - 1, p[1] + 1

            if self.in_board(*p1) and self.est_vide(p1):
                L.append((p, p1))

            if self.in_board(*p2) and self.est_vide(p2):
                L.append((p, p2))

        for d in dames:
            for dx in -1, 1:
                for dy in -1, 1:
                    c = 1
                    pos = d[0]+c*dx, d[1]+c*dy
                    while self.in_board(*pos) and self.est_vide(pos):
                        L.append((d, pos))
                        c += 1
                        pos = d[0] + c * dx, d[1] + c * dy

        return L

    def prises_possibles_pions(self, p, pions_adv, dames_adv):
        L= []
        self.prises_possibles_pions_rec(p,p,[],[], L, pions_adv, dames_adv)
        return L


    def prises_possibles_dames(self, p, pions_adv, dames_adv):
        L= []
        self.prises_possibles_dames_rec(p,p,[],[], L, pions_adv, dames_adv)
        return L


    def get_prises(self, piece, prises, pions_adv, dames_adv, pos_init):
        L=[]
        for i in -1, 1:
            for j in -1, 1:
                
                final = piece[0] + 2*i, piece[1] + 2*j
                
                if self.in_board(*final) and (self.est_vide(final) or final == pos_init):
                    
                    prise = piece[0] + i, piece[1] + j
                    
                    if (prise in pions_adv or prise in dames_adv) and prise not in prises:
                        L.append((final, prise))
        return L
                        
    def est_vide(self, position):
        return position not in self.pions_blanc and position not in self.pions_noir and position not in self.dames_blanc and position not in self.dames_noir


    def prises_possibles_pions_rec(self, position_initial, position_actuel, mouvement_en_cours, prises_effectuees, mouvements, pions_adv, dames_adv):

        res = self.get_prises(position_actuel, prises_effectuees, pions_adv, dames_adv, position_initial)

        if not res:
            if prises_effectuees:
                mouvements.append((position_initial, (tuple(mouvement_en_cours), tuple(prises_effectuees))))
        else:
            for arrive, prise in res:
                mouvement_en_cours.append(arrive)
                prises_effectuees.append(prise)
                self.prises_possibles_pions_rec(position_initial, arrive, mouvement_en_cours, prises_effectuees, mouvements, pions_adv, dames_adv)
                mouvement_en_cours.pop()
                prises_effectuees.pop()

    def prises_possibles_dames_rec(self, position_initial, position_actuel, mouvement_en_cours, prises_effectuees, mouvements, pions_adv, dames_adv):

        res = self.get_prises_dames(position_actuel, prises_effectuees, pions_adv, dames_adv, position_initial)

        if not res:
            if prises_effectuees:
                mouvements.append((position_initial, (tuple(mouvement_en_cours), tuple(prises_effectuees))))
        else:
            for arrive, prise in res:
                mouvement_en_cours.append(arrive)
                prises_effectuees.append(prise)
                self.prises_possibles_dames_rec(position_initial, arrive, mouvement_en_cours, prises_effectuees, mouvements, pions_adv, dames_adv)
                mouvement_en_cours.pop()
                prises_effectuees.pop()



    def get_prises_dames(self, piece, prises, pions_adv, dames_adv, pos_init):
        L= []
        for dx in -1, 1:
            for dy in -1, 1:
                c = 1

                case = piece[0] + c*dx, piece[1] + c*dy

                while self.in_board(*case) and self.est_vide(case):
                    c += 1
                    case = piece[0] + c * dx, piece[1] + c * dy

                if (case in pions_adv or case in dames_adv) and case not in prises:
                    prise = case
                    c += 1
                    case = piece[0] + c * dx, piece[1] + c * dy

                    while self.in_board(*case) and (self.est_vide(case) or case == pos_init):
                        L.append((case, prise))
                        c += 1
                        case = piece[0] + c * dx, piece[1] + c * dy
        return L


    def test_promotion(self, position):

        if (self.blancJoue and position[0] == self.taille-1) or (not self.blancJoue and position[0] == 0):
            #print(self.plateau[:,:, 0])
            #print(self.plateau[:, :, 1])
            #print(self.coup)
            self.retirer_pion(position, self.blancJoue)
            self.ajouter_dame(position, self.blancJoue)

            return True


    def retirer_pion(self, position, blanc_joue):
        if blanc_joue:
            pions = self.pions_blanc
        else:
            pions = self.pions_noir

        pions.remove(position)
        self.plateau[position + (0,)] = 0


    def ajouter_pion(self, position, blanc_joue):
        if blanc_joue:
            self.pions_blanc.add(position)
            self.plateau[position + (0,)] = 1
        else:
            self.pions_noir.add(position)
            self.plateau[position + (0,)] = -1

    def retirer_dame(self, position, blanc_joue):
        if blanc_joue:
            dames = self.dames_blanc
        else:
            dames = self.dames_noir

        dames.remove(position)
        self.plateau[position +(1,)] = 0

    def ajouter_dame(self, position, blanc_joue):
        if blanc_joue:
            self.dames_blanc.add(position)
            self.plateau[position + (1,)] = 1
        else:
            self.dames_noir.add(position)
            self.plateau[position + (1,)] = -1



if __name__ == '__main__':
    s = Dames()

