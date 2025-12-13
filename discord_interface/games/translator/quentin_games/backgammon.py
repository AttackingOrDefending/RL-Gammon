

import numpy as np


class Backgammon():


    def test_fini(self):

        if self.plateau[25, 0, 0] == 15:
            self.fini = True
            self.gagnant = 'noir'
            assert self.score_joueur[0] - self.score_joueur[1] > 0

        elif self.plateau[0, 0, 1] == 15:
            assert self.score_joueur[0] - self.score_joueur[1] < 0
            self.fini = True
            self.gagnant = 'blanc'

        if not self.fini:
            if self.borne and len(self.historique) > self.borne:
                self.fini = True
                if self.score_joueur[0] - self.score_joueur[1] > 0:
                    self.gagnant = 'noir'

                elif self.score_joueur[0] - self.score_joueur[1] < 0:
                    self.gagnant = 'blanc'

                else:
                    assert self.score_joueur[0] - self.score_joueur[1] == 0
                    self.gagnant = 'nul'

        if self.fini:
            self.calcul_score()

    def positions_jans(self, joueur=None):
        if joueur == None:
            return list(range(1+18,1+24)) + list(range(1+0,1+6))
        elif joueur == 0:
            return list(range(1+18,1+24))
        elif joueur == 1:
            return list(range(1+0,1+6))
        else:
            raise Exception()


    def calcul_score(self):
        self.score = self.score_joueur[0] - self.score_joueur[1]

    def init(self):
        self.arrivee = {}
        self.arrivee[0] = 25
        self.arrivee[1] = 0

        self.score = 0
        self.fini = False
        self.gagnant = None

        self.des1 = 0
        self.des2 = 0

        self.score_joueur = {}
        self.score_joueur[0] = 0
        self.score_joueur[1] = 0

        self.nombre_dans_jan = {}
        self.nombre_dans_jan[0] = 5
        self.nombre_dans_jan[1] = 5

        self.barre = {}
        self.barre[0] = 0
        self.barre[1] = 0

        self.joueur_en_cours = -1
        self.prochain_joueur = 0

        self.plateau = self.create_backgammon_state()

        self.historique = []

        self.actu_plateau()
        self.calcul_coups_licites()



    def borner(self, x):
        return min(max(x, 0), 25)

    def calcul_coups_licites(self):

        self.coups_licites = []
        self.probabilite_actions = None

        if len(self.historique) == 0:
            self.probabilite_actions = {}

            for des1 in range(1,7):
                for des2 in range(1,7):
                    if des1 != des2:
                        try:
                            self.probabilite_actions[max(des1, des2), min(des1, des2)] += 1
                        except:
                            self.probabilite_actions[max(des1, des2), min(des1, des2)] = 1

            self.coups_licites = list(self.probabilite_actions.keys())
            #assert set(self.coups_licites) == set(self.probabilite_actions.keys())
        elif self.joueur_en_cours == -1:

            self.probabilite_actions = {}
            for des1 in range(1,7):
                for des2 in range(1,7):
                    try:
                        self.probabilite_actions[max(des1, des2), min(des1, des2)] += 1
                    except:
                        self.probabilite_actions[max(des1, des2), min(des1, des2)] = 1

            self.coups_licites = list(self.probabilite_actions.keys())
            #assert set(self.coups_licites) == set(self.probabilite_actions.keys())
        else:


                if self.des1 == self.des2:
                    L = []
                    self.calcul_coups_licites_double_rec(self.plateau, [], self.des1, L, self.nombre_dans_jan[self.joueur_en_cours])

                    if not L:
                        self.calcul_coups_licites_double_rec(self.plateau, [], self.des1, L, self.nombre_dans_jan[self.joueur_en_cours], N=3)

                        if not L:
                            self.calcul_coups_licites_double_rec(self.plateau, [], self.des1, L, self.nombre_dans_jan[self.joueur_en_cours], N=2)

                            if not L:
                                self.calcul_coups_licites_double_rec(self.plateau, [], self.des1, L, self.nombre_dans_jan[self.joueur_en_cours], N=1)

                                if not L:
                                    L = [(None, None)]

                    self.coups_licites = L
                else:

                    L = []
                    self.calcul_coups_licites_simple_rec(self.plateau, [], [self.des1, self.des2], [], L, self.nombre_dans_jan[self.joueur_en_cours])
                    if L:
                        self.coups_licites = L
                    else:
                        self.calcul_coups_licites_simple_rec(self.plateau, [], [self.des1,], [], L, self.nombre_dans_jan[self.joueur_en_cours], N=1)
                        if L:
                            self.coups_licites = [(cp[0], None) for cp in L]
                        else:
                            self.calcul_coups_licites_simple_rec(self.plateau, [], [self.des2, ], [], L, self.nombre_dans_jan[self.joueur_en_cours], N=1)
                            if L:
                                self.coups_licites = [(cp[0], None) for cp in L]
                            else:
                                self.coups_licites = [(None, None)]


        self.coups_licites = list(set(self.coups_licites))

        if self.probabilite_actions is not None:
            s = sum(self.probabilite_actions.values())
            for cp in self.probabilite_actions:
                self.probabilite_actions[cp] /= s

    def calcul_coups_licites_simple_rec(self, plateau, a, des, des_choisis, L, nombre_dans_jan, N=2):
        #print('?', N)
        if N==0:
            L.append(tuple(a))
        else:
            for de in des:
                if de not in des_choisis:
                    des_choisis.append(de)
                    #assert self.pieces_deplacables(plateau, de, nombre_dans_jan)
                    for p in self.pieces_deplacables(plateau, de, nombre_dans_jan):
                        a.append((p, de))
                        P, nombre_dans_jan2 = self.deplacer_virtuel(p, de, plateau, nombre_dans_jan)
                        self.calcul_coups_licites_simple_rec(P, a, des, des_choisis, L, nombre_dans_jan2, N-1)
                        a.pop()
                    des_choisis.pop()

    def calcul_coups_licites_double_rec(self, plateau, a, de, L, nombre_dans_jan, N=4):
        #print('??',N)
        if N==0:
            L.append((tuple(a), de))
        else:
            #assert self.pieces_deplacables(plateau, de, nombre_dans_jan)
            for p in self.pieces_deplacables(plateau, de, nombre_dans_jan):
                a.append(p)
                P, nombre_dans_jan2 = self.deplacer_virtuel(p, de, plateau, nombre_dans_jan)
                self.calcul_coups_licites_double_rec(P, a, de, L, nombre_dans_jan2, N-1)
                a.pop()

    def pieces_deplacables(self, plateau, de, nombre_dans_jan):

        if self.joueur_en_cours:
            v = -de
        else:
            v = de
        #print([i for i in np.where(plateau[:, self.joueur_en_cours] > 0)])
        #positions = [int(i) for i in np.where(plateau[:, self.joueur_en_cours] > 0)]
        positions = [i for i in range(0,26) if plateau[i, 0, self.joueur_en_cours] > 0]
        if self.joueur_en_cours:
            if 0 in positions:
                positions.remove(0)
            if 25 in positions:
                positions = [25]
        else:
            if 25 in positions:
                positions.remove(25)
            if 0 in positions:
                positions=[0]

        #print('!',positions)
        if nombre_dans_jan < 15:
            L = []
            for p in positions:
                #print(p, v, plateau[p+v, not self.joueur_en_cours])
                """if isinstance(p, tuple) or isinstance(v, tuple):
                    print(p, v, '(', self.joueur_en_cours, self.coups_licites,-self.plateau[:,0]+self.plateau[:,1])"""
                if 0 < p + v < 25 and plateau[p+v, 0, int(not self.joueur_en_cours)] < 2:
                    L.append(p)
            return L

        else:
            if self.joueur_en_cours:
                p = de
            else:
                p = 25 - de
            #print('!',p, positions)

            if self.joueur_en_cours:
                positions_avant = range(6, p, -1)
            else:
                positions_avant = range(19, p)
            positions_avant = [p for p in positions_avant if p in positions and plateau[p + v, 0, int(not self.joueur_en_cours)] < 2]  # if 0 < p + v < 25 and plateau[p+v, int(not self.joueur_en_cours)] < 2:

            if p in positions:
                #print('?',[p])
                return [p] + positions_avant
            else:

                if positions_avant:
                    return positions_avant
                else:
                    if self.joueur_en_cours:
                        positions_apres = [p for p in range(1,p) if p in positions]
                        if positions_apres:
                            return [max(positions_apres)]
                        else:
                            return []
                    else:
                        positions_apres = [p for p in range(p+1,25) if p in positions]
                        if positions_apres:
                            return [min(positions_apres)]
                        else:
                            return []



    def deplacer_virtuel(self, position, de, plateau, nombre_dans_jan):

        P = np.copy(plateau)

        if self.joueur_en_cours:
            variation = -de
        else:
            variation = de

        dest = self.borner(position+variation)

        P[position, 0, self.joueur_en_cours] -= 1
        P[dest, 0, self.joueur_en_cours] += 1


        if position not in self.positions_jans(self.joueur_en_cours) and dest in self.positions_jans(self.joueur_en_cours):
            return P, nombre_dans_jan + 1
        else:
            return P, nombre_dans_jan


    def deplacer(self, position, de):
        #nombre_dans_jan_temp = self.nombre_dans_jan[self.joueur_en_cours]
        #p_temp = np.copy(self.plateau)
        if self.joueur_en_cours:
            variation = -de
        else:
            variation = de

        dest = self.borner(position+variation)

        self.plateau[position, 0, self.joueur_en_cours] -= 1
        self.plateau[dest, 0, self.joueur_en_cours] += 1

        if dest == self.arrivee[self.joueur_en_cours]:
            self.score_joueur[self.joueur_en_cours] += 1

        adv = int(not self.joueur_en_cours)
        if self.plateau[dest, 0, adv] == 1:
            self.plateau[dest, 0, adv] = 0
            if adv:
                self.plateau[25, 0, adv] += 1
            else:
                self.plateau[0, 0, adv] += 1
            self.barre[adv] += 1
            prise = True
            if dest in self.positions_jans(adv):
                self.nombre_dans_jan[adv] -= 1
        else:
            """if not self.plateau[dest, adv] == 0:
                print('\n*',position, de)
                print(-p_temp[:,0]+p_temp[:,1])
                print('>>>',self.pieces_deplacables(p_temp, de, nombre_dans_jan=nombre_dans_jan_temp))"""
            assert dest in [0,25] or self.plateau[dest, 0, adv] == 0
            prise = False

        if position not in self.positions_jans(self.joueur_en_cours) and dest in self.positions_jans(self.joueur_en_cours):
            self.nombre_dans_jan[self.joueur_en_cours] += 1

        return prise

    def actu_plateau(self):

        if self.joueur_en_cours == -1:
            self.plateau[:, 0, 2] = 0
            self.plateau[:, 0, 3] = 0
        else:
            self.plateau[:, 0, 2] = self.des1
            self.plateau[:, 0, 3] = self.des2

        self.plateau[:, 0, 4] = self.joueur_en_cours
        self.plateau[:, 0, 5] = self.prochain_joueur

        self.plateau[:, 0, 6] = (self.joueur_en_cours == -1)


    def jouer(self, a, b):
        #print(len(self.historique))
        """print(len(self.historique), self.joueur_en_cours,-self.plateau[:,0]+self.plateau[:,1])
        if self.joueur_en_cours !=-1:
            print('-')
            L = self.pieces_deplacables(self.plateau, self.des1, self.nombre_dans_jan[self.joueur_en_cours])
            print(L)
            print(list(L))
            print('--')
        print(a,b, self.coups_licites)
        print(self.nombre_dans_jan[0],self.nombre_dans_jan[1])"""
        prises = []
        self.historique.append((a, b, self.joueur_en_cours, self.prochain_joueur, list(self.coups_licites), dict(self.probabilite_actions) if self.probabilite_actions is not None else None, dict(self.arrivee), dict(self.score_joueur), self.score, dict(self.nombre_dans_jan), dict(self.barre), self.des1, self.des2, self.gagnant, self.fini, prises, np.copy(self.plateau)))

        if self.joueur_en_cours == -1:

            self.des1, self.des2 = a, b
            assert isinstance(a, int) and isinstance(b, int)

            self.joueur_en_cours = self.prochain_joueur
            self.prochain_joueur = int(not self.joueur_en_cours)

        else:

            if isinstance(a, tuple) and isinstance(b, int):

                for e in a:
                    prise = self.deplacer(e, b)
                    prises.append(prise)

            else:
                #print(a, b)
                #print(self.coups_licites)
                if a is not None:
                    prise = self.deplacer(*a)
                    prises.append(prise)

                    if b is not None:
                        prise = self.deplacer(*b)
                        prises.append(prise)

            self.test_fini()
            self.joueur_en_cours = -1

        self.calcul_coups_licites()
        self.actu_plateau()



    def undo(self):
        a, b, self.joueur_en_cours, self.prochain_joueur, self.coups_licites, self.probabilite_actions, self.arrivee, self.score_joueur, self.score, self.nombre_dans_jan, self.barre, self.des1, self.des2, self.gagnant, self.fini, prises, self.plateau = self.historique.pop()
        if self.joueur_en_cours==-1:
            assert set(self.coups_licites) == set(self.probabilite_actions.keys())

    def __init__(self):
        self.borne = None
        if self.borne:
            self.longueur_max = self.borne
        else:
            self.longueur_max = 500
        self.init()

    def raz(self):
        self.init()

    def create_backgammon_state(self):
        """
        Crée un état initial de backgammon sous forme de tableau numpy (26x8).

        Structure des 8 'panels' (2ème dimension) :
        0 : pièces du joueur 1 (0-12)
        1 : pièces du joueur 2 (0-12)
        2 : valeur du dé 1 (0-6) - 0 si non lancé
        3 : valeur du dé 2 (0-6) - 0 si non lancé
        4 : joueur en cours (booléen)
        5 : phase de lancement des dés (booléen)
        6 : zone de jan intérieur (booléen)
        7 : pions sur la barre (0-12)
        """

        # Tableau principal
        # 26 lignes : 24 cases de jeu + 2 zones d'arrivée
        state = np.zeros((26, 1, 13), dtype=np.int8)

        # ---------------------------
        # Initialisation des pièces
        # ---------------------------
        # Configuration classique du backgammon :
        # Index 0 = point 1 côté joueur 1, index 23 = point 24 côté joueur 1
        # Exemple d'une position initiale standard

        # Joueur 1
        state[1+0, 0, 0] = 2  # 2 pions sur point 1
        state[1+11, 0, 0] = 5  # 5 pions sur point 12
        state[1+16, 0, 0] = 3  # 3 pions sur point 17
        state[1+18, 0, 0] = 5  # 5 pions sur point 19

        # Joueur 2 (symétrique)
        state[1+23, 0, 1] = 2
        state[1+12, 0, 1] = 5
        state[1+7, 0, 1] = 3
        state[1+5, 0, 1] = 5

        # ---------------------------
        # Dés (0 au départ)
        # ---------------------------
        state[:, 0, 2] = 0  # dé 1
        state[:, 0, 3] = 0  # dé 2

        # ---------------------------
        # Joueur en cours (booléen)
        # ---------------------------
        state[:, 0, 4] = -1  # joueur 1 commence

        # ---------------------------
        # Prochain joueur (booléen)
        # ---------------------------
        state[:, 0, 5] = 0

        # ---------------------------
        # Phase de lancer de dés (booléen)
        # ---------------------------
        state[:, 0, 6] = 1  # vrai au début

        # ---------------------------
        # Zones de jan intérieur
        # ---------------------------
        # Pour chaque joueur : 6 cases juste avant sa zone d'arrivée
        # Joueur 1 = cases 18 à 23
        # Joueur 2 = cases 0 à 5
        state[1+18:1+24, 0, 7] = 1
        state[1+0:1+6, 0, 8] = 1

        # ---------------------------
        # Zone Pieces de la reserve joueur de la barre
        # ---------------------------
        state[0, 0, 9] = 1
        state[25, 0, 10] = 1

        # ---------------------------
        # Zone Pieces de sortie
        # ---------------------------
        state[25, 0, 11] = 1
        state[0, 0, 12] = 1

        return state

    def coupsLicites(self):
        return self.coups_licites
