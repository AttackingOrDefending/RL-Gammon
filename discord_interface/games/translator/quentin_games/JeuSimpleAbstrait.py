import numpy as np


taille_par_defaut = 8

class JeuSimpleAbstrait():


    def __init__(self, taille=taille_par_defaut):

        self.dtype = self.get_type_board()

        self.taille = taille

        self.panels = self.get_nb_panels()

        self.init()

        #self.longueur_max = taille * taille
        #self.longueur_moyenne = self.longueur_max


    def get_nb_panels(self):
        return 4


    def init(self):


        self.coups_licites = []

        self.fini = False

        self.gagnant = None
        #print(self.panels)
        self.plateau = np.zeros((self.taille, self.taille, self.panels), dtype=self.dtype)


        self.blancJoue = False

        self.actu_couleur_plateau()

        self.init_plateau()

        self.historique = []  # deque() # les opérations associées sont plus rapides mais le total est pourtant plus lent ???

        self.calcul_coups_licites()





    def actu_couleur_plateau(self):
        if self.blancJoue:
            self.plateau[:, :, -1] = np.ones((self.taille, self.taille), dtype=self.dtype)
        else:
            self.plateau[:, :, -1] = -np.ones((self.taille, self.taille), dtype=self.dtype)

    def undo(self,):

        (i,j), self.blancJoue, self.coups_licites= self.historique.pop()

        self.actu_couleur_plateau()


        self.fini = False

        self.gagnant = None




    def calcul_coups_licites(self):
        raise NotImplementedError()

    def test_fini(self):
        raise NotImplementedError()

    def raz(self):

        self.init()

    def jouer(self, i, j):

        self.historique.append(((i,j), self.blancJoue, list(self.coups_licites)))

        self.blancJoue = not self.blancJoue
        self.actu_couleur_plateau()

        self.calcul_coups_licites()

        self.test_fini()


    def coupsLicites(self):

        return list(self.coups_licites)

