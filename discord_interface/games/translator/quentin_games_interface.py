import numpy as np

from discord_interface.games.mygame import Game
from discord_interface.games.translator.traductor_methods import correspondance_action_python_ludii, correspondance_action_ludii_python


class InterfaceJeuDiscord(Game):

    """
    Ensuite, il faut ajouter ce jeu dans l'énumeration EnumGames (main\games\games_enum.py) de la manière suivante :
            class EnumGames(Enum):
                NEW_GAME = New_Game()
                ...

    Note : Pour que le jeu soit celui sélectionné par défaut, il faut le mettre en première position dans l'énumération.

    Enfin, il suffit d'exécuter main_referee.py pour pouvoir en profiter."""

    CORRESPONDENCE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',  #
                      12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S'}  #
    #
    ANTI_CORRESPONDENCE = {'C': 2, 'H': 7, 'S': 18, 'J': 9, 'E': 4, 'F': 5, 'O': 14, 'B': 1, 'R': 17, 'K': 10, 'I': 8,  #
                           'G': 6, 'L': 11, 'N': 13, 'Q': 16, 'P': 15, 'D': 3, 'A': 0, 'M': 12}  #
    #

    def __init__(self, nom, Jeu, params=None, rules='standard', move_keywords=[]):
        super().__init__(name = nom, rules = rules, move_keywords=move_keywords)

        if params is not None:
            self.jeu = Jeu(**params)
        else:
            self.jeu = Jeu()

        if hasattr(self.jeu, 'taille'):
            self.taille = self.jeu.taille
        elif hasattr(self.jeu, 'longueur'):
            self.longueur = self.jeu.longueur
            self.largeur = self.jeu.largeur
        elif hasattr(self.jeu, 'hauteur'):
            self.hauteur = self.jeu.hauteur
            self.hauteur = self.jeu.hauteur

        self.plateau = self.jeu.plateau

    def ended(self):
        return self.jeu.fini

    def terminate(self, winner):
        self.jeu.fini = True

        self.winner = winner


    def actu_winner(self):
        self.gagnant = self.jeu.gagnant
        if self.jeu.gagnant == 'noir':
            self.winner = 0
        elif self.jeu.gagnant == 'blanc':
            self.winner = 1
        elif self.jeu.gagnant == 'nul':
            self.winner = 0.5
        else:
            self.winner = None


    def reset(self):
        if hasattr(self.jeu, 'raz'):
            self.jeu.raz()
        else:
            self.jeu.init()

    def get_current_player(self):
        if self.jeu.blancJoue:
            return 1
        else:
            return 0

    def plays(self, move):
        #print(len(self.jeu.historique))
        if move not in self.valid_actions():
            print(move)
            print(self.jeu.coupsLicites())
            print([(a,b) for a, b, *_ in self.jeu.historique])
        assert move in self.valid_actions()
        self.jeu.jouer(*move)
        self.actu_winner()
        #print(len(self.jeu.historique))
        #print(self.jeu.plateau[:,:, 0])
        self.plateau = self.jeu.plateau
        self.fini = self.jeu.fini

    def jouer(self, a, b):
        self.plays((a,b))

    def undo(self):
        self.jeu.undo()
        self.actu_winner()
        self.plateau = self.jeu.plateau
        self.fini = self.jeu.fini

    def valid_actions(self):
        return self.jeu.coupsLicites()

    def coupsLicites(self):
        return self.valid_actions()

    def reorientation(self, coup):


        lettre = coup[0]
        chiffre = coup[1:]

        lettre = self.ANTI_CORRESPONDENCE[lettre]
        chiffre = int(chiffre)

        return chiffre - 1, lettre

    def action_to_string(self, object):

        return correspondance_action_python_ludii(self.name.lower().replace(' ','-'), self.jeu, object)
        #print(correspondance_action_python_ludii(self.name.lower().replace(' ','-'), self.jeu, object))

        """i, j = object

        if isinstance(i, tuple) and isinstance(j, tuple):
            i1, j1 = i
            i2, j2 = j

            return self.CORRESPONDENCE[j1] + str(i1 + 1) + '-' + self.CORRESPONDENCE[j2] + str(i2 + 1)
        else:
            return self.CORRESPONDENCE[j] + str(i + 1)"""


    def numbification(self, string):
        #print('>',string)
        if string[0].isdigit():
            return string
        else:
            lettre = string[0]
            chiffre = string[1:]

            if lettre in self.ANTI_CORRESPONDENCE and chiffre.isdigit():

                lettre=str(self.ANTI_CORRESPONDENCE[lettre])

                return lettre+','+str(int(chiffre)-1)

            else:
                return string

        # String.valueOf(letter - 'A') + "," + (Integer.parseInt(s.substring(1)) - 1);

    def string_to_action(self, string):

        if '-' in string:
            cp1, cp2 = string.split('-')

            move = self.numbification(cp1)+'-'+self.numbification(cp2)
        else:
            move = self.numbification(string)

        return correspondance_action_ludii_python(self.name.lower().replace(' ','-'), self.jeu, move)

        """if '-' in string:
            cp1, cp2 = string.split('-')

            return (self.reorientation(cp1), self.reorientation(cp2))
        else:
            return self.reorientation(string)"""

    def get_numpy_board(self, copy=True):# copy=False only if you do not modify the numpy board
        if copy:
            return np.copy(self.jeu.plateau)
        else:
            return self.jeu.plateau

    def show_game(self):
        print('TLDC!')#
        #correspondance_action_python_ludii(nom_jeu, jeu,correspondance_action_ludii_python(nom_jeu, jeu, move))