import sys

from discord_interface.games.translator.quentin_games.deplacement.Jeu_a_deplacement import Jeu_a_deplacement
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Canon import Canon
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Chariot import Chariot
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Cheval import Cheval
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Elephant import Elephant
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Garde import Garde
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.General import General
from discord_interface.games.translator.quentin_games.deplacement.Xiangqi.Soldat import Soldat


class Xiangqi(Jeu_a_deplacement):


    def __init__(self, borne=200, codage_ultra_compact=False, num_type_hash=0, repetition_max=0, defaite_si_repet=True, codage_nb_repet=False): # pour tournois 1er joueur :  repetition_max=1, defaite_si_repet=False # pour tournois 2er joueur : repetition_max=1, defaite_si_repet=True  # APP : repetition_max=0, defaite_si_repet=True ## NE SURTOUT PAS METTRE defaite_si_repet À fA

        canon=Canon()
        cheval=Cheval()
        elephant=Elephant()
        garde=Garde()
        general=General()
        soldat=Soldat()
        tour=Chariot()

        liste_pieces = [canon, cheval,elephant,garde,general,soldat,tour]

        valeurs={}
        valeurs[canon]= 4.5
        valeurs[cheval]= 4
        valeurs[elephant]= 2
        valeurs[garde]= 2
        valeurs[general]= 0
        valeurs[soldat]= 1.5
        valeurs[tour]= 9

        positions_compacts = [((0,4), general), ((0,3), garde), ((0, 2), elephant), ((0,1), cheval), ((0,0), tour), ((2, 1), canon), ((3, 0), soldat), ((3,2),soldat), ((3,4),soldat)]


        positions_init_pieces = []
        for (i,j), piece in positions_compacts:
            positions_init_pieces.append(((i,j),(piece,True)))
            if j != 8-j:
                positions_init_pieces.append(((i, 8-j), (piece, True)))

            positions_init_pieces.append(((9-i, j), (piece, False)))
            if j != 8 - j:
                positions_init_pieces.append(((9-i, 8 - j), (piece, False)))


        piece_a_prendre = general

        sys.setrecursionlimit(1501)

        super().__init__(10, 9, liste_pieces, positions_init_pieces, piece_a_prendre, borne=borne, codage_ultra_compact=codage_ultra_compact, num_type_hash=num_type_hash, repetition_max=repetition_max, valeurs_pieces=valeurs, defaite_si_repet=defaite_si_repet, codage_nb_repet=codage_nb_repet)


    def init(self):

        self.general_noir = (9,4)
        self.general_blanc = (0,4)

        super().init()

    def inter_general(self, i_piece, j_piece):

        i_b, j_b = self.general_blanc
        i_n, j_n = self.general_noir

        if not j_b == j_n or not j_b == j_piece:
            return False

        assert i_b < i_n

        inter = 0
        for i in range(i_b+1, i_n):
            if (i, j_b) in self.pieces_noir or (i, j_b) in self.pieces_blanc:
                inter+=1

            if inter > 1:
                return False

        return True


    def deplacer(self, i, j, k, l):
        res = super().deplacer(i, j, k, l)

        if self.type_pieces[k, l].__class__.__name__ == 'General':
            if self.blancJoue:
                self.general_blanc = (k, l)
            else:
                self.general_noir = (k, l)

        return res

if __name__ == '__main__':

    jeu = Xiangqi()
    print(jeu.plateau[:,:,-1])
    print(jeu.bordification(jeu.plateau)[:,:,-2])
    print(jeu.bordification(jeu.plateau).ndim)