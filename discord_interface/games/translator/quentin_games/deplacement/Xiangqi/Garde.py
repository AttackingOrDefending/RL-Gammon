from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece


class Garde(Piece):

    def deplacements(self, i, j, jeu):
        L = []

        inter = jeu.inter_general(i, j)

        if not inter:
            for k,l in [(i-1,j-1),(i+1,j-1),(i-1,j+1),(i+1,j+1)]:
                if (jeu.largeur-1)/2-1 <= l <= (jeu.largeur-1)/2+1 and (k <= 2 and jeu.blancJoue and (k,l) not in jeu.pieces_blanc or k >= jeu.longueur-3 and not jeu.blancJoue and (k,l) not in jeu.pieces_noir):
                    L.append((k,l))

        return L
