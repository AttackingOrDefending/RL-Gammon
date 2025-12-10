from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece


class Cheval(Piece):

    def deplacements(self, i, j, jeu):
        L = []

        inter = jeu.inter_general(i, j)

        if not inter:

            for d in [-1, 1]:
                if (i+d,j) in jeu.positions_vides:
                    for p in [(i+2*d,j+1), (i+2*d,j-1)]:
                        #if not jeu.hors_jeu(*p) and (not p in jeu.pieces_blanc and jeu.blancJoue or not p in jeu.pieces_noir and not jeu.blancJoue):
                            L.append(p)

                if (i,j+d) in jeu.positions_vides:
                    for p in [(i+1,j+2*d), (i-1,j+2*d)]:
                        #if not jeu.hors_jeu(*p) and (not p in jeu.pieces_blanc and jeu.blancJoue or not p in jeu.pieces_noir and not jeu.blancJoue):
                            L.append(p)

        return L