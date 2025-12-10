from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece


class Soldat(Piece):

    def deplacements(self, i, j, jeu):
        L = []

        inter = jeu.inter_general(i,j)

        if jeu.blancJoue:
            p = i+1,j
            if not jeu.hors_jeu(*p) and not p in jeu.pieces_blanc:
                L.append(p)

            if not inter:
                if i >= jeu.longueur /2:
                    p = i, j+1
                    if not jeu.hors_jeu(*p) and not p in jeu.pieces_blanc:
                        L.append(p)
                    p = i, j-1
                    if not jeu.hors_jeu(*p) and not p in jeu.pieces_blanc:
                        L.append(p)
        else:
            p = i-1,j
            if not jeu.hors_jeu(*p) and not p in jeu.pieces_noir:
                L.append(p)

            if not inter:
                if i < jeu.longueur /2:
                    p = i, j+1
                    if not jeu.hors_jeu(*p) and not p in jeu.pieces_noir:
                        L.append(p)
                    p = i, j-1
                    if not jeu.hors_jeu(*p) and not p in jeu.pieces_noir:
                        L.append(p)

        return L