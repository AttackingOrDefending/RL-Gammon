from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece

class Chariot(Piece):

    def deplacements(self, i, j, jeu):
        L = []

        inter = jeu.inter_general(i, j)

        d = 1
        while i + d < jeu.longueur:

            if (i+d,j) in jeu.pieces_noir or (i+d,j) in jeu.pieces_blanc:
                break
            L.append((i + d, j))
            d+=1
        p = (i+d,j)
        if jeu.blancJoue and p in jeu.pieces_noir or not jeu.blancJoue and p in jeu.pieces_blanc:
            L.append(p)

        d = 1
        while i - d >= 0:

            if (i - d, j) in jeu.pieces_noir or (i - d, j) in jeu.pieces_blanc:
                break
            L.append((i - d, j))
            d += 1
        p = (i - d, j)
        if jeu.blancJoue and p in jeu.pieces_noir or not jeu.blancJoue and p in jeu.pieces_blanc:
            L.append(p)

        if not inter:
            d = 1
            while j + d < jeu.largeur:

                if (i, j + d) in jeu.pieces_noir or (i, j + d) in jeu.pieces_blanc:
                    break
                L.append((i, j + d))
                d += 1
            p = (i, j + d)
            if jeu.blancJoue and p in jeu.pieces_noir or not jeu.blancJoue and p in jeu.pieces_blanc:
                L.append(p)

            d = 1
            while j - d >= 0:

                if (i, j - d) in jeu.pieces_noir or (i, j - d) in jeu.pieces_blanc:
                    break
                L.append((i, j - d))
                d += 1
            p = (i, j - d)
            if jeu.blancJoue and p in jeu.pieces_noir or not jeu.blancJoue and p in jeu.pieces_blanc:
                L.append(p)

        return L