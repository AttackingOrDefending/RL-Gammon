from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece


class Elephant(Piece):

    def deplacements(self, i, j, jeu):
        L = []

        inter = jeu.inter_general(i, j)

        if not inter:
            for di in [-2, 2]:
                for dj in [-2, 2]:
                    if i <= 4 and i+di <=4 or i>=5 and i+di >=5:
                        if (i+di/2, j+dj/2) not in jeu.pieces_noir and (i+di/2, j+dj/2) not in jeu.pieces_blanc:
                            L.append((i+di, j+dj))

        return L