from discord_interface.games.translator.quentin_games.deplacement.Piece import Piece


class General(Piece):



    def creer_vis_a_vis_deux_generaux(self, k, l, jeu):
        if jeu.blancJoue:
            i_adv, j_adv = jeu.general_noir

        else:
            i_adv, j_adv = jeu.general_blanc

        if not l == j_adv:
            return False

        inter = 0
        for i in range(min(k,i_adv) + 1, max(k,i_adv)):
            #print(i,l, inter)
            if (i, l) in jeu.pieces_noir or (i, l) in jeu.pieces_blanc:
                """inter += 1
                if inter > 1:"""
                return False

        return True

    def deplacements(self, i, j, jeu):
        L = []

        for k,l in [(i-1,j),(i+1,j)]:

            if not self.creer_vis_a_vis_deux_generaux(k,l, jeu):
                if (jeu.largeur-1)/2-1 <= l <= (jeu.largeur-1)/2+1 and (k <= 2 and jeu.blancJoue and (k,l) not in jeu.pieces_blanc or k >= jeu.longueur-3 and not jeu.blancJoue and (k,l) not in jeu.pieces_noir):
                    L.append((k,l))


        for k,l in [(i, j + 1), (i, j - 1)]:
            if not self.creer_vis_a_vis_deux_generaux(k,l, jeu):
                #print('+')
                if (jeu.largeur-1)/2-1 <= l <= (jeu.largeur-1)/2+1 and (k <= 2 and jeu.blancJoue and (k,l) not in jeu.pieces_blanc or k >= jeu.longueur-3 and not jeu.blancJoue and (k,l) not in jeu.pieces_noir):
                    L.append((k,l))

        return L