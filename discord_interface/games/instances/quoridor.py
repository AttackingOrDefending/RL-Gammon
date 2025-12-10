from discord_interface.games.translator.quentin_games.Amazons import Amazons
from discord_interface.games.translator.quentin_games.Quoridor import Quoridor
from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord




class QuoridorDiscord(InterfaceJeuDiscord):

    def __init__(self):
        super().__init__('quoridor', Quoridor, params=None, rules='standard', move_keywords=[])


    def string_to_action(self, string):

        def trad(string):

            string = self.numbification(string)
            #print(string)
            lettre, chiffre = string.split(',')
            lettre = int(lettre)
            chiffre = int(chiffre)
            return self.jeu.taille - 1 - chiffre, lettre

        if string.count('-') == 1:
            (i1, i2), (j1, j2) = super().string_to_action(string)
            return (2*i1, 2*i2), (2*j1, 2*j2)
        else:

            l = string.split('-')

            if l[0][0] == l[1][0]:

                i1, i2 = trad(l[0])
                j1, j2 = trad(l[2])

                return (2*i1+1, 2*i2), (2*j1+1, 2*j2)

            else:

                i1, i2 = trad(l[0])
                j1, j2 = trad(l[2])

                return (2 * i1, 2 * i2 + 1), (2 * j1, 2 * j2 + 1)



        """def quoridor_pos(i, j):
            return correspondance[j] + str(Jeu.taille - i)

        if i1 % 2 == 0 and i2 % 2 == 0:
            return quoridor_pos(i1//2, j1//2) + '-' + quoridor_pos(i2//2, j2//2)
            #return correspondance[j1//2] + str(Jeu.taille - i1//2) + '-' + correspondance[j2//2] + str(Jeu.taille - i2//2)
        else:
            if i1 % 2 == 0:
                assert j1 % 2 == 0

                i1 = i1 // 2
                j1 = j1 // 2

                i2m = i2 // 2
                j2m = j2 // 2

                i2p = i2 // 2 + 1
                j2p = j2 // 2 + 1

                return quoridor_pos(i1, i2m) + '-' + quoridor_pos(i1, i2p) + '-' + quoridor_pos(j1, j2m) + '-' + quoridor_pos(j1, j2p)


            else:
                assert i2 % 2 == 0
                assert j2 % 2 == 0

                i2 = i2 // 2
                j2 = j2 // 2

                i1m = i1 // 2
                j1m = j1 // 2

                i1p = i1 // 2 + 1
                j1p = j1 // 2 + 1

                return quoridor_pos(i1m, i2) + '-' + quoridor_pos(i1p, i2) + '-' + quoridor_pos(j1m, j2) + '-' + quoridor_pos(j1p, j2)"""