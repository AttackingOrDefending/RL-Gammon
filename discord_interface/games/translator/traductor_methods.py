from math import ceil
from random import choice, randint
from statistics import mean
from time import time

import numpy as np


def troncature(x):
    return int(100*x)/100

def correspondance_nom_jeu_ludii_python(nom_jeu):
    if nom_jeu == 'Reversi':
        return 'othello'
    elif nom_jeu == 'International Draughts':
        return 'dames'
    elif nom_jeu == 'Brazilian Draughts':
        return 'dames-bresiliennes'
    elif nom_jeu == 'Canadian Draughts':
        return 'dames-canadiennes'
    elif  nom_jeu == 'Hex':
        return 'hex-swap'
    elif  nom_jeu == 'Havannah':
        return 'havannah-swap'
    elif  nom_jeu == 'Gomoku':
        return 'outer-open-gomoku'
    elif nom_jeu == 'Chinese Checkers':
        return 'dames-chinoises'
    else:
        return nom_jeu.replace('_','-').replace(' ','-').lower()

def correspondance_nom_jeu_python_ludii(nom_jeu):

    if 'othello' == nom_jeu:
        return 'Reversi'
    elif 'amazons'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'outer-open-gomoku'  == nom_jeu:
        return 'Outer_Open_Gomoku'
    elif 'hex'  == nom_jeu:
        return 'Hex Nash'
    elif 'hex-swap' == nom_jeu:
        return 'Hex'
    elif 'breakthrough'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'clobber'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'connect6'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'contagion'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'lines-of-action' == nom_jeu:
        return 'Lines of Action'
    elif 'surakarta'  in nom_jeu:
        return 'Surakarta'
    elif 'santorini' in nom_jeu:
        return  'Santorini'
    elif 'xiangqi'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'shogi' == nom_jeu:
        return nom_jeu.capitalize()
    elif 'minishogi' == nom_jeu:
        return nom_jeu.capitalize()
    elif 'kyoto-shogi' == nom_jeu:
        return 'Kyoto Shogi'
    elif 'ataxx'  == nom_jeu:
        return nom_jeu.capitalize()
    elif 'havannah' == nom_jeu:
        return nom_jeu.capitalize()
    elif 'havannah-swap' == nom_jeu:
        return nom_jeu.replace('-swap','').capitalize()
    elif 'echec' in nom_jeu:
        return 'Chess'
    elif 'bresiliennes' in nom_jeu:
        return 'Brazilian Draughts'
    elif 'canadiennes' in nom_jeu:
        return 'Canadian Draughts'
    elif 'dames' in nom_jeu or 'draughts' in nom_jeu:
        return 'International Draughts'
    elif 'shobu' in nom_jeu:
        return 'Shobu'
    elif 'tron' in nom_jeu:
        return 'Tron'
    elif 'quoridor' in nom_jeu:
        return 'Quoridor'




correspondance = {0 : 'A', 1 : 'B', 2 : 'C', 3 :'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I', 9 : 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R', 18 : 'S'}

anti_correspondance = {'C': 2, 'H': 7, 'S': 18, 'J': 9, 'E': 4, 'F': 5, 'O': 14, 'B': 1, 'R': 17, 'K': 10, 'I': 8, 'G': 6, 'L': 11, 'N': 13, 'Q': 16, 'P': 15, 'D': 3, 'A': 0, 'M': 12}

def reorientation_inverse(nom_jeu, jeu, cp):
    if 'bord' in jeu.__class__.__name__ and ('_codebord' not in jeu.__class__.__name__ or jeu.__class__.__name__.count('bord') > 1):  # if 'bord' in nom_jeu:
        chiffre, lettre = jeu.taille_interieur - 1 - cp[0], cp[1]
    else:
        chiffre, lettre = jeu.taille - 1 - cp[0], cp[1]

    return str(lettre) + ',' + str(chiffre)

def correspondance_action_ludii_ludii_number(nom_jeu, jeu, coup):

    if '-' in coup:

        cp1, cp2 = coup.split('-')

        cp1=reorientation_bis(nom_jeu, jeu, cp1)
        cp2=reorientation_bis(nom_jeu, jeu, cp2)


        return reorientation_inverse(nom_jeu, jeu, cp1)+'-'+reorientation_inverse(nom_jeu, jeu, cp2)
    else:
        raise
        #return correction_si_shobu(nom_jeu, jeu, reorientation_bis(nom_jeu, jeu, coup))

def correspondance_action_python_ludii(jeu, Jeu, coup):
    #print(jeu, coup)
    """f = open('test','w')
    f.write(str(coup))
    f.write(Jeu.__class__.__name__)
    f.close()"""

    i,j = coup

    """if blanc_joue:
        player = '2'
    else:
        player = '1'"""

    if i == 'swap':
        return 'swap'

    if i == 'notswap':
        return 'notswap'


    if 'swap' in jeu:  # Jeu.__class__.__name__:#
        if (Jeu.swapped == (i, j) and (not 'bord' in Jeu.__class__.__name__  or 'hiera' in Jeu.__class__.__name__)) or (Jeu.swapped == (i + 1, j + 1) and 'bord' in Jeu.__class__.__name__  and not 'hiera'  in Jeu.__class__.__name__) :
            # assert Jeu_ludii.codage_coups_complet
            # return Jeu_ludii.codage_coup('[[Swap:player1=1,player2=2,decision=true], [SetNextPlayer:player=2]]')
            # return 'swap'
            #print('swap')
            # print(Jeu_ludii.coupsLicites_actuel)
            return 'swap'

    if 'othello' in jeu:
        return correspondance[j] + str(Jeu.taille - i)
    elif 'dames-chinoises' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[4+j1] + str(17-i1) + '-' + correspondance[4+j2] + str(17-i2)
    elif 'amazons' in jeu:
        if isinstance(i, tuple):
            i1, j1 = i
            i2, j2 = j
            return correspondance[j1] + str(i1 + 1) + '-' + correspondance[j2] + str(i2 + 1)
        else:
            return correspondance[j] + str(i + 1)
    elif 'gomoku' in jeu:
        return correspondance[j] + str(Jeu.taille - i)

    elif 'hex' in jeu:
        #return correspondance[j] + str(Jeu.taille - i)
        if 'bord' in Jeu.__class__.__name__:#if 'bord' in jeu:
            return correspondance[j] + str(Jeu.taille_interieur - i)
        else:
            return correspondance[j]+ str(Jeu.taille -i)
        # return '[Add:type=Cell,to=0,what=1,decision=true]' # [Add:type=Cell,to=116,what=2,decision=true]
    elif 'breakthrough' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
    elif 'clobber' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(i1 + 1) + '-' + correspondance[j2] + str(i2 + 1)
    elif 'connect6' in jeu:
        return correspondance[j] + str(Jeu.taille - i)
    elif 'contagion' in jeu:
        return None
    elif 'lines-of-action' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
    elif 'surakarta' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
    elif 'dames' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
    elif 'santorini' in jeu:
        if isinstance(i, tuple):
            i1, j1 = i
            i2, j2 = j
            return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
        else:
            return correspondance[j] + str(Jeu.taille - i)

    elif 'xiangqi' in jeu or 'chinese-chess' in jeu or 'chinese_chess' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.longueur - i1) + '-' + correspondance[j2] + str(Jeu.longueur - i2)
    elif 'shogi' in jeu:
        return
    elif 'ataxx' in jeu:
        i1, j1 = i
        i2, j2 = j
        return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
    elif 'havannah' in jeu:
        return correspondance[j] + str(Jeu.taille - i)
    elif 'shobu' in jeu and j is not None:#'hiera' not in Jeu.__class__.__name__:

        z1, (a1, b1), (a2, b2,) = i
        z2, (a1b, b1b), (a2b, b2b,), *_ = j

        if z1 == 0:
            i1 = a1
            j1 = b1
        elif z1 == 1:
            j1 = Jeu.taille + b1 #+ 1
            i1 = a1
        elif z1 == 2:
            i1 = Jeu.taille + a1 #+ 1
            j1 = b1
        elif z1 == 3:
            i1 =Jeu.taille + a1 #+ 1
            j1 = Jeu.taille + b1 #+ 1

        if z1 == 0:
            i2 = a2
            j2 = b2
        elif z1 == 1:
            j2 = Jeu.taille + b2 #+ 1
            i2 = a2
        elif z1 == 2:
            i2 = Jeu.taille + a2 #+ 1
            j2 = b2
        elif z1 == 3:
            i2 =Jeu.taille + a2 #+ 1
            j2 = Jeu.taille + b2 #+ 1

        if z2 == 0:
            i1b = a1b
            j1b = b1b
        elif z2 == 1:
            j1b = Jeu.taille + b1b  # + 1
            i1b = a1b
        elif z2== 2:
            i1b = Jeu.taille + a1b  # + 1
            j1b = b1b
        elif z2 == 3:
            i1b = Jeu.taille + a1b  # + 1
            j1b = Jeu.taille + b1b  # + 1

        if z2 == 0:
            i2b = a2b
            j2b = b2b
        elif z2 == 1:
            j2b = Jeu.taille + b2b  # + 1
            i2b = a2b
        elif z2 == 2:
            i2b = Jeu.taille + a2b  # + 1
            j2b = b2b
        elif z2 == 3:
            i2b = Jeu.taille + a2b  # + 1
            j2b = Jeu.taille + b2b  # + 1

        #return correspondance[j1] + str(2*Jeu.taille - i1) + '-' + correspondance[j2] + str(2*Jeu.taille - i2)+'>'+correspondance[j1b] + str(2*Jeu.taille - i1b) + '-' + correspondance[j2b] + str(2*Jeu.taille - i2b)
        return correspondance[j1] + str(2 * Jeu.taille - i1) + '-' + correspondance[j2] + str(2 * Jeu.taille - i2), correspondance[j1b] + str(2 * Jeu.taille - i1b) + '-' + correspondance[
        j2b] + str(2 * Jeu.taille - i2b)

    elif 'shobu' in jeu and j is None:# and 'hiera' in Jeu.__class__.__name__:

        z, (a1, b1), (a2, b2), *_ = i

        if z == 0:
            i1 = a1
            j1 = b1
        elif z == 1:
            j1 = Jeu.taille + b1  # + 1
            i1 = a1
        elif z == 2:
            i1 = Jeu.taille + a1  # + 1
            j1 = b1
        elif z == 3:
            i1 = Jeu.taille + a1  # + 1
            j1 = Jeu.taille + b1  # + 1

        if z == 0:
            i2 = a2
            j2 = b2
        elif z == 1:
            j2 = Jeu.taille + b2  # + 1
            i2 = a2
        elif z == 2:
            i2 = Jeu.taille + a2  # + 1
            j2 = b2
        elif z == 3:
            i2 = Jeu.taille + a2  # + 1
            j2 = Jeu.taille + b2  # + 1

        return correspondance[j1] + str(2*Jeu.taille - i1) + '-' + correspondance[j2] + str(2*Jeu.taille - i2)
    elif 'tron' in jeu:
        return correspondance[j]+ str(Jeu.taille -i)
    elif 'arimaa' in jeu:
        if i == None:
            return 'pass'
        elif isinstance(i, int):

            corresp_tour_zone_piece = {
                0: 64,
                1: 64,
                2: 64,
                3: 64,
                4: 64,
                5: 64,
                6: 64,
                7: 64,
                8: 65,
                9: 65,
                10: 66,
                11: 66,
                12: 67,
                13: 67,
                14: 68,
                15: 69,
                16: 70,
                17: 70,
                18: 70,
                19: 70,
                20: 70,
                21: 70,
                22: 70,
                23: 70,
                24: 71,
                25: 71,
                26: 72,
                27: 72,
                28: 73,
                29: 73,
                30: 74,
                31: 75,
            }

            return correspondance[j] + str(Jeu.taille - i)
            if Jeu.historique and (i,j) == Jeu.historique[-1][:2]:
                return str(corresp_tour_zone_piece[Jeu.tour - 1]) + '-' + correspondance[j] + str(Jeu.taille - i)
            else:
                return str(corresp_tour_zone_piece[Jeu.tour])+'-'+correspondance[j] + str(Jeu.taille - i)
        else:
            i1, j1 = i
            i2, j2 = j

            if isinstance(i1, int) and isinstance(i2, int):
                return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)
            elif isinstance(i1, int):
                i2a, j2a = i2
                i2b, j2b = j2

                """return (correspondance[j2a] + str(Jeu.taille - i2a) + '-' + correspondance[j2b] + str(Jeu.taille - i2b),
                        correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2a] + str(Jeu.taille - i2a),
                        )"""
                return correspondance[j2a] + str(Jeu.taille - i2a) + '-' + correspondance[j2b] + str(Jeu.taille - i2b) +'-'+ correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2a] + str(Jeu.taille - i2a)
            else:
                i1a, j1a = i1
                i1b, j1b = j1

                #return (correspondance[j1a] + str(Jeu.taille - i1a) + '-' + correspondance[j1b] + str(Jeu.taille - i1b), correspondance[j2] + str(Jeu.taille - i2), )#correspondance[j1b] + str(Jeu.taille - i1b) + '-' +
                return correspondance[j1a] + str(Jeu.taille - i1a) + '-' + correspondance[j1b] + str(Jeu.taille - i1b) + '-' + correspondance[j2] + str(Jeu.taille - i2)

    elif 'quoridor' in jeu:
        (i1, i2), (j1, j2) = i, j

        def quoridor_pos(i, j):
            return correspondance[j] + str(Jeu.taille - i)

        if i1 % 2 == 0 and i2 % 2 == 0:
            #print('?')
            return quoridor_pos(i1//2, i2//2) + '-' + quoridor_pos(j1//2, j2//2)
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

                return quoridor_pos(i1m, i2) + '-' + quoridor_pos(i1p, i2) + '-' + quoridor_pos(j1m, j2) + '-' + quoridor_pos(j1p, j2)

        #raise NotImplementedError()

    elif 'backgammon' == jeu:
        def coord(pos):
            return 'P'+str(pos)

        if isinstance(i, int):
            return str(i)+'-'+str(j)
        elif i is None:
            return 'pass'
        elif isinstance(j, int):# double
            return str(j)+'-'+'-'.join([coord(p) for p in i])
        elif j is None: # simple tronqué
            return coord(i[0])+'-'+str(i[1])
        else: # simple
            return coord(i[0]) + '-' + str(i[1])+'-'+ coord(j[0])+'-'+str(j[1])
    else:
        if isinstance(i, int):
            return correspondance[j] + str(Jeu.taille - i)
        else:
            i1, j1 = i
            i2, j2 = j
            return correspondance[j1] + str(Jeu.taille - i1) + '-' + correspondance[j2] + str(Jeu.taille - i2)

def reorientation(nom_jeu, jeu, coup):
    if coup == 'swap':
        if 'bord' in jeu.__class__.__name__ and not 'hiera' in jeu.__class__.__name__ :
            i, j = jeu.historique[-1][0]
            return i-1, j-1
        else:
            return jeu.historique[-1][0] # si probleme avec swap, c est surement que dans l historiquement le premier element n est pas le coup jouer ou alors ce n est pas une liste

    if ',' in coup:

        lettre, chiffre = coup.split(',')
        lettre = int(lettre)
        chiffre = int(chiffre)

    else:


        lettre = coup[0]
        chiffre = coup[1:]

        if nom_jeu == 'arimaa' and lettre.isdigit():
            return coup

        lettre = anti_correspondance[lettre]#jeu.taille - anti_correspondance[lettre]
        chiffre = int(chiffre)#jeu.taille - int(chiffre)

        #chiffre, lettre = lettre, chiffre
        if 'bord' in jeu.__class__.__name__:#if 'bord' in nom_jeu:
            return chiffre - 1, jeu.taille_interieur - lettre
        else:
            return chiffre-1, jeu.taille - lettre # ok einstein en interface textuel

    if 'amazons' in nom_jeu or 'clobber' in nom_jeu:
        return chiffre, lettre
    elif 'xiangqi' in nom_jeu or 'chinese-chess' in nom_jeu or 'chinese_chess' in nom_jeu:
        return jeu.longueur - 1 - chiffre, lettre
    elif 'dames-chinoises' in nom_jeu:

        return 17 - 1 - chiffre, lettre-4#correspondance[4+j1] + str(17-i1) + '-' + correspondance[4+j2] + str(17-i2)
    elif 'shobu' in nom_jeu:# and 'hiera' in jeu.__class__.__name__:

        return 2*jeu.taille - 1 - chiffre, lettre

    else:#if 'othello' in jeu or 'gomoku' in jeu or 'hex' in jeu or 'breakthrough' in jeu or 'connect6' in jeu or 'lines-of-action' in jeu or 'surakarta' in jeu or  'santorini' in jeu or 'ataxx' in jeu or 'havannah' in jeu:
        #return jeu.taille - 1 - chiffre, lettre
        if 'bord' in jeu.__class__.__name__ and ('_codebord' not in jeu.__class__.__name__ or  jeu.__class__.__name__.count('bord') > 1):#if 'bord' in nom_jeu:
            return jeu.taille_interieur - 1 - chiffre, lettre
        else:
            return jeu.taille - 1 - chiffre, lettre



def reorientation_bis(nom_jeu, jeu, coup):
    if coup == 'swap':
        reorientation(nom_jeu, jeu, coup)

    if ',' in coup:

        return reorientation(nom_jeu, jeu, coup)

    else:

        lettre = coup[0]
        chiffre = coup[1:]

        if nom_jeu == 'arimaa' and lettre.isdigit():
            return coup

        lettre = anti_correspondance[lettre]#+1

        chiffre = int(chiffre)-1

        if hasattr(jeu, 'taille_interieur'):
            lettre = jeu.taille_interieur-lettre
            chiffre = jeu.taille_interieur-chiffre
        else:
            lettre = jeu.taille-lettre
            chiffre = jeu.taille-chiffre


        #chiffre, lettre = lettre, chiffre
        if 'bord' in jeu.__class__.__name__:#if 'bord' in nom_jeu:
            return chiffre - 1, jeu.taille_interieur - lettre
        else:
            return chiffre-1, jeu.taille - lettre # ok einstein en interface textuel



def correction_si_shobu(nom_jeu, jeu, coup):

    if 'shobu' in nom_jeu:# and 'hiera' in jeu.__class__.__name__:

        (i,j), (k,l) = coup

        z = 0

        if i >= jeu.taille:
            i = i# - 1
            z = 2

        if j >= jeu.taille:
            j = j# - 1
            z += 1

        i = i % jeu.taille
        j = j % jeu.taille

        if k >= jeu.taille:
            k = k# - 1

        if l >= jeu.taille:
            l = l# - 1

        k = k % jeu.taille
        l = l % jeu.taille
        #print('??', (z, (i, j), (k, l)))
        #f = open('test2', 'w')
        #f.write(str((z, (i, j), (k, l)))+'\n')
        #f.write(str(jeu.coupsLicites())+'\n')

        for cp in jeu.coupsLicites():
            #print(cp[0][:3])

            #f.write(str((z, (i, j), (k, l)) == cp[0][:3])+' '+str(cp[0][:3]) + '\n')
            if (z, (i, j), (k, l)) == cp[0][:3]:
                #print(cp,'?')
                #f.close()
                return cp
        #f.close()
    else:
        return coup

def last_correction(nom_jeu, jeu, coup):

    if 'shobu' in nom_jeu:# and 'hiera' in jeu.__class__.__name__:

        return correction_si_shobu(nom_jeu, jeu, coup)
    elif 'tron' in nom_jeu:
        return coup[1]
    else:
        return coup

def correspondance_action_ludii_python(nom_jeu, jeu, coup):
    if coup == 'pass':
        return (None, None)
    if '-' in coup:

        cp1, cp2 = coup.split('-')
        return last_correction(nom_jeu, jeu, (reorientation(nom_jeu, jeu, cp1), reorientation(nom_jeu, jeu, cp2)))
    else:
        return correction_si_shobu(nom_jeu, jeu, reorientation(nom_jeu, jeu, coup))

def correspondance_action_ludii_python_bis(nom_jeu, jeu, coup):
    if coup == 'pass':
        return (None, None)
    if '-' in coup:

        cp1, cp2 = coup.split('-')
        return last_correction(nom_jeu, jeu, (reorientation_bis(nom_jeu, jeu, cp1), reorientation_bis(nom_jeu, jeu, cp2)))
    else:
        return correction_si_shobu(nom_jeu, jeu, reorientation_bis(nom_jeu, jeu, coup))

def correction(ludii_coup):
    return str(anti_correspondance[ludii_coup[0]]) + ',' + str(int(ludii_coup[1:])-1)

def correspondance_action_inverse(nom_jeu, jeu, coup):
    #print(coup)
    if 'tron' in nom_jeu:
        coup = coup.split('-')[1].split('[')[0].replace(']','')
    else:
        coup = coup.replace('[','').replace(']','')
        if ',' in coup:
            coup = coup.split(',')[1].replace(' ','')
    #print(coup,'!')
    if '-' in coup:
        cp1, cp2 = coup.split('-')
        #print(cp1, cp2,'?')
        return last_correction(nom_jeu, jeu, (reorientation(nom_jeu, jeu, correction(cp1)), reorientation(nom_jeu, jeu, correction(cp2))))
    else:
        return reorientation(nom_jeu, jeu, correction(coup))



def empreinte(nom, plateau):
    if 'amazons' in nom:
        if '-panels' in nom:
            return np.array([plateau[:,:,0], plateau[:,:,1]+plateau[:,:,2]])
        else:
            return np.array([plateau[:,:,0], plateau[:,:,1]])
    else:
        if '-panels' in nom:
            return plateau[:,:,0]+plateau[:,:,1]
        else:
            if plateau.ndim > 2:
                return plateau[:,:,0]
            else:
                return plateau


def affichage_plateau(p):
    if p.ndim == 3:
        for index in p.shape[2]:
            print(p[:,:,index])
    else:
        print(p)
    print()
