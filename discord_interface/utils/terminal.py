
def print_clef_valeur_evidence(clef, valeur):
    print(red(clef+' :'), gras(red(valeur)))

def textify(list):
    if isinstance(list, str):
        return list
    else:
        list = [str(e) for e in list]
        return ' '.join(list)

def blue(*txt):
    txt = textify(txt)
    return '\033[94m'+txt+'\033[0m'


def cyan(*txt):
    txt = textify(txt)
    return '\033[96m'+txt+'\033[0m'

def green(*txt):
    txt = textify(txt)
    return '\033[92m'+txt+'\033[0m'

def red(*txt):
    txt = textify(txt)
    return '\033[91m'+txt+'\033[0m'

def fushia(*txt):
    txt = textify(txt)
    return '\033[95m'+txt+'\033[0m'


def orange(*txt):
    txt = textify(txt)
    return '\033[93m'+txt+'\033[0m'


def gras(*txt):
    txt = textify(txt)
    return '\033[1m'+txt+'\033[0m'


def italique(*txt):
    txt = textify(txt)
    return '\033[3m'+txt+'\033[0m'

def souligner(*txt):
    txt = textify(txt)
    return '\033[4m'+txt+'\033[0m'

def inverser(*txt):
    txt = textify(txt)
    return '\033[;7m' + txt + '\033[0m'

def gris(*txt):
    txt = textify(txt)
    return '\033[0;37m'+txt+'\033[0m'



def traitement_pourcentage(n, retrait_debut=0, precision=2, taille = 6):
    demi_taille = taille // 2
    txt = str(int(round(n*100*10**precision))/(10**precision))+'%'
    txt = txt.replace('.0%','%')
    if '.' in txt:
        d, f = txt.split('.')
        d = len(d)
        f = len(f)
    else:
        d = len(txt)
        f = -1
    full_texte = ' '*(demi_taille-d-retrait_debut)+txt+' '*(demi_taille+(taille % 2)-f)
    if len(full_texte) == taille:
        return full_texte
    elif len(full_texte) == taille+1:
        if full_texte[0] == ' ':
            return full_texte[1:]
        else:
            if full_texte[-1] == ' ':
                return full_texte[:-1]
            else:
                return full_texte

    elif len(full_texte) == taille+2:
        if full_texte[0] == ' ' and full_texte[-1] == ' ':
            return full_texte[1:-1]
        else:
            return full_texte
    elif len(full_texte) > taille:
        return full_texte
    else:
        print(full_texte, len(full_texte), taille)
        raise Exception('trautement pourcentage ???')


"""Black        0;30     Dark Gray     1;30
Red          0;31     Light Red     1;31
Green        0;32     Light Green   1;32
Brown/Orange 0;33     Yellow        1;33
Blue         0;34     Light Blue    1;34
Purple       0;35     Light Purple  1;35
Cyan         0;36     Light Cyan    1;36
Light Gray   0;37     White         1;37"""
