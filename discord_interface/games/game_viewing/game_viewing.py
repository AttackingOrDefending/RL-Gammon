import json

from discord_interface.games.game_viewing.interface.Interface_hex_dyna import Interface_hex_dyna
from discord_interface.games.game_viewing.interface.visualisation_backgammon import interface_backgammon
from discord_interface.games.games_enum import Discord_Game
from discord_interface.games.game_viewing.interface.Interface_square_amazons import Interface_square_amazons
from discord_interface.games.game_viewing.interface.Interface_square_deplacement import Interface_square_deplacement

import os



def more_recent_file(dossier: str) -> str | None:
    """
    Retourne le chemin complet du fichier le plus récent dans un dossier.
    Si le dossier est vide ou ne contient pas de fichiers, retourne None.
    """
    # Liste les fichiers seulement (pas les sous-dossiers)
    #print()
    fichiers = [os.path.join(dossier, f) for f in os.listdir(dossier)
                if os.path.isfile(os.path.join(dossier, f)) and f != '.DS_Store']

    if not fichiers:
        return None

    # Cherche le fichier avec le temps de modification le plus récent
    return max(fichiers, key=os.path.getmtime)

def game_construction(log_file):
    _json_file = more_recent_file(log_file)

    with (open(_json_file, 'r') as file):
        _bot_ref_log = json.load(file)
        #print(_bot_ref_log)


        histo = _bot_ref_log['moves']

        game_name = _bot_ref_log['game_name']
        #print('game_name:', game_name)
        #print('H:',histo)
        game = Discord_Game(game_name).__class__()
        #histo = [game.string_to_action(m) for m in histo]

        for a in histo:
            #print(game.action_to_string(a))
            #print(len(game.jeu.historique), game.valid_actions())
            #assert a in game.valid_actions()
            assert a in game.textual_legal_moves()
            #game.plays(a)
            game.textual_plays(a)


    return game_name, game



def go_interface(log_file):
    global game_name
    game_name, game = game_construction(log_file)

    def lire_fichier_et_maj():
        global game_name

        old_game_name = game_name
        game_name, game = game_construction(log_file)

        if old_game_name != game_name:
            interface.quit()
            # raise GameChange()

        interface.game = game

        #print(len(jeu.historique))

        interface.actualisation_graphique(interface.canvas)

        # Relancer la lecture dans 1000 ms (1 seconde)
        interface.after(1000, lire_fichier_et_maj)


    while True:
        try:
            if game_name == 'clobber' or game_name == 'breakthrough':
                interface = Interface_square_deplacement(game, ia_noir=None, ia_blanc=None, permu_couleur=False, affichage_valeurs=False)
            elif game_name == 'amazons':
                interface = Interface_square_amazons(game, ia_noir=None, ia_blanc=None, permu_couleur=False, affichage_valeurs=False)
            elif 'hex' in game_name or 'havannah' in game_name:
                interface = Interface_hex_dyna(game, ia_noir=None, ia_blanc=None, permu_couleur=False, affichage_valeurs=False, affiche_coup_licites=False)
            elif 'backgammon' == game_name:
                interface = interface_backgammon(game)
            # Démarrer la boucle de lecture/maj
            lire_fichier_et_maj()

            interface.lancer()

            break

        except Exception:
            import traceback
            traceback.print_exc()
            """"""
