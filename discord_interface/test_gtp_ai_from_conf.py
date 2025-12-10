import asyncio
from random import randint, choice

from discord_interface.games.games_enum import EnumGames
from discord_interface.player.instances.gtp_ai import GTP_AI
from discord_interface.utils.configuration_files import load_configurations
from discord_interface.utils.mytime import Time

import sys
import time

from discord_interface.utils.terminal import traitement_pourcentage


def progress_bar(pct, width=40, text=""):
    """
    Affiche une barre de progression sur une seule ligne dans le terminal.

    Arguments:
    - pct   : float ou int — pourcentage d'avancement (0..100). Les valeurs
              en dehors seront clampées.
    - width : int — largeur en caractères de la barre (par défaut 40).
    - text  : str — texte à afficher après la barre (par ex. 'Téléchargement').
    """
    # normaliser / clamp
    try:
        pct_val = float(pct)
    except (TypeError, ValueError):
        pct_val = 0.0
    if pct_val < 0:
        pct_val = 0.0
    if pct_val > 100:
        pct_val = 100.0

    filled_len = int(round(width * pct_val / 100.0))
    bar = "█" * filled_len + "░" * (width - filled_len)  # caractères visuels
    # chaîne finale — % with one decimal if not integer
    pct_display = f"{pct_val:.1f}%" if (pct_val % 1) else f"{int(pct_val)}%"

    # \r pour retour début de ligne, end="" pour rester sur la même ligne
    sys.stdout.write(f"\r[{bar}] {pct_display} | {text}")
    sys.stdout.flush()

    # quand on atteint 100%, on termine la ligne par un saut de ligne
    if pct_val >= 100.0:
        sys.stdout.write("\n")
        sys.stdout.flush()


async def test(game_name, stats):
    gpt = load_configurations()

    ai = GTP_AI(program_name=gpt['program_name'], program_arguments=gpt['program_arguments'], program_directory=gpt['program_directory'] if gpt['program_directory'] is not None else '')

    await ai.update_game(game_name)

    game = EnumGames.find_game(game_name)

    current_player_ok = True

    wins = 0

    progress_bar(0, width=30, text="")

    for _ in range(stats):

        game.reset()
        await ai.reset()
        is_first = randint(0, 1)

        if is_first:
            ai.player_number = 0
        else:
            ai.player_number = 1

        t = 0


        while not game.ended():

            progress_bar(int((100 * (_ + 1) / stats)), width=30, text="Wins: " + str(traitement_pourcentage(wins / (_ + 1))) + 'running match' + '.' * randint(1, 3))

            #print(game.get_current_player(), await ai.get_current_player())
            try:
                if not game.get_current_player() == await ai.get_current_player():
                    current_player_ok = False
            except:
                current_player_ok = False

            if is_first and game.get_current_player() == 0 or not is_first and game.get_current_player() == 1:

                tps = Time(second=max(1, 100 - t))

                cp = await ai.plays(tps, tps)

                if randint(0, 10) == 0:
                    await ai.undo()
                    cp = await ai.plays(tps, tps)
                #print('GTP:',"'"+cp+"'")
                
                cp = game.action_to_string(cp)
            else:

                cp = choice(game.textual_legal_moves())

                await ai.opponent_plays(game.string_to_action(cp))
                #print('R:', "'" + cp + "'")

            game.textual_plays(cp)

            t += 1

        if is_first and game.winner == 0 or not is_first and game.winner == 1:
            wins += 1

        progress_bar(int((100 * (_+1) / stats)), width=30, text="Wins: " + str(traitement_pourcentage(wins / (_ + 1)))+ 'running match'+'.'*randint(1,3))

    progress_bar(int((100 * (_ + 1) / stats)), width=30, text="Wins: " + str(traitement_pourcentage(wins / (_ + 1))))

    if current_player_ok:
        print('"player" command is ok.')
    else:
        print('"player" command is NOT ok.')

    print('Success!')
    if wins / stats < 0.98:
        print('But suspect win rate against random AI: ', traitement_pourcentage(wins / stats))


if '__main__' == __name__:

    game_name = input('Game name: ').lower()
    stats = int(input('Number of matchs for the test (100-300 recommanded): '))

    asyncio.run(test(game_name, stats))
