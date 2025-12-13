import os

from discord_interface.games.game_viewing.game_viewing import go_interface


if __name__ == '__main__':
    log_file = os.path.dirname(os.path.abspath(__file__))+'/../../log/bot_ref_log/'#'../../log/bot_ref_log/'
    go_interface(log_file)