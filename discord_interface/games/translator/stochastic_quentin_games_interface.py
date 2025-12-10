from discord_interface.games.translator.quentin_games_interface import InterfaceJeuDiscord


class StochasticInterfaceJeuDiscord(InterfaceJeuDiscord):

    def get_current_player(self):
        return self.jeu.joueur_en_cours


    def get_action_probabilities(self):
        return self.jeu.probabilite_actions