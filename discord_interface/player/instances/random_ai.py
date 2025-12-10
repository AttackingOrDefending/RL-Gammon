import random
from discord_interface.player.model.player import *
from discord_interface.utils.mytime import Time

class RandomAI(Player):

    def __init__(self, game:Game=None):
        super().__init__(game=game)

    def opponent_plays(self, action):
        #print('o:',action)
        self.game.plays(action)



    def replays(self, action):

        #print('sr:',action)
        self.game.plays(action)



    def plays(self, time_left: Time=None, opponent_time_left: Time=None):

        action = self.best_move(self.game)



        self.game.plays(action)

        #print('s:',action)
        return action


    def best_move(self, jeu):


        if jeu.get_current_player() == 1:
            joueur = 1
            adv = 0
        else:
            joueur = 0
            adv = 1

        gagnants = []
        perdants = []
        nuls = []
        autres = []



        for coup in jeu.valid_actions():
            jeu.plays(coup)

            if jeu.ended():
                if jeu.winner == joueur:
                    gagnants.append(coup)
                elif jeu.winner == adv:
                    perdants.append(coup)
                else:
                    nuls.append(coup)
            else:
                autres.append(coup)

            jeu.undo()

        if gagnants:
            return random.choice(gagnants)
        if autres:
            return random.choice(autres)
        if nuls:
            return random.choice(nuls)
        if perdants:
            return random.choice(perdants)