from discord_interface.games.translator.quentin_games.Shobu import Shobu


class Shobu_hiera_passif(Shobu):

    def hiera_agressif(self):
        return 0

    def coupsLicites(self):

        if self.phase == 0:
            return list(set([(cp[self.hiera_agressif()], None) for cp in self.coups_licites]))
        else:
            return [(cp[1-self.hiera_agressif()], None) for cp in self.coups_licites if cp[self.hiera_agressif()] == self.historique[-1][0][0]]



    def hash_plateau(self):
        return self.plateau[:,:,:7].tobytes()

    def actu_couleur_plateau(self, ):
        super().actu_couleur_plateau()

        assert self.plateau[0, 0, 4] != 0

        panel_hiera = -self.codage_regle_50c()-self.codage_repetitions()-1

        self.plateau[:,:,panel_hiera] = self.phase
        #print(self.plateau.shape[2],self.plateau.shape[2]+panel_hiera)
        #assert panel_hiera != 4 and panel_hiera-1 != 4 and panel_hiera -2 != 4

        if self.phase:
            #print('>>>',self.historique[-1][0][0])
            z, (i, j), (k, l), *_ = self.historique[-1][0][0]
            self.plateau[i, j, panel_hiera-2] = 1
            self.plateau[k, l, panel_hiera-1] = 1

            assert self.plateau[0, 0, 4] != 0
        else:
            self.plateau[:, :, panel_hiera-2:panel_hiera] = 0
            #print(self.plateau.shape[2]+panel_hiera-2, self.plateau.shape[2]+panel_hiera)
            #print(self.plateau.shape[2],  panel_hiera - 2, panel_hiera)
            #assert self.plateau[0, 0, 4] != 0


    def get_panel(self):
        return super().get_panel()+3

    def init(self):
        self.phase = 0

        self.est_jeu_biphase = True

        super().init()

    def jouer(self, a, b):

        if self.phase == 0:
            if self.use_repetitions():
                self.hash_old_positions.append(self.hash_plateau())

        coup = a

        self.historique.append(((a,b), self.blancJoue, list(self.coups_licites), self.tour, self.tour_sans_prise, self.nb_repet, self.phase))

        self.application_coup_elementaire(coup, self.phase)

        if self.phase == 1:
            self.blancJoue = not self.blancJoue

        self.phase = 1 - self.phase

        self.actu_couleur_plateau()


        if self.phase == 0:

            self.tour += 1

            self.nb_repet = self.hash_old_positions.count(self.hash_plateau())
            self.actu_codage_extras()

            self.test_fini_A()

            self.calcul_coups_licites()

            self.test_fini_B()

            if self.fini:
                self.calcul_score()

        else:
            self.actu_codage_extras()
            assert self.coups_licites
            assert self.coupsLicites()


    def undo(self, ):

        assert self.plateau[0, 0, 4] != 0
        coup, self.blancJoue, self.coups_licites, self.tour, self.tour_sans_prise, self.nb_repet, self.phase = self.historique.pop()

        assert self.plateau[0, 0, 4] != 0
        if self.phase == 0:
            if self.hash_old_positions:
                self.hash_old_positions.pop()

        assert self.plateau[0, 0, 4] != 0
        self.unapplication_coup_elementaire(coup[0])

        assert self.plateau[0, 0, 4] != 0
        self.actu_couleur_plateau()
        assert self.plateau[0, 0, 4] != 0
        self.actu_codage_extras()

        assert self.plateau[0, 0, 4] != 0
        self.fini = False

        self.gagnant = None
        assert self.plateau[0, 0, 4] != 0

    def fusionner_coups_phasiques(self, cp1, cp2):
        return cp1[0], cp2[0]

    """def get_coup_complet(self, coup_phasique):
        assert coup_phasique[1] is None
        assert self.phase == 1
        return (self.historique[-1][0][0], coup_phasique[0])"""

    def get_coup_phasique(self, cp):
        if self.phase == 0:
            return cp[0]
        else:
            return cp[1]


    def get_coups_phasiques(self, cp_complet):
        return ((cp_complet[0], None),(cp_complet[1], None))


    def est_tour_hierarchique(self):
        return self.phase == 0
