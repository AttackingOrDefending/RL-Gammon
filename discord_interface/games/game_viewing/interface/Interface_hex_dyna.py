import tkinter.font as tkFont
from math import *
from tkinter import *



class Interface_hex_dyna(Tk):

    def __init__(self, hex, hauteur = 1000, largeur = 1500, ia_noir = None, ia_blanc = None, affichage_valeurs=False, titre=None, permu_couleur= False, permu_plateau = False, resize_txt = 1, affiche_coup_licites=True):
        self.affiche_coup_licites = affiche_coup_licites

        Tk.__init__(self)

        if titre:
            self.title("Hex - "+titre)
        else:
            self.title("Hex")


        menubar = Menu(self)
        menu = Menu(menubar)
        menubar.add_cascade(label='Actions', menu=menu)
        menu.add_command(label="Relancer", command=self.relancer)

        # display the menu
        self.config(menu=menubar)

        self.permu_couleur = permu_couleur
        self.permu_plateau = permu_plateau


        if not permu_couleur:
            self.couleur_pion_premier = "black"
            self.couleur_pion_second = "white"

            if not permu_plateau:
                self.couleur_bord_horizontal = "white"
                self.couleur_bord_vertical = "black"
            else:
                self.couleur_bord_horizontal = "black"
                self.couleur_bord_vertical = "white"
        else:
            self.couleur_pion_premier = "white"
            self.couleur_pion_second = "black"

            if not permu_plateau:
                self.couleur_bord_horizontal = "black"
                self.couleur_bord_vertical = "white"
            else:
                self.couleur_bord_horizontal = "white"
                self.couleur_bord_vertical = "black"




        self.affichage_valeurs = affichage_valeurs

        self.blancHumain = ia_blanc == None
        self.noirHumain = ia_noir == None

        self.ia_noir = ia_noir
        self.ia_blanc = ia_blanc


        self.hex = hex

        self.canvas = Canvas(self, width=largeur, height=hauteur, background='grey90')

        self.hexs = {}

        self.hexs_inv = {}

        self.hexs_centre = {}

        self.hexs_inv_centre = {}

        self.ordre_valeurs = {}
        self.valeurs = {}

        self.largeur = largeur
        self.hauteur = hauteur


        S = hauteur / (hex.taille + 1)

        R = 2 * S / 3
        W = 2 * R
        H = (int)(R * sqrt(3))

        self.r = (int)(H * 5.0 / 16)

        self.dernier_blanc = 0
        self.dernier_noir = 0

        bordX = {}
        bordY = {}

        for i in range(0, 2):
            for j in range(0, 2):
                X = j * (hex.taille - 1) * H + i * (hex.taille - 1) * H / 2
                Y = i * (hex.taille - 1) * S

                # Correctif
                X += H / 4
                Y += 5 * R / 6
                decalage = (largeur - 3 * H * hex.taille / 2) / 2

                bordX[i, j] = decalage + X + H / 2
                bordY[i, j] = Y + R

        coef = 2

        dec_anti_X = (bordX[1, 1] - bordX[0, 0]) / (hex.taille - 1) / coef
        dec_anti_Y = (bordY[1, 1] - bordY[0, 0]) / (hex.taille - 1) / coef

        dec_X = (bordX[0, 1] - bordX[1, 0]) / (hex.taille - 1) / coef
        dec_Y = (bordY[1, 0] - bordY[0, 1]) / (hex.taille - 1) / coef

        self.canvas.create_polygon(bordX[0, 0], bordY[0, 0], bordX[0, 0] - dec_anti_X, bordY[0, 0] - dec_anti_Y,
                              bordX[0, 1] + dec_X, bordY[0, 1] - dec_Y, bordX[0, 1], bordY[0, 1], fill=self.couleur_bord_horizontal)

        self.canvas.create_polygon(bordX[1, 1], bordY[1, 1], bordX[1, 1] + dec_anti_X, bordY[1, 1] + dec_anti_Y,
                              bordX[0, 1] + dec_X, bordY[0, 1] - dec_Y, bordX[0, 1], bordY[0, 1], fill=self.couleur_bord_vertical)

        self.canvas.create_polygon(bordX[1, 0], bordY[1, 0], bordX[1, 0] - dec_X, bordY[1, 0] + dec_Y,
                              bordX[1, 1] + dec_anti_X, bordY[1, 1] + dec_anti_Y, bordX[1, 1], bordY[1, 1],
                              fill=self.couleur_bord_horizontal)

        self.canvas.create_polygon(bordX[0, 0], bordY[0, 0], bordX[0, 0] - dec_anti_X, bordY[0, 0] - dec_anti_Y,
                              bordX[1, 0] - dec_X, bordY[1, 0] + dec_Y, bordX[1, 0], bordY[1, 0], fill=self.couleur_bord_vertical)

        self.id_pions = []
        if not permu_plateau:
            for i in range(0, hex.taille):
                for j in range(0, hex.taille):
                    X = j * H + i * H / 2
                    Y = i * S

                    # Correctif
                    X += H / 4
                    Y += 5 * R / 6
                    decalage = (largeur - 3 * H * hex.taille / 2) / 2

                    p1 = (decalage + X, Y + W - S)
                    p2 = (decalage + X + H / 2, Y)
                    p3 = (decalage + X + H, Y + W - S)
                    p4 = (decalage + X + H, Y + S)
                    p5 = (decalage + X + H / 2, Y + W)
                    p6 = (decalage + X, Y + S)

                    centreX = decalage + X + H / 2
                    centreY = Y + R

                    self.hexs[i, j] = self.canvas.create_polygon(*(p1 + p2 + p3 + p4 + p5 + p6), outline="black", fill="grey",
                                                       activefill="orange", activeoutline="red", tags="hex")
                    self.hexs_inv[self.hexs[i, j]] = (i, j)
                    self.hexs_inv_centre[self.hexs[i, j]] = (centreX, centreY)
                    self.hexs_centre[i, j] = (centreX, centreY)
                    self.ordre_valeurs[i, j] = self.canvas.create_text(centreX, centreY, fill="purple", font=tkFont.Font(size=int(20 * resize_txt)))
                    self.valeurs[i, j] = self.canvas.create_text(centreX, centreY+15, fill="purple", font=tkFont.Font(size=int(15 * resize_txt)))
        else:
            for i in range(0, hex.taille):
                for j in range(0, hex.taille):
                    X = i * H + j * H / 2
                    Y = j * S

                    # Correctif
                    X += H / 4
                    Y += 5 * R / 6
                    decalage = (largeur - 3 * H * hex.taille / 2) / 2

                    p1 = (decalage + X, Y + W - S)
                    p2 = (decalage + X + H / 2, Y)
                    p3 = (decalage + X + H, Y + W - S)
                    p4 = (decalage + X + H, Y + S)
                    p5 = (decalage + X + H / 2, Y + W)
                    p6 = (decalage + X, Y + S)

                    centreX = decalage + X + H / 2
                    centreY = Y + R

                    self.hexs[i, j] = self.canvas.create_polygon(*(p1 + p2 + p3 + p4 + p5 + p6), outline="black",
                                                                 fill="grey",
                                                                 activefill="orange", activeoutline="red", tags="hex")
                    self.hexs_inv[self.hexs[i, j]] = (i, j)
                    self.hexs_inv_centre[self.hexs[i, j]] = (centreX, centreY)
                    self.hexs_centre[i, j] = (centreX, centreY)
                    self.ordre_valeurs[i, j] = self.canvas.create_text(centreX, centreY, fill="purple",
                                                                       font=tkFont.Font(size=int(20 * resize_txt)))
                    self.valeurs[i, j] = self.canvas.create_text(centreX, centreY + 15, fill="purple",
                                                                 font=tkFont.Font(size=int(15 * resize_txt)))

        self.canvas.bind("<Button-1>", self.clic)
        self.bind("<Key>", self.clavier)
        self.bind('<Control-z>', self.undo)
        self.bind('<Command-z>', self.undo)

        self.pieces = []
        self.banderole = []

        self.actualisation_graphique(self.canvas)

        if not self.noirHumain:
            self.jouer_noir()




    def jouer_noir(self):

        self.raz_txt()

        if self.affichage_valeurs:
            (i, j), dic = self.ia_noir.meilleur_coup_valeurs(self.hex)

        else:
            (i, j) = self.ia_noir.meilleur_coup(self.hex)

        #print('>', self.hex.blancJoue, i, j)
        self.hex.jouer(i, j)

        self.actualisation_graphique(self.canvas)
        """"(x, y) = self.hexs_centre[i, j]

        self.dernier_noir = self.canvas.create_oval(x - self.r, y - self.r, x + self.r, y + self.r, fill=self.couleur_pion_premier)
        self.pieces.append(self.dernier_noir)"""

        if self.affichage_valeurs:
            try:
                i = 0
                for coup, v in sorted(dic.items(), key=lambda t: -t[1]):
                    i += 1

                    id_v = self.valeurs[coup]
                    id_o = self.ordre_valeurs[coup]
                    self.canvas.tag_raise(id_v)
                    self.canvas.tag_raise(id_o)
                    self.canvas.itemconfigure(id_v, text=formatage(v))
                    self.canvas.itemconfigure(id_o, text=str(i))
            except:
                i = 0
                for coup, v in sorted(dic.items(), key=lambda t: (-t[1][0], -t[1][1])):
                    i += 1

                    id_v = self.valeurs[coup]
                    id_o = self.ordre_valeurs[coup]
                    self.canvas.tag_raise(id_v)
                    self.canvas.tag_raise(id_o)
                    self.canvas.itemconfigure(id_v, text=formatage(v[0])+' '+formatage(v[1]))
                    self.canvas.itemconfigure(id_o, text=str(i))



    def jouer_blanc(self):

        self.raz_txt()

        if self.affichage_valeurs:
            (i, j), dic = self.ia_blanc.meilleur_coup_valeurs(self.hex)
            #print(dic)

        else:
            (i, j) = self.ia_blanc.meilleur_coup(self.hex)

        print('>', self.hex.blancJoue, i, j)
        self.hex.jouer(i, j)

        self.actualisation_graphique(self.canvas)
        """(x, y) = self.hexs_centre[i, j]

        self.dernier_blanc = self.canvas.create_oval(x - self.r, y - self.r, x + self.r, y + self.r, fill=self.couleur_pion_second)
        self.pieces.append(self.dernier_blanc)"""

        if self.affichage_valeurs:

            i = 0
            for coup, v in sorted(dic.items(), key=lambda t: -t[1]):
                i += 1

                id_v = self.valeurs[coup]
                id_o = self.ordre_valeurs[coup]
                self.canvas.tag_raise(id_v)
                self.canvas.tag_raise(id_o)
                self.canvas.itemconfigure(id_v, text=formatage(v))
                self.canvas.itemconfigure(id_o, text=str(i))


    def raz_txt(self):

        for (i,j), id in self.valeurs.items() :
            self.canvas.itemconfigure(id, text="")

        for (i,j), id in self.ordre_valeurs.items() :
            self.canvas.itemconfigure(id, text="")

    def listing(self, l):
        if isinstance(l, list):
            return l
        else:
            return l.tolist()

    def actualisation_graphique(self, widget):

        k = 4

        for id in self.id_pions:
            widget.delete(id)

        coups = self.hex.coupsLicites()
        #print(self.hex.plateau[:,:,0])
        for i in range(0, self.hex.taille):
            for j in range(0, self.hex.taille):

                x, y = self.hexs_centre[i, j]
                #
                if self.hex.plateau.ndim == 2 and self.hex.plateau[i,j] == 1 or self.hex.plateau.ndim == 3 and self.hex.plateau[i,j,0] == 1:
                    self.dernier_blanc = widget.create_oval(x - self.r, y - self.r, x + self.r, y + self.r,
                                                fill=self.couleur_pion_second, outline=self.couleur_pion_second)
                    self.id_pions.append(self.dernier_blanc)

                elif self.hex.plateau.ndim == 2 and self.hex.plateau[i, j] == -1 or self.hex.plateau.ndim == 3 and (self.hex.plateau[i,j,0] == -1 or self.hex.plateau[i,j,1] == 1):
                    self.dernier_noir = widget.create_oval(x - self.r, y - self.r, x + self.r, y + self.r,
                                               fill=self.couleur_pion_premier)
                    self.id_pions.append(self.dernier_noir)


                if self.affiche_coup_licites:
                    if (i,j) in coups and not hasattr(self.hex, 'taille_interieur') or (i-1,j-1) in coups and  hasattr(self.hex, 'taille_interieur'):
                        self.id_pions.append(widget.create_oval(x - self.r/k, y - self.r/k, x + self.r/k, y + self.r/k, fill="red", outline="red"))

                try:
                    d=2
                    if len(self.hex.jeu.historique) and (i, j) == self.hex.jeu.historique[-1][0] and not hasattr(self.hex.jeu, 'taille_interieur') or (i - 1, j - 1) == self.hex.jeu.historique[-1][0] and hasattr(self.hex.jeu, 'taille_interieur'):
                        self.id_pions.append(widget.create_oval(x - self.r / d, y - self.r / d, x + self.r / d, y + self.r / d, fill="red", outline="red"))
                except:
                    import traceback
                    traceback.print_exc()

    def clic(self, event):
        widget = event.widget

        l = set(widget.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)) & set(
            widget.find_withtag("hex"))

        if l :

            id = l.pop()

            (i, j) = self.hexs_inv[id]

            #print(i,j)
            #print(self.hex.coupsLicites())
            if (i, j) in self.hex.coupsLicites():

                (x, y) = self.hexs_inv_centre[id]

                if self.hex.blancJoue and self.blancHumain:
                    print('>',self.hex.blancJoue, i, j)
                    self.hex.jouer(i, j)

                    self.actualisation_graphique(widget)


                elif not self.hex.blancJoue and self.noirHumain:

                    print('>',self.hex.blancJoue, i, j)
                    self.hex.jouer(i, j)

                    self.actualisation_graphique(widget)


            if not self.hex.fini:

                if self.hex.blancJoue and  not self.blancHumain :

                    self.jouer_blanc()

                elif not self.hex.blancJoue and not self.noirHumain:

                        self.jouer_noir()

            if self.hex.fini:
                print([(a,b) for a,b, *_ in self.hex.historique])
                if self.hex.gagnant == 'blanc':
                    gagnant = '2e joueur'
                else:
                    gagnant = '1e joueur'
                id = self.canvas.create_text(self.largeur / 2, 40, text="Gagnant : " + gagnant, font="Arial 50",
                                   fill="black")

                self.banderole.append(id)
                self.id_pions.append(id)



    def lancer(self):

        self.canvas.pack()

        self.mainloop()


    def relancer(self):

        txt = self.title()

        self.destroy()
        self.quit()

        if self.ia_noir:
            self.ia_noir.vider_memoire()

        if self.ia_blanc:
            self.ia_blanc.vider_memoire()

        self.hex.raz()

        i_hex = Interface_hex_dyna(self.hex, self.hauteur, self.largeur, self.ia_noir, self.ia_blanc, self.affichage_valeurs, titre= txt, permu_plateau=self.permu_plateau, permu_couleur=self.permu_couleur)

        i_hex.lancer()

    def clavier(self, event):
        touche = event.keysym
        print(touche)

        if touche == 'u':
            self.undo(event)

    def undo(self, event):
        if self.hex.historique:
            self.hex.undo()
            self.actualisation_graphique(self.canvas)
            """self.canvas.delete(self.pieces.pop())
            if self.banderole:
                self.canvas.delete(self.banderole.pop())"""

def formatage(i):
    if isinf(i):
        return i
    return str(int(i * 100)/100)
    # return "%.2e"%i

#http://apprendre-python.com/page-tkinter-interface-graphique-python-tutoriel

