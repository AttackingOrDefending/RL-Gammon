import tkinter
import tkinter.font as tkFont

from math import isinf
from tkinter import *



class Interface_square_deplacement(Tk):

    def __init__(self, hex, hauteur = 1000, largeur = 1500, ia_noir = None, ia_blanc = None, affichage_valeurs=False, titre=None, permu_couleur= False, permu_plateau = False, resize_txt = 1, ia_auto=False):
        self.ia_auto = ia_auto
        self.last_pos = None

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


        hauteur_case = hauteur / (hex.taille + 1)
        largeur_case = hauteur / (hex.taille + 1)


        self.r = (int)(3*min(hauteur_case, largeur_case)/4/2)

        self.dernier_blanc = 0
        self.dernier_noir = 0

        """
        bordX = {}
        bordY = {}

        for i in range(0, 2):
            for j in range(0, 2):
                X = j * largeur_case
                Y = i * hauteur_case

                bordX[i, j] = X
                bordY[i, j] = Y

        self.canvas.create_polygon(bordX[0, 0], bordY[0, 0], bordX[0, 0] - dec_anti_X, bordY[0, 0] - dec_anti_Y,
                              bordX[0, 1] + dec_X, bordY[0, 1] - dec_Y, bordX[0, 1], bordY[0, 1], fill=self.couleur_bord_horizontal)

        self.canvas.create_polygon(bordX[1, 1], bordY[1, 1], bordX[1, 1] + dec_anti_X, bordY[1, 1] + dec_anti_Y,
                              bordX[0, 1] + dec_X, bordY[0, 1] - dec_Y, bordX[0, 1], bordY[0, 1], fill=self.couleur_bord_vertical)

        self.canvas.create_polygon(bordX[1, 0], bordY[1, 0], bordX[1, 0] - dec_X, bordY[1, 0] + dec_Y,
                              bordX[1, 1] + dec_anti_X, bordY[1, 1] + dec_anti_Y, bordX[1, 1], bordY[1, 1],
                              fill=self.couleur_bord_horizontal)

        self.canvas.create_polygon(bordX[0, 0], bordY[0, 0], bordX[0, 0] - dec_anti_X, bordY[0, 0] - dec_anti_Y,
                              bordX[1, 0] - dec_X, bordY[1, 0] + dec_Y, bordX[1, 0], bordY[1, 0], fill=self.couleur_bord_vertical)"""

        self.id_pions = []
        if not permu_plateau:
            for i in range(0, hex.taille):
                for j in range(0, hex.taille):
                    X = (j + 1) * hauteur_case
                    Y = (i + 1) * largeur_case

                    p1 = (X, Y)
                    p2 = (X + hauteur_case, Y)
                    p3 = (X + hauteur_case, Y + largeur_case)
                    p4 = (X, Y + largeur_case)

                    centreX = X + hauteur_case / 2
                    centreY = Y + largeur_case / 2

                    self.hexs[i, j] = self.canvas.create_polygon(*(p1 + p2 + p3 + p4), outline="black", fill="grey",
                                                       activefill="orange", activeoutline="red", tags="hex")
                    self.hexs_inv[self.hexs[i, j]] = (i, j)
                    self.hexs_inv_centre[self.hexs[i, j]] = (centreX, centreY)
                    self.hexs_centre[i, j] = (centreX, centreY)
                    self.ordre_valeurs[i, j] = self.canvas.create_text(centreX, centreY, fill="purple", font=tkFont.Font(size=int(40 * resize_txt)))
                    self.valeurs[i, j] = self.canvas.create_text(centreX, centreY+50, fill="purple", font=tkFont.Font(size=int(18 * resize_txt)))

        else:
            """
            for i in range(0, hex.taille):
                for j in range(0, hex.taille):
                    X = (j + 1) * largeur_case
                    Y = (i + 1) * hauteur_case

                    p1 = (X, Y)
                    p2 = (X + largeur_case, Y)
                    p3 = (X + largeur_case, Y + largeur_case)
                    p4 = (X, Y + largeur_case)

                    centreX = X + largeur_case / 2
                    centreY = Y + largeur_case / 2

                    self.hexs[i, j] = self.canvas.create_polygon(*(p1 + p2 + p3 + p4), outline="black",
                                                                 fill="grey",
                                                                 activefill="orange", activeoutline="red", tags="hex")
                    self.hexs_inv[self.hexs[i, j]] = (i, j)
                    self.hexs_inv_centre[self.hexs[i, j]] = (centreX, centreY)
                    self.hexs_centre[i, j] = (centreX, centreY)
                    self.ordre_valeurs[i, j] = self.canvas.create_text(centreX, centreY, fill="purple",
                                                                       font=tkFont.Font(size=int(40 * resize_txt)))
                    self.valeurs[i, j] = self.canvas.create_text(centreX, centreY + 50, fill="purple",
                                                                 font=tkFont.Font(size=int(18 * resize_txt)))
                    
                    if hex.plateau[i,j] == 1:
                        self.canvas.create_oval(centreX - self.r, centreY - self.r, centreX + self.r, centreY + self.r,
                                            fill="white")
                    elif hex.plateau[i,j] == -1:
                        self.canvas.create_oval(centreX - self.r, centreY - self.r, centreX + self.r, centreY + self.r,
                                            fill="black")"""

        self.actualisation_graphique(self.canvas)

        self.canvas.bind("<Button-1>", self.clic)
        self.bind("<Key>", self.clavier)
        self.bind('<Control-z>', self.undo)
        self.bind('<Command-z>', self.undo)

        if not self.noirHumain:
            self.jouer_noir()




    def jouer_noir(self):

        self.raz_txt()

        if self.affichage_valeurs:
            (i, j), dic = self.ia_noir.meilleur_coup_valeurs(self.hex)
            print(dic)

        else:
            (i, j) = self.ia_noir.meilleur_coup(self.hex)

        self.hex.jouer(i, j)

        self.actualisation_graphique(self.canvas)

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
                dic_v = {}
                dic_i = {}
                for coup, v in sorted(dic.items(), key=lambda t: (-t[1][0], -t[1][1])):
                    i += 1
                    try:
                        dic_v[coup[0]].append(v[0])
                    except:
                        dic_v[coup[0]] = [v[0]]

                    try:
                        dic_i[coup[0]].append(v[1])
                    except:
                        dic_i[coup[0]] = [v[1]]

                    id_v = self.valeurs[coup[0]]
                    id_o = self.ordre_valeurs[coup[0]]
                    self.canvas.tag_raise(id_v)
                    self.canvas.tag_raise(id_o)
                    self.canvas.itemconfigure(id_v, text=formatage(dic_v[coup[0]]))#+' / '+formatage(dic_i[coup[0]])
                    self.canvas.itemconfigure(id_o, text=str(i))

        if self.ia_auto and not self.hex.fini:
            if self.hex.blancJoue:
                self.jouer_blanc()
            else:
                self.jouer_noir()



    def jouer_blanc(self):

        self.raz_txt()

        if self.affichage_valeurs:
            (i, j), dic = self.ia_blanc.meilleur_coup_valeurs(self.hex)
            #print(dic)

        else:
            (i, j) = self.ia_blanc.meilleur_coup(self.hex)

        self.hex.jouer(i, j)

        self.actualisation_graphique(self.canvas)

        if self.affichage_valeurs:

            i = 0

            valeurs={}
            ordres={}
            for coup, v in sorted(dic.items(), key=lambda t: -t[1]):
                i += 1

                id_v_init = self.valeurs[coup[0]]
                id_v_final = self.valeurs[coup[1]]

                id_o = self.ordre_valeurs[coup[0]]

                try:
                    valeurs[coup[1]]+='\n'+formatage(v)
                except:
                    valeurs[coup[1]] = formatage(v)

                try:
                    valeurs[coup[0]]+='\n'+str(i)
                except:
                    valeurs[coup[0]] = str(i)

                id = self.canvas.create_line(*self.canvas.coords(id_v_init), *self.canvas.coords(id_v_final), arrow=tkinter.LAST)
                self.id_pions.append(id)

                self.canvas.tag_raise(id_v_final)
                self.canvas.tag_raise(id_o)
                self.canvas.itemconfigure(id_v_final, text=valeurs[coup[1]])
                self.canvas.itemconfigure(id_o, text=valeurs[coup[0]])

        if self.ia_auto and not self.hex.fini:
            if self.hex.blancJoue:
                self.jouer_blanc()
            else:
                self.jouer_noir()


    def raz_txt(self):

        for (i,j), id in self.valeurs.items() :
            self.canvas.itemconfigure(id, text="")

        for (i,j), id in self.ordre_valeurs.items() :
            self.canvas.itemconfigure(id, text="")

    def actualisation_graphique(self, widget):

        k = 4

        for id in self.id_pions:
            widget.delete(id)

        coups = self.hex.coupsLicites()

        for i in range(0, self.hex.taille):
            for j in range(0, self.hex.taille):

                x, y = self.hexs_centre[i, j]
                #
                if self.hex.plateau.ndim == 2 and self.hex.plateau[i,j] == 1 or self.hex.plateau.ndim == 3 and self.hex.plateau[i,j,0] == 1:
                    self.dernier_blanc = widget.create_oval(x - self.r, y - self.r, x + self.r, y + self.r,
                                                fill=self.couleur_pion_second, outline=self.couleur_pion_second)
                    self.id_pions.append(self.dernier_blanc)

                elif self.hex.plateau.ndim == 2 and self.hex.plateau[i, j] == -1 or self.hex.plateau.ndim == 3 and self.hex.plateau[i,j,0] == -1:
                    self.dernier_noir = widget.create_oval(x - self.r, y - self.r, x + self.r, y + self.r,
                                               fill=self.couleur_pion_premier)
                    self.id_pions.append(self.dernier_noir)


                if (i,j) in coups:
                    self.id_pions.append(widget.create_oval(x - self.r/k, y - self.r/k, x + self.r/k, y + self.r/k, fill="red", outline="red"))


        if self.hex.fini:
            try:
                id = self.canvas.create_text(self.largeur / 2, 40, text="Gagnant : " + self.hex.gagnant, font="Arial 50",
                                             fill="black")
            except:
                id = self.canvas.create_text(self.largeur / 2, 40, text="Gagnant : " + self.hex.winner, font="Arial 50",
                                             fill="black")
            self.id_pions.append(id)

    def simplification_coups(self, coups):
        L=[]
        for coup in coups:
            if not isinstance(coup[1][0], int):
                L.append(self.simplification_coup(coup))
            else:
                L.append(coup)
        return L

    def simplification_coup(self, coup):
        return (coup[0], coup[1][0][-1])

    def complexification(self, coup, coups):
        for cp in coups:
            if isinstance(coup[1][0], int) and coup == cp or not isinstance(coup[1][0], int) and coup == self.simplification_coup(cp):
                return cp


    def clic(self, event):
        widget = event.widget

        l = set(widget.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)) & set(
            widget.find_withtag("hex"))

        if l :

            id = l.pop()

            (i, j) = self.hexs_inv[id]

            print(i,j)
            print(self.hex.coupsLicites())
            print(self.simplification_coups(self.hex.coupsLicites()))
            print()

            if not self.last_pos:
                self.last_pos = (i,j)
            else:

                coup = (self.last_pos, (i, j))
                print(coup, coup in self.hex.coupsLicites(), self.hex.coupsLicites())
                print(coup, coup in self.simplification_coups(self.hex.coupsLicites()), self.simplification_coups(self.hex.coupsLicites()))

                self.last_pos = None
                if coup in self.simplification_coups(self.hex.coupsLicites()):

                    coup = self.complexification(coup, self.hex.coupsLicites())

                    (x, y) = self.hexs_inv_centre[id]

                    if self.hex.blancJoue and self.blancHumain:

                        self.hex.jouer(*coup)

                        self.actualisation_graphique(widget)


                    elif not self.hex.blancJoue and self.noirHumain:

                        self.hex.jouer(*coup)

                        self.actualisation_graphique(widget)

            if not self.hex.fini:

                if self.hex.blancJoue and  not self.blancHumain :

                    self.jouer_blanc()

                elif not self.hex.blancJoue and not self.noirHumain:

                        self.jouer_noir()
                        #print(self.ia_noir.evaluation(jeu))

            if self.hex.fini:
                    id = self.canvas.create_text(self.largeur / 2, 40, text="Gagnant : " + self.hex.gagnant, font="Arial 50",
                                       fill="black")
                    self.id_pions.append(id)

    def clavier(self, event):
        touche = event.keysym
        print(touche)

        if touche == 'u':
            self.undo(event)

    def undo(self, event):
        if self.hex.historique:
            self.hex.undo()
        self.actualisation_graphique(self.canvas)
        

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

        i_hex = Interface_square_deplacement(self.hex, self.hauteur, self.largeur, self.ia_noir, self.ia_blanc, self.affichage_valeurs, titre= txt, permu_plateau=self.permu_plateau, permu_couleur=self.permu_couleur)

        i_hex.lancer()


def formatage(i):
    if isinstance(i, list):
        return ' '.join(formatage(e) for e in i)
    if isinf(i):
        return i
    return str(int(i * 100)/100)
    # return "%.2e"%i

