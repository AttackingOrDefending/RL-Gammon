Interface Computer Olympiad, Oscar Perianayagassamy, développé en Python 3.13, version du 22 juillet 2025

Quel est le contenu de ce dossier ?

    Il y a 5 sous-dossiers :
        - games
        - log
        - player
        - referee
        - utils
    et 2 fichiers main_player.py et main_referee.py qui contiennent les instructions pour configurer et exécuter les bots joueur et arbitre.

    log

        Dossier qui contient : un sous-dossier "error_handling" qui accueille des fichiers ayant pour but de sauvegarder les erreurs et exceptions qui auraient pu être levées au cours de l'exécution.
                               un sous-dossier bot_play_log avec des fichiers play_log_<id_player>_<date> qui enregistrent pour le joueur d'id <id_player> l'heure de début de partie, les coups qu'il a joués, le jeu ainsi que son résultat à la fin de la partie (vainqueur/perdant).
                                            e.g. :
                                                {"13:31": {
                                                    "moves": ["C1-D1", "A3-A2", "C3-C2", "B2-A2"],
                                                    "game_name": "clobber",
                                                    "winner": true
                                                    },
                                                    ...
                                                }
                               un sous-dossier bot_ref_log avec des fichiers ref_log_<id_player>_<date> qui enregistrent pour l'heure de début de partie : chaque joueur d'id <id_player> et ses coups, le nom du jeu, le nombre de secondes accordées à chaque joueur, si oui ou non le mode time_per_move est activé, l'id du vainqueur et celui du perdant.
                                            e.g. :
                                                {"13:31": {
                                                    "1382641583186378752": ["C1-D1", "A3-A2", "C3-C2", "B2-A2"],
                                                    "1390259229612314725": ["B3-B4", "B1-A1", "D3-D4", "A1-A2"],
                                                    "game_name": "clobber",
                                                    "time_per_player": 5,
                                                    "time_per_move_mode": false,
                                                    "winner": 1382641583186378752,
                                                    "loser": 1390259229612314725
                                                    },
                                                    ...
                                                }


    utils

        Dossier qui contient : fichier mytime.py qui contient des outils pour la gestion du temps.
                               fichier pattern_enum.py qui contient deux classes en lien avec les expressions régulières et leur version compilée.


    games

        Dossier qui contient : fichier mygame.py qui contient la classe abstraite Game qui représente un jeu dans notre structure.
                               fichier games_enum.py qui contient une énumération de tous les jeux disponibles.
                                    e.g. : par défaut, il y a le jeu du Clobber
                                        class EnumGames(Enum):
                                            CLOBBER = Clobber()
                               sous-dossier instances qui contient la liste des jeux ajoutées. A priori, les classes dans ces fichiers devraient héritées la classe Game de mygame.py.


    player

        Dossier qui contient : sous-dossier model qui contient le fichier player.py où est présente la classe Player représentant un joueur, le fichier player_bot.py qui contient la classe utilisée pour créer un bot joueur ainsi que deux Cogs qui sont des fichiers qui contiennent une partie du code du bot joueur (pour des questions de lisibilité et d'espace).
                               sous-dossier instances qui contient les codes de bot joueur particulier.


    referee

        Dossier qui contient : sous-dossier model qui contient le fichier referee.py où est présente la classe Referee représentant un arbitre, le fichier referee_bot.py qui contient la classe utilisée pour créer un bot arbitre ainsi que deux Cogs qui sont des fichiers qui contiennent une partie du code du bot arbitre (pour des questions de lisibilité et d'espace).


Comment ajouter un jeu ?

    Il faut créer une classe qui hérite de Game (main\games\mygame.py) dans un fichier à placer ensuite dans main\games\instances, et qui implémente les méthodes suivantes:
        - action_to_string(action: Any) -> str
        - string_to_action(string: str) -> Any
        - plays(action: Any) -> None
        - valid_actions() -> set[Any]
        - ended() -> bool
        - reset() -> None
        - (bonus) show_game() -> str

    Et dans le constructeur de cette nouvelle classe, il faut appeler le constructeur de la classe mère Player en spécifiant (si possible) les attributs suivants :
        - name: str
        - rules: str
        - winner: Any|None
        - players_order: list[Any] qui représente la structure interne de la gestion des tours des joueurs (e.g. ça pourrait être ['black', 'white'] si un joueur joue les pions noirs, l'autre les pions blancs et que les pions noirs commencent à jouer)

    Ensuite, il faut ajouter ce jeu dans l'énumeration EnumGames (main\games\games_enum.py) de la manière suivante :
            class EnumGames(Enum):
                NEW_GAME = New_Game()
                ...

    Note : Pour que le jeu soit celui sélectionné par défaut, il faut le mettre en première position dans l'énumération.

    Enfin, il suffit d'exécuter main_referee.py pour pouvoir en profiter.


Comment ajouter un joueur ?

    Il faut créer une classe qui hérite de Player (main\player\player.py) dans un fichier à placer dans main\player\instances, et qui implémente les méthodes suivantes :
        - opponent_plays(action: str) -> None
        - plays(time_left: Time) -> None

    Note : Player possède déjà les attributs suivants : last_actions_opponents, last_actions_self, in_game, starting_time, game.
    Note 2 : l'argument time_left est par mot-clé, donc optionnel. Il faut néanmoins utilisé un objet de la classe Time si nécessaire (cette classe a pour constructeur Time(hour, minute, second, millisecond))


Comment assigner les bots au serveur désiré ?

    Il faut modifier dans le fichier .env la valeur du champs GUILD_ID.


Comment modifier l'ID du propriétaire du bot joueur ?

    Il faut modifier dans le fichier .env la valeur du champs OWNER_ID.


Comment créer et utiliser un bot arbitre ?

/!\ token du bot et non token du client secret (dans l'onglet bot)

    Il faut créer un bot depuis le "developper portal" offert par discord (page d'accueil > applications > New Application) en lui accordant les droits d'administrateur (page du bot > Bot > Bot Permissions > cocher Administrator) et cocher la case "Message Content Intent" (page du bot > Bot > Message Content Intent).
    Trouvez son Token (page du bot > OAuth2 > Client Information > Client Secret > Reset).
    Ensuite, il s'agit de l'inviter (page du bot > OAuth2 > OAuth2 URL Generator > cocher Bot dans Scopes et Administrator dans General Permissions puis utiliser le lien en bas de la page dans la rubrique Generated URL) sur le serveur de votre choix en notant l'ID du serveur.
    Puis, il faut ajouter au sein du fichier .env le TOKEN dans le champs "REFEREE_BOT_DISCORD_TOKEN" et l'ID dans la rubrique "GUILD_ID".
    Pour finir, il faut exécuter le fichier main_referee.py depuis la racine du projet.

# DETAILS :
- créer un salon discord : le gros +, à gauche, en bas de la liste des serveurs
- activer le mode développeur du compte (PARAMÈTRE > AVANCE)
Pour créer un nouveau bot, il faut d’abord se connecter au site https://discord.com/developers/applications.
Ensuite, il faut cliquer sur « New Application » en haut à droite de l’écran (encadré en bleu) et donner un nom au bot.
Il faut aussi cocher la case "Message Content Intent" (page du bot > Bot > Message Content Intent) ; et lui donner les droits d’administrateur : page du bot > Bot > Bot Permissions > cocher Administrator.
Après, il faut récupérer son TOKEN en allant depuis la page du bot dans OAuth2 > Rubrique Client Information > Client Secret > Reset Secret. C’est ce TOKEN que tu pourras renseigner dans le fichier .env après REFEREE_BOT_DISCORD_TOKEN.
Ensuite, il faut l'inviter (page du bot > OAuth2 > OAuth2 URL Generator > cocher Bot dans Scopes et Administrator dans General Permissions puis utiliser le lien en bas de la page dans la rubrique Generated URL) sur ton serveur. On colle le lien généré dans le navigateur et on choisit le serveur
Et concernant l’ID à mettre dans GUILD_ID de fichier .env, il s’agit de l’identifiant de ton serveur que tu peux récupérer en faisant clic droit sur ton serveur dans la liste de serveurs à gauche de l’interface.
- Pour ajouter un bot le "+" à côté de rôle/membres dans l'onglet permission de la fenêtre obtenu en cliquant sur l'engrenage du salon


Comment créer et utiliser un bot joueur ?


/!\ token du bot et non token du client secret (dans l'onglet bot)


    Il faut créer un bot depuis le "developper portal" offert par discord (page d'accueil > applications > New Application) en lui accordant les droits d'administrateur (page du bot > Bot > Bot Permissions > cocher Administrator) et cocher la case "Message Content Intent" (page du bot > Bot > Message Content Intent).
    Trouvez son Token (page du bot > OAuth2 > Client Information > Client Secret > Reset).
    Ensuite, il s'agit de l'inviter (page du bot > OAuth2 > OAuth2 URL Generator > cocher Bot dans Scopes et Administrator dans General Permissions puis utiliser le lien en bas de la page dans la rubrique Generated URL) sur le serveur de votre choix en notant l'ID du serveur.
    Puis, il faut ajouter au sein du fichier .env le TOKEN dans le champs "REFEREE_BOT_DISCORD_TOKEN" et l'ID dans la rubrique "GUILD_ID".
    Egalement, il faut ajouter votre ID dans le champs OWNER_ID afin de pouvoir utiliser certaines commandes.
    Modifiez aussi la ligne
            21 AIPROGRAM = <AI Program Class>
    dans le fichier main_player.py en prenant soin d'importer votre classe d'IA de joueur.
    Il reste plus qu'à exécuter main_player.py depuis la racine du projet.

    Note : les commandes du bot joueur sont à envoyer par message privé au bot.
- Pour ajouter un bot le "+" à côté de rôle/membres dans l'onglet permission de la fenêtre obtenu en cliquant sur l'engrenage du salon


Quel est le préfixe utilisé par le bot arbitre ?

    Il s'agit du point d'exclamation ("!").


Quel est le préfixe utilisé par le bot joueur ?

    Il s'agit du pourcentage ("%").


Quelles sont les commandes disponibles pour le bot arbitre ?

    La liste complète est la suivante :
        !history : affiche l'historique des parties du jour.
        !pause : permet de mettre en pause la partie en cours (commande qui nécessite les droits d'administrateur)
        !resume : permet de reprendre la partie précédemment mise en pause (commande qui nécessite les droits d'administrateur). Attention il faut que ce soit le même utilisateur qui envoie cette requête.
        !set
            !set time <seconds> : permet de modifier le nombre de secondes accordées à chaque joueur par tour/partie selon le mode de temps choisi.
            !set game <game_name> : permet de changer le mode de jeu courant du bot en game_name (si game_name est énuméré dans la classe EnumGames).
            !set time_per_move : permet de changer le mode de gestion du temps. Deux modes sont disponibles : "time_per_move" et "regular". "regular" signifie que les joueurs ont <time_per_player> secondes pour jouer AU TOTAL. Au contraire, le mode "time_per_move" signifie que les joueurs ont <time_per_player> secondes pour jouer à chaque tour.
            !set display : permet d'activer/désactiver la fonctionnalité d'affichage du plateau de jeu (si le jeu le permet).
            !set stop : permet d'activer/désactiver la fonctionnalité de terminaison prématurée de la partie.
        !start <@player1> <@player2> : permet d'initier une partie du jeu déjà enregistré dans l'instance de Referee avec les joueurs @player1 et @player2 (mentions discord). Après réception du message, le bot arbitre affiche le nom du jeu et les règles, et attend des deux joueurs de se manifester en ajoutant une réaction pouce en l'air à son message.
        !stop : permet d'initier un vote pour mettre fin à la partie prématurément. Fonctionne uniquement si l'option stop est activée et si le jeu a été mis en pause au préalable.
        !dump : permet de mettre à jour le dossier main\log\bot_ref_log en ajoutant ou modifiant le fichier json contenant l'historique de parties. (commande qui nécessite les droits d'administrateur)
        !info : permet d'afficher une multitude d'informations sur le bot arbitre et ses attributs.
        !quit : permet d'arrêter l'exécution du programme du bot arbitre. (commande qui nécessite les droits d'administrateur)
        !reset : permet de réinitialiser l'instance de Referee associé au bot en conservant le même jeu (remis à zéro également). (commande qui nécessite les droits d'administrateur)
        !help : affiche une aide générée automatiquement par Discord.


Quel format doivent respecter les coups joués par les IA et les joueurs humains ?

    Pour le moment, le bot arbitre accepte exclusivement les coups respectant la syntaxe suivante :
                [A-Z][0-9]*-[A-Z][0-9]*
    c'est-à-dire deux éléments préfixés par une lettre et suffixés par un nombre, et séparés par un tiret du 6 (-).

    Pour modifier cette option, il faut modifier le pattern RAW_PATTERN_MOVE dans main\utils\pattern_enum.


Quelles sont les commandes disponibles pour le bot joueur ?

    Pour l'instant, il y en a 3 :
        %history : affiche l'historique de partie du jour du bot joueur.
        %quit : interrompt l'exécution du bot joueur (commande qui nécessite que l'auteur ait un ID identique à celui stocké dans OWNER_ID).
        %info : affiche différentes informations sur le bot joueur.


Comment exécuter le programme depuis le terminal ?

    Pour exécuter ce projet depuis le terminal, il faut utiliser la commande suivante en étant placé à la racine du projet au préalable :

                python3 -m main.main_referee

                python3 -m main.main_player


Puis-je utiliser le même bot arbitre sur deux serveurs différents ?

    Pour le moment non car l'instance de Referee est unique et est associée à l'instance de RefereeBot considérée.


Est-ce que le bot arbitre peut arbitrer un match entre joueurs humains ?

    Oui.



=========================


Il y a un autre fichier main pour avoir un deuxième bot joueur et une nouvelle ligne dans le fichier environnement si jamais ça peut te faciliter le test avec deux bots joueurs. Néanmoins, les fichiers main n’ont pas été modifiés donc il n’est pas forcément utile pour toi d’adapter cette partie-là à tes besoins.



Pour le bot joueur :

Désormais le fichier .json est actualisé après chaque modification d’un des attributs qui y est stocké. J’ai ajouté dans le dictionnaire qui fait la passerelle avec le fichier .json du bot joueur (PlayerBot.bot_play_log) les clés suivantes :

    self_moves : tous les coups du bot.
    moves : tous les coups joués.
    ended : vrai si la partie s’est terminée correctement et faux sinon.
    stopped : vrai si la partie a connu un arrêt anticipé (avec la commande !stop si elle a été invoquée) et faux sinon.
    players : les joueurs de la partie.
    referee_id : l’id de l’arbitre de la partie.



Il n’y a plus un fichier par jour mais un par partie pour simplifier la récupération de la dernière partie jouée. Ce fichier prend désormais le nom suivant :

play_{id-joueur}_{nom-du-serveur}_{nom-du-channel}_{date}_{heure-de-début-de-partie}.json



Grâce au nom de fichier et des attributs il est normalement de possible de reconstruire la structure de joueur en cas de problème.



J’ai commencé à écrire du code qui permet de récupérer le dernier fichier de log : il le charge et l’affiche dans le cas où la partie précédente ne s’est pas terminée correctement. Cela se situe dans le fichier

main\player\model\player_bot.py

au niveau de la ligne 53 : méthode setup_hook.

Il s’agit d’une méthode native de l’API qui est exécutée une seule fois au démarrage du bot avant qu’il soit connecté aux serveurs sur lesquels il est déjà présent.



Pour le bot arbitre

J’ai ajouté les attributs suivants pour la sérialisation :

    moves
    players
    ended
    stopped

Ils ont la même signification qu’au-dessus.

Et le nom de fichier est désormais :

ref_{nom-du-serveur}_{nom-du-channel}_{date}_{heure-de-début-de-partie}.json



Pour lui aussi les enregistrements se font après chaque modification du dictionnaire (RefereeBot.bot_ref_log) faisant la passerelle avec le fichier .json.



Penses-tu que nous devrions nous pencher sur un système de récupération pour l’arbitre également ? Auquel cas je me demande si la sérialisation .json est vraiment adéquat vu le nombre d’attributs de la classe Referee.