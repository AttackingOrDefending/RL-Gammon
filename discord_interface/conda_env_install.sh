#!/bin/bash

# Nom du dossier de l'environnement virtuel
ENV_NAME="DCOI_env"


echo "Création de l'environnement '$VENV_NAME'..."

conda create -n $VENV_NAME python=3.10


# 2️⃣ Activer l'environnement virtuel
echo "Activation de l'environnement ..."
# Source doit être utilisé pour activer dans le shell courant
conda activate $VENV_NAME

# 3️⃣ Affichage confirmation
echo "Environnement  '$VENV_NAME' activé !"

source install.sh