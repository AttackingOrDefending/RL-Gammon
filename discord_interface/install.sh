#!/bin/bash


EXPECTED_DIR_NAME="discord_interface"

CURRENT_DIR_NAME=$(basename "$PWD")

# Verification
if [ "$CURRENT_DIR_NAME" = "$EXPECTED_DIR_NAME" ]; then


pip3 install pexpect

pip3 install -U discord.py

pip3 install -U numpy

pip3 install -U aiofiles

UPPER_DIR_DCOI=$(dirname "$PWD")

if [ -f "$HOME/.zshrc" ]; then
    echo "export PYTHONPATH=$PYTHONPATH:"$UPPER_DIR_DCOI >> ~/.zshrc
fi
    echo "export PYTHONPATH=$PYTHONPATH:"$UPPER_DIR_DCOI >> ~/.bashrc


export PYTHONPATH=$PYTHONPATH:$UPPER_DIR_DCOI
echo
echo "DCOI Installation successfull."
echo

else
    echo "Current directory is NOT '$EXPECTED_DIR_NAME'."
    echo "Current directory : '$CURRENT_DIR_NAME'"
fi

