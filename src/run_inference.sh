#!/bin/bash

# Exit immediately if a command fails
set -e

# Path to your virtual environment
PROJ_PATH=~/text-restoration

# Activate venv
source $PROJ_PATH/venv/bin/activate

# Run the Python script
python ~/text-restoration/src/inference.py

# Deactivate venv
deactivate
