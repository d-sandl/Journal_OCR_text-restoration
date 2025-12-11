#!/bin/bash

# Exit immediately if a command fails
set -e

# Path to your virtual environment
BROKEN_SRC_PATH=~/text-restoration/dataset-preparing/$1
CLEAN_SRC_PATH=~/text-restoration/dataset-preparing/$2

BROKEN_TGT_PATH=~/text-restoration/data/broken
CLEAN_TGT_PATH=~/text-restoration/data/clean

# Activate venv
source $PROJ_PATH/venv/bin/activate

# Run the Python script
python synthetic_atm_sans-serif.py

# Deactivate venv
deactivate
