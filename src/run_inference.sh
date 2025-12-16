#!/bin/bash

# Exit immediately if a command fails
set -e

# Path to your virtual environment
PROJ_PATH=~/Journal_OCR_text-restoration

# Activate venv
source $PROJ_PATH/venv/bin/activate

# Run the Python script
python $PROJ_PATH/src/inference.py $1

# Deactivate venv
deactivate
