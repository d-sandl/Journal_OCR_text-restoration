#!/bin/bash

# how to use: bash run_synthetic.sh ./raw/good_sans-serif_1st
# Example: bash run_synthetic.sh ./raw/good_sans-serif_1st

# Exit immediately if a command fails
set -e

# Path to your virtual environment
PROJ_PATH=~/text-restoration

# Activate venv
source $PROJ_PATH/venv/bin/activate

# Run the Python script
python ~/text-restoration/dataset-preparing/synthetic.py $1

# Deactivate venv
deactivate
