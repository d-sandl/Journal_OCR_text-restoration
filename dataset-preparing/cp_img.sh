#!/bin/bash

# how to use:
# e.g. bash cp_img.sh 20251211/broken_20251211 20251211/clean_20251211

# Exit immediately if a command fails
set -e

PROJ_PATH=/Journal_OCR_text-restoration

# Check number of arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset-preparing/train/dataset_name>
    echo "Examples: bash cp_img.sh sans-serif_1st
    exit 1
fi

# Path to your virtual environment
BROKEN_SRC_PATH=$PROJ_PATH/dataset-preparing/train/$1/broken
CLEAN_SRC_PATH=$PROJ_PATH/dataset-preparing/train/$1/clean

BROKEN_TGT_PATH=$PROJ_PATH/data/broken
CLEAN_TGT_PATH=$PROJ_PATH/data/clean

# Ask user before clearing
read -p "Do you want to clear target (data) folders first? (y/n): " ans
if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
    echo ">>> Clearing old target directories: [$BROKEN_TGT_PATH]"
    echo ">>> Clearing old target directories: [$CLEAN_TGT_PATH]"
    rm -rf "$BROKEN_TGT_PATH"/*
    rm -rf "$CLEAN_TGT_PATH"/*
else
    echo ">>> Skipping clear step"
fi

echo ">>> Copying from [$BROKEN_SRC_PATH] to [$BROKEN_TGT_PATH]"
echo ">>> Copying from [$CLEAN_SRC_PATH] to [$CLEAN_TGT_PATH]"
cp "$BROKEN_SRC_PATH"/* "$BROKEN_TGT_PATH"/
cp "$CLEAN_SRC_PATH"/* "$CLEAN_TGT_PATH"/

echo ">>> Done!"
