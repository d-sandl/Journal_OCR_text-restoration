#!/bin/bash

# how to use:
# e.g. bash cp_img.sh 20251211/broken_20251211 20251211/clean_20251211

# Exit immediately if a command fails
set -e

TODAY=$(date +%Y%m%d)

# Path to your virtual environment
BROKEN_SRC_PATH=~/text-restoration/dataset-preparing/$TODAY/broken_$TODAY
CLEAN_SRC_PATH=~/text-restoration/dataset-preparing/$TODAY/clean_$TODAY

BROKEN_TGT_PATH=~/text-restoration/data/broken
CLEAN_TGT_PATH=~/text-restoration/data/clean

echo ">>> Clearing old target directories: [$BROKEN_TGT_PATH]"
echo ">>> Clearing old target directories: [$CLEAN_TGT_PATH]"
rm -rf "$BROKEN_TGT_PATH"/*
rm -rf "$CLEAN_TGT_PATH"/*

echo ">>> Copying from [$BROKEN_SRC_PATH] to [$BROKEN_TGT_PATH]"
echo ">>> Copying from [$CLEAN_SRC_PATH] to [$CLEAN_TGT_PATH]"
cp "$BROKEN_SRC_PATH"/* "$BROKEN_TGT_PATH"/
cp "$CLEAN_SRC_PATH"/* "$CLEAN_TGT_PATH"/

echo ">>> Done!"
