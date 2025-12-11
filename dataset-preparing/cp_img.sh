# how to use:
# e.g. bash cp_img.sh 20251211/broken_20251211 20251211/clean_20251211

# Exit immediately if a command fails
set -e

# Path to your virtual environment
BROKEN_SRC_PATH=~/text-restoration/dataset-preparing/$1
CLEAN_SRC_PATH=~/text-restoration/dataset-preparing/$2

BROKEN_TGT_PATH=~/text-restoration/data/broken
CLEAN_TGT_PATH=~/text-restoration/data/clean

echo ">>> Clearing old target directories...:" "$BROKEN_TGT_PATH"
echo ">>> Clearing old target directories...:" "$CLEAN_TGT_PATH"
rm -rf $BROKEN_TGT_PATH
rm -rf $CLEAN_TGT_PATH

echo ">>> copying to target directories...:" "$BROKEN_TGT_PATH"
echo ">>> copying to target directories...:" "$CLEAN_TGT_PATH"
cp "$BROKEN_SRC_PATH"/* "$BROKEN_TGT_PATH"/
cp "$CLEAN_SRC_PATH"/* "$CLEAN_TGT_PATH"/
