#!/bin/bash

echo "-------- Environment set up --------------------------------------------"
date
echo ""

# Load CC modules
echo ""
echo "Loading modules"
echo ""
module load python/3.8
module load cuda cudnn
module load scipy-stack

# Make an environment, housed on the node's local SSD storage
ENV_DIR="$SLURM_TMPDIR/env"
if [ -f "$ENV_DIR"]; then
    echo ""
    echo "Creating environment $ENV_DIR"
    echo ""
    virtualenv --no-download "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Install pytorch
echo ""
echo "Installing packages into $ENV_DIR"
echo ""
# The recommend way is just to do this
python -m pip install --no-index torch==1.9.1 torchvision==0.10.0
python -m pip install -r requirements.txt
