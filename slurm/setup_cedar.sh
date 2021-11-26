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
if [ ! -d "$ENV_DIR" ]; then
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

# Print env status
source "slurm/report_env_config.sh"

# Set checkpoint and output path environment variables
source "slurm/set_output_paths_cedar.sh"

echo "-------- Dataset transfer ---------------------------------------------"
date
echo ""

if [ "$DATASET" == "imagenet" ]; then
    DATA_DIR="$ROOT_DATA_DIR/$DATASET"

    if [ -d "${DATA_DIR}/val" ]; then
        echo "# Dataset already present in directory $DATA_DIR, skipping copy"

    else
        echo "# Copying imagenet dataset to local node's SSD storage, ${DATA_DIR}"

        rsync -vz /project/rrg-ttt/datasets/imagenet/* "${DATA_DIR}/"

        mkdir -p "$DATA_DIR/train"
        mkdir -p "$DATA_DIR/val"

        echo "Extracting training data"
        tar -C "${DATA_DIR}/train" -xf "${DATA_DIR}/ILSVRC2012_img_train.tar"
        find "${DATA_DIR}/train" -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        echo ""

        echo "Extracting validation data"
        tar -C "${DATA_DIR}/val" -xf "${DATA_DIR}/ILSVRC2012_img_val.tar"

        # Move validation images to subfolders:
        VAL_PREPPER="$DATA_DIR/valprep.sh"
        if test -f "$VAL_PREPPER"; then
            echo "Downloading valprep.sh";
            wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/b81dbb1/valprep.sh -O "$VAL_PREPPER";
        fi
        echo "Moving validation data into subfolders"
        (cd "${DATA_DIR}/val"; sh "$VAL_PREPPER")
        echo ""

        # Check total files after extract
        #
        #  find train/ -name "*.JPEG" | wc -l
        #  # Should be 1281167
        #  find val/ -name "*.JPEG" | wc -l
        #  # Should be 50000
    fi

elif [ "$DATASET" == "imagenette2-160" ] || [ "$DATASET" == "imagenette" ]; then
    DATA_DIR="$ROOT_DATA_DIR/$DATASET"

    if [ -d "$DATA_DIR/imagenette2-160" ]; then
        echo "# Dataset imagenette2-160 already on local node's SSD storage, ${DATA_DIR}"

    else
        echo "# Copying imagenette2-160 dataset to local node's SSD storage, ${DATA_DIR}"

        rsync -zv "/project/rrg-ttt/datasets/imagenette2/imagenette2-160.tgz" "${DATA_DIR}/"

        echo ""
        echo "Extracting data"
        tar -C "${DATA_DIR}" -xf "${DATA_DIR}/imagenette2-160.tgz"

    fi;
    DATA_DIR="$DATA_DIR/imagenette2-160"

elif [ -d "$DATASET" ]; then
    echo ""
    echo "# Copying directory $DATASET to local node's SSD storage, ${DATA_DIR}"
    rsync -zv "$DATASET" "${DATA_DIR}/"

elif [ -f "$DATASET" ]; then
    echo ""

    DATASET_FNAME="${DATASET##*/}"
    DATASET_NAME="${DATASET_FNAME%%.*}"
    DATASET_EXT="${DATASET_FNAME#*.}"
    TARGET_FNAME="${ROOT_DATA_DIR}/${DATASET_FNAME}"
    DATA_DIR="$ROOT_DATA_DIR/$DATASET_NAME"

    if [ -d "$DATA_DIR" ]; then
        echo "$DATA_DIR already exists, skipping copy and extraction step"

    else
        if [ "$DATASET_EXT" == "tar.gz" ] || [ "$DATASET_EXT" == "tgz" ] || [ "$DATASET_EXT" == "tar" ]; then
            echo ""
        else
            echo "Unsupported file extension: $DATASET_EXT"
            exit;
        fi
        echo "# Copying file $DATASET to local node's SSD storage, ${DATA_DIR}"
        rsync -zv "$DATASET" "${ROOT_DATA_DIR}/"

        echo "Untarring file $TARGET_FNAME"
        tar -C "${ROOT_DATA_DIR}" -xf "$TARGET_FNAME"

    fi

else
    echo "Invalid dataset name: $DATASET"
    exit;

fi

if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "------------------------------------------------------------------------"
