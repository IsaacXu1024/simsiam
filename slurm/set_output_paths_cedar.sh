#!/bin/bash

echo ""
echo "-------- Setting checkpoint and output path variables ------------------"
date

# Set the JOB_LABEL environment variable
source "slurm/set_job-label.sh"

echo "# Handling data on the node"
echo ""
echo "SLURM_TMPDIR = $SLURM_TMPDIR"
echo ""
echo "ls -lh ${SLURM_TMPDIR}:"
ls -lh "${SLURM_TMPDIR}"
echo ""

ROOT_DATA_DIR="${SLURM_TMPDIR}/datasets"
rm -rf "$ROOT_DATA_DIR"
mkdir -p "$ROOT_DATA_DIR"

echo "ls -lh ${ROOT_DATA_DIR}:"
ls -lh "${ROOT_DATA_DIR}"
echo ""

CKPT_DIR="${SLURM_TMPDIR}/checkpoint"
echo "CKPT_DIR = $CKPT_DIR"
echo ""

# Ensure the checkpoint dir exists
mkdir -p "$CKPT_DIR"

# Create a symlink to the job's checkpoint directory within a subfolder of the
# current directory (repository directory) named checkpoint.
mkdir -p "checkpoints_working"
ln -sfn "$CKPT_DIR" "$PWD/checkpoints_working/$SLURM_JOB_NAME"

# Specify an output directory to place checkpoints for long term storage once
# the job is finished.
# OUTPUT_DIR is the directory that will contain all completed jobs for this
# project.
OUTPUT_DIR="$PWD/checkpoints_finished"
# JOB_OUTPUT_DIR will contain the outputs from this job.
JOB_OUTPUT_DIR="$OUTPUT_DIR/$JOB_LABEL"

echo "Current contents of ${CKPT_DIR}:"
ls -lh "${CKPT_DIR}"
echo ""
echo "JOB_OUTPUT_DIR = $JOB_OUTPUT_DIR"
if [[ -d "$JOB_OUTPUT_DIR" ]];
then
    echo "Current contents of ${JOB_OUTPUT_DIR}"
    ls -lh "${JOB_OUTPUT_DIR}"
fi
echo ""

if [[ "$start_time" != "" ]];
then
    echo "------------------------------------"
    elapsed=$(( SECONDS - start_time ))
    eval "echo Running total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
fi
echo "------------------------------------------------------------------------"
