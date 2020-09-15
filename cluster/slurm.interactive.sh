#!/bin/bash

## This is an example of a script to run  tensorflow interactively
## using Singularity to run the tensorflow image.
##
## You will be dropped into an interactive shell with the tensorflow environment.
##
## Make a copy of this script and change reserved resources on the srun command as needed for your job.
##
## Just execute this script on the command line.

unset OMP_NUM_THREADS
REPO_BASE="fast-influence-functions"
DOCKER_TAG="IMAGE_TAG"
BUILD_UNIQUE_TAG=$1

SINGULARITY_BUILD_DIR="/pine/scr/h/a/hanhan1/DLResearch/SingularityCache/${REPO_BASE}-${DOCKER_TAG}-${BUILD_UNIQUE_TAG}"

# Run interactive job to GPU node using Singularity
# Note that users might see the message:
# WARNING: NVIDIA binaries may not be bound with --writable
# And `nvidia-smi` will not be accessible
# Don't worry, one can safely ignore this message.
# https://github.com/sylabs/singularity/issues/2944
srun \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=128G \
    --time=11-0 \
    --partition=volta-gpu \
    --gres=gpu:4 \
    --qos=gpu_access \
    --pty bash -c "\
        singularity build --sandbox \
            ${SINGULARITY_BUILD_DIR} \
            docker://hanguo97/${REPO_BASE}:${DOCKER_TAG} && \
        singularity shell \
            --writable --nv \
            -B /proj/bansallab/users/:/nlp \
            -H /proj/bansallab/users/han/:/home/ \
            ${SINGULARITY_BUILD_DIR}"


