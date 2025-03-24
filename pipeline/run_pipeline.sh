#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

PIPELINE_DIR=$(dirname "$0")
SCRIPT_PATH="$PIPELINE_DIR/pipeline.py"
LICENSE_PATH="$PIPELINE_DIR/license.txt"
CONTAINER_PATH="$PIPELINE_DIR/PipelineContainer.sif"
DATA_PATH="$PIPELINE_DIR/../path/to/data"

# Extract the directory name and filename from the data path
DIRNAME=$(dirname "$DATA_PATH")
FILENAME=$(basename "$DATA_PATH")

# Check for nvidia GPU
if command -v nvidia-smi &> /dev/null; then
    NV_FLAG="--nv"
else
    NV_FLAG=""
fi

# Build the Singularity container if it doesn't exist
if [ ! -f "$CONTAINER_PATH" ]; then
    singularity build $CONTAINER_PATH $PIPELINE_DIR/PipelineContainer.def
fi

singularity exec $NV_FLAG \
    --no-home \
    -B $DATA_PATH:$DATA_PATH \
    -B $LICENSE_PATH:$LICENSE_PATH \
    -B $SCRIPT_PATH:$SCRIPT_PATH \
    $CONTAINER_PATH \ python3 $SCRIPT_PATH $DATA_PATH $LICENSE_PATH
