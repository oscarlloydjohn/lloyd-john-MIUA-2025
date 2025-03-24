#!/bin/bash

PIPELINE_DIR=$(dirname "$0")
IMAGE_NAME="pipelineimage"
DOCKERFILE_PATH="$PIPELINE_DIR/Dockerfile"
SINGULARITY_DEF_PATH="$PIPELINE_DIR/pipeline.def"
HOST_DIR="$PIPELINE_DIR"
CONTAINER_DIR="/app"

# Check if nvidia-smi is available to determine GPU presence
if command -v nvidia-smi &> /dev/null; then
    GPU_SUPPORT=true
else
    GPU_SUPPORT=false
fi

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    if [ "$GPU_SUPPORT" = true ]; then
        echo "Running Docker with GPU support"
        docker run --gpus all -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME
    else
        echo "Running Docker without GPU support"
        docker run -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME
    fi
# Check if Singularity is installed and Docker is not available
elif command -v singularity &> /dev/null; then
    echo "Building Singularity image from Dockerfile..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    singularity build $IMAGE_NAME.sif $SINGULARITY_DEF_PATH
    if [ "$GPU_SUPPORT" = true ]; then
        echo "Running Singularity with GPU support"
        singularity run --nv -B $HOST_DIR:$CONTAINER_DIR $IMAGE_NAME.sif
    else
        echo "Running Singularity without GPU support"
        singularity run -B $HOST_DIR:$CONTAINER_DIR $IMAGE_NAME.sif
    fi
else
    echo "Cannot run, no docker or singularity found"
    exit 1
fi