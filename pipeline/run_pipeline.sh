#!/bin/bash

PIPELINE_DIR=$(dirname "$0")
IMAGE_NAME="pipelineimage"
DOCKERFILE_PATH="$PIPELINE_DIR/Dockerfile"
SINGULARITY_DEF_PATH="$PIPELINE_DIR/pipeline.def"
HOST_DIR="$PIPELINE_DIR"
CONTAINER_DIR="/app"

# Detect the platform
MACHINE_TYPE=$(uname -m)
if [ "$MACHINE_TYPE" == "x86_64" ]; then
    PLATFORM="linux/amd64"
elif [ "$MACHINE_TYPE" == "aarch64" ] || [ "$MACHINE_TYPE" == "arm64" ]; then
    PLATFORM="linux/arm64"
else
    echo "Unsupported platform: $MACHINE_TYPE"
    exit 1
fi

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build --platform $PLATFORM -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    echo "Running Docker without GPU support"
    docker run --platform $PLATFORM -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME
# Check if Singularity is installed and Docker is not available
elif command -v singularity &> /dev/null; then
    echo "Building Singularity image from Dockerfile..."
    docker build --platform $PLATFORM -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    singularity build $IMAGE_NAME.sif $SINGULARITY_DEF_PATH
    echo "Running Singularity without GPU support"
    singularity run -B $HOST_DIR:$CONTAINER_DIR $IMAGE_NAME.sif
else
    echo "Cannot run, no docker or singularity found"
    exit 1
fi