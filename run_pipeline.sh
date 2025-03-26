#!/bin/bash

PIPELINE_DIR=$(dirname "$0")
IMAGE_NAME="pipelineimage"
DOCKERFILE_PATH="$PIPELINE_DIR/Dockerfile"
SINGULARITY_DEF_PATH="$PIPELINE_DIR/pipeline.def"
HOST_DIR="$PIPELINE_DIR"
CONTAINER_DIR="/app"

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH --load .
    echo "Running pipeline in Docker..."
    docker run -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME
else
    echo "Cannot run, no docker found"
    exit 1
fi