#!/bin/bash

IMAGE_NAME="your-image-name"
DOCKERFILE_PATH="path/to/Dockerfile"
HOST_DIR="/path/to/host/dir"
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
        docker run --gpus all -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME /app/run_fastsurfer.sh
    else
        echo "Running Docker without GPU support"
        docker run -v $HOST_DIR:$CONTAINER_DIR -it $IMAGE_NAME /app/run_fastsurfer.sh
    fi
# Check if Singularity is installed
elif command -v singularity &> /dev/null; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    echo "Running Singularity container from Docker image..."
    if [ "$GPU_SUPPORT" = true ]; then
        echo "Running Singularity with GPU support"
        singularity run --nv -B $HOST_DIR:$CONTAINER_DIR docker-daemon://$IMAGE_NAME:latest /app/run_fastsurfer.sh
    else
        echo "Running Singularity without GPU support"
        singularity run -B $HOST_DIR:$CONTAINER_DIR docker-daemon://$IMAGE_NAME:latest /app/run_fastsurfer.sh
    fi
else
    echo "Neither Docker nor Singularity is installed on this system."
    exit 1
fi