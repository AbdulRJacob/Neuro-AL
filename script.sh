#!/bin/bash

# Function to setup environment
setup() {
    echo "Checking Git submodules..."
    if [ ! -d "symbolic/modules/aba_asp/.git" ]; then
        echo "Initializing aba_asp submodule..."
        (cd symbolic_modules && git submodule update --init --recursive aba_asp)
    else
        echo "Submodule aba_asp already initialized."
    fi

    echo "Copying aba_asp contents to src/symbolic_modules/"
    cp -r symbolic_modules/aba_asp src/symbolic_modules/
    
}

# Function to train SHAPES model
train_shapes() {

    echo "Building Docker container..."
    docker build -t neuro-al .
    
    echo "Storing container ID..."
    CONTAINER_ID=$(docker create --name neuro-al-container -v $(pwd)/src:/neuro_al neuro-al)
    echo "$CONTAINER_ID" > container_id.txt
    echo "Container ID: $CONTAINER_ID"


    if [ ! -f container_id.txt ]; then
        echo "Error: Container not found. Run --setup first."
        exit 1
    fi
    CONTAINER_ID=$(cat container_id.txt)
    
    echo "Starting container..."
    docker start "$CONTAINER_ID"
    
    echo "Running SHAPES training scripts..."
    docker exec "$CONTAINER_ID" python3 data/SHAPES.py
    docker exec "$CONTAINER_ID" python3 training/train_shapes.py
    docker exec "$CONTAINER_ID" python3 results.py
}


train_clevr() {
    echo "Extracting CLEVR dataset path from config..."
    VOLUME_CLEVR=$(grep 'dataset_source:' src/config/clevr_config.yaml | awk '{print $2}' | tr -d '"')

    
    if [ -z "$VOLUME_CLEVR" ]; then
        echo "Error: Could not find dataset_source path in clevr_config.yaml."
        exit 1
    fi
    
    echo "Setting volume mount path: $VOLUME_CLEVR"
    export VOLUME_CLEVR=$VOLUME_CLEVR
    echo "VOLUME_CLEVR=$VOLUME_CLEVR" > .env
    
    echo "Building and starting containers using Docker Compose..."
    docker-compose --env-file .env up -d --build
    
    echo "Finding running container ID..."
    CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "Error: Could not find running container."
        exit 1
    fi
    
    echo "Running CLEVR training scripts..."
    docker exec "$CONTAINER_ID" python3 training/clevr_train_neural.py
}

# Function to clean up environment
cleanup() {
    echo "Removing Docker container..."
    if [ -f container_id.txt ]; then
        CONTAINER_ID=$(cat container_id.txt)
        docker rm -f "$CONTAINER_ID"
        rm -f container_id.txt
        echo "Docker container removed."
    else
        echo "No container found."
    fi
    
    echo "Removing aba_asp directory..."
    rm -rf src/symbolic_modules/aba_asp
    
    echo "Removing backgrounds files from src/ directory..."
    find src/ -type f -name "*.pl" -delete
    
    echo "Cleanup complete."
}

# Argument parsing
case "$1" in
    --setup)
        setup
        ;;
    --train_shapes)
        train_shapes
        ;;
    --train_clevr)
        train_clevr
        ;;
    --cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 --setup | --train_shapes | --train_clevr | --cleanup"
        exit 1
        ;;
esac
