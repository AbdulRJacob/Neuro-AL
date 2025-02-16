#!/bin/bash

# Function to setup environment
setup() {
    echo "Checking Git submodules..."
    if [ ! -d "symbolic_modules/aba_asp/.git" ]; then
        echo "Initializing aba_asp submodule..."
        (cd symbolic_modules && git submodule update --init --recursive aba_asp)
    else
        echo "Submodule aba_asp already initialized."
    fi


    echo "Copying aba_asp contents to src/symbolic_modules/"
    cp -r symbolic_modules/aba_asp src/symbolic_modules/
    
    echo "Building Docker container..."
    docker build -t neuro-al .
    
    echo "Storing container ID..."
    CONTAINER_ID=$(docker create --name neuro-al-container -v $(pwd)/src:/neuro_al neuro-al)
    echo "$CONTAINER_ID" > container_id.txt
    echo "Container ID: $CONTAINER_ID"
}

# Function to start training
train() {
    if [ ! -f container_id.txt ]; then
        echo "Error: Container not found. Run --setup first."
        exit 1
    fi
    CONTAINER_ID=$(cat container_id.txt)
    
    echo "Starting container..."
    docker start "$CONTAINER_ID"
    
    echo "Running training scripts..."
    docker exec "$CONTAINER_ID" python3 data/SHAPES.py
    docker exec "$CONTAINER_ID" python3 training/shapes_train_neural.py
    docker exec "$CONTAINER_ID" python3 training/shapes_train_symbolic.py
    docker exec "$CONTAINER_ID" python3 inference/shapes_inference.py
}

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
    --train)
        train
        ;;
    --cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 --setup | --train | --cleanup"
        exit 1
        ;;
esac
