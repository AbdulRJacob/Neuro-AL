#!/bin/bash

# Function to setup environment
setup() {
    echo "Checking Git submodules..."
    if [ ! -d "symbolic/modules/aba_asp/.git" ]; then
        echo "Initializing aba_asp submodule..."
        (cd symbolic/modules && git submodule update --init --recursive aba_asp)
    else
        echo "Submodule aba_asp already initialized."
    fi
    
    echo "Copying aba_asp contents to src/symbolic_modules/"
    mkdir -p src/symbolic_modules/
    cp -r symbolic/modules/aba_asp/* src/symbolic_modules/
    
    echo "Building Docker container..."
    docker build -t neuro-al .
    
    echo "Storing container ID..."
    CONTAINER_ID=$(docker create --name neuro-al-container -v $(pwd)/src:/neuro_al neuro-al)
    echo "$CONTAINER_ID" > container_id.txt
    echo "Container ID: $CONTAINER_ID"
}

# Function to start the container if not already running
start_container() {
    echo "Finding running container ID..."
    CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -z "$CONTAINER_ID" ]; then
        if [ ! -f container_id.txt ]; then
            echo "Error: Container not found. Run --setup first."
            exit 1
        fi
        CONTAINER_ID=$(cat container_id.txt)
        echo "Starting container..."
        docker start "$CONTAINER_ID"
        docker exec "$CONTAINER_ID" python3 data/SHAPES.py

    fi
}

# Function to start Docker Compose if not already running
start_docker_compose() {
    echo "Checking if a Docker Compose container is already running..."
    COMPOSE_CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -n "$COMPOSE_CONTAINER_ID" ]; then
        echo "Docker Compose container is already running. Skipping startup."
        return
    fi


    echo "Building and starting containers using Docker Compose..."
    docker-compose --env-file .env up -d --build
}

# Function to train SHAPES model
train_shapes() {
    start_container
    
    echo "Running SHAPES training scripts..."
    docker exec "$CONTAINER_ID" python3 data/SHAPES.py
    docker exec "$CONTAINER_ID" python3 training/train_shapes.py
}

# Function to train CLEVR model using Docker Compose
train_clevr() {

    start_docker_compose
    
    echo "Finding running container ID..."
    CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "Error: Could not find running container."
        exit 1
    fi
    
    echo "Running CLEVR training scripts..."
    docker exec "$CONTAINER_ID" python3 training/train_clevr.py
}

# Function to run SHAPES inference
shapes_inference() {
    start_container
    
    echo "Running SHAPES inference..."
    docker exec "$CONTAINER_ID" python3 inference/shape_inference.py
}

# Function to run CLEVR inference
clevr_inference() {
    if [ -z "$2" ]; then
        echo "Error: Please provide an image path."
        exit 1
    fi
    IMAGE_PATH=$2

    start_docker_compose
    
    echo "Finding running container ID..."
    CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "Error: Could not find running container."
        exit 1
    fi
    
    echo "Running CLEVR inference on $IMAGE_PATH..."
    docker exec "$CONTAINER_ID" python3 inference/clevr_inference.py --image "$IMAGE_PATH"
}

# Function to evaluate SHAPES model
eval_shapes() {
    start_container
    
    echo "Running SHAPES evaluation..."
    docker exec "$CONTAINER_ID" python3 evaluation/shapes_eval.py
}

# Function to evaluate CLEVR model
eval_clevr() {
    start_docker_compose
    
    echo "Finding running container ID..."
    CONTAINER_ID=$(docker ps -q --filter "name=neuro-al")
    
    if [ -z "$CONTAINER_ID" ]; then
        echo "Error: Could not find running container."
        exit 1
    fi
    
    echo "Running CLEVR evaluation..."
    docker exec "$CONTAINER_ID" python3 evaluation/eval_clevr.py
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
    --eval_shapes)
        eval_shapes
        ;;
    --eval_clevr)
        eval_clevr
        ;;
    --shapes)
        shapes_inference
        ;;
    --clevr)
        clevr_inference "$@"
        ;;
    --cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 --setup | --train_shapes | --train_clevr | --eval_shapes | --eval_clevr | --shapes | --clevr <image_path> | --cleanup"
        exit 1
        ;;
esac
