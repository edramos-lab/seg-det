#!/bin/bash

# Docker Build Script for Strawberry Semantic Segmentation
# This script provides convenient commands to build and run the Docker container

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

# Function to check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker info | grep -q "nvidia"; then
        print_warning "NVIDIA Docker runtime not detected. GPU support may not work."
        print_status "Install NVIDIA Docker runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    else
        print_success "NVIDIA Docker runtime detected"
    fi
}

# Function to build production image
build_production() {
    print_status "Building production Docker image..."
    docker build --target production -t seg-det:latest .
    print_success "Production image built successfully"
}

# Function to build development image
build_development() {
    print_status "Building development Docker image..."
    docker build --target app -t seg-det:dev .
    print_success "Development image built successfully"
}

# Function to run with docker-compose
run_compose() {
    local service=${1:-seg-det}
    print_status "Running with docker-compose (service: $service)..."
    docker-compose up -d $service
    print_success "Container started successfully"
    print_status "To view logs: docker-compose logs -f $service"
    print_status "To stop: docker-compose down"
}

# Function to run interactive development container
run_dev_interactive() {
    print_status "Starting interactive development container..."
    docker-compose run --rm seg-det-dev
}

# Function to run production container
run_production() {
    print_status "Running production container..."
    docker run --rm \
        --gpus all \
        -v $(pwd)/data:/app/data:ro \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/exports:/app/exports \
        -v $(pwd)/config:/app/config:ro \
        -p 6006:6006 \
        seg-det:latest
}

# Function to run with custom command
run_custom() {
    local cmd="$1"
    print_status "Running container with custom command: $cmd"
    docker run --rm \
        --gpus all \
        -v $(pwd)/data:/app/data:ro \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/exports:/app/exports \
        -v $(pwd)/config:/app/config:ro \
        seg-det:latest $cmd
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker images..."
    docker rmi seg-det:latest seg-det:dev 2>/dev/null || true
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "Docker Build Script for Strawberry Semantic Segmentation"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build-prod     Build production Docker image"
    echo "  build-dev      Build development Docker image"
    echo "  build-all      Build both production and development images"
    echo "  run-prod       Run production container"
    echo "  run-dev        Run development container interactively"
    echo "  run-compose    Run with docker-compose"
    echo "  run-custom     Run with custom command"
    echo "  cleanup        Clean up Docker images"
    echo "  check          Check Docker and NVIDIA Docker setup"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build-prod                    # Build production image"
    echo "  $0 run-prod                      # Run production container"
    echo "  $0 run-custom 'python main.py'   # Run with custom command"
    echo "  $0 run-custom '/bin/bash'        # Run interactive shell"
    echo ""
    echo "Prerequisites:"
    echo "  - Docker installed"
    echo "  - NVIDIA Docker runtime (for GPU support)"
    echo "  - Dataset mounted at ./data"
    echo "  - Config files in ./config"
}

# Function to check environment
check_environment() {
    print_status "Checking Docker environment..."
    check_docker
    check_nvidia_docker
    
    # Check if dataset directory exists
    if [ ! -d "./data" ]; then
        print_warning "Dataset directory ./data not found"
        print_status "Please mount your dataset at ./data"
    else
        print_success "Dataset directory found"
    fi
    
    # Check if config directory exists
    if [ ! -d "./config" ]; then
        print_error "Config directory ./config not found"
        exit 1
    else
        print_success "Config directory found"
    fi
    
    print_success "Environment check completed"
}

# Main script logic
main() {
    case "${1:-help}" in
        "build-prod")
            check_docker
            build_production
            ;;
        "build-dev")
            check_docker
            build_development
            ;;
        "build-all")
            check_docker
            build_production
            build_development
            ;;
        "run-prod")
            check_docker
            check_nvidia_docker
            run_production
            ;;
        "run-dev")
            check_docker
            check_nvidia_docker
            run_dev_interactive
            ;;
        "run-compose")
            check_docker
            check_nvidia_docker
            run_compose "$2"
            ;;
        "run-custom")
            if [[ -z "$2" ]]; then
                print_error "Please provide a custom command"
                echo "Usage: $0 run-custom <command>"
                exit 1
            fi
            check_docker
            check_nvidia_docker
            run_custom "$2"
            ;;
        "cleanup")
            cleanup
            ;;
        "check")
            check_environment
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 