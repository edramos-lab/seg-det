#!/bin/bash

# Strawberry Semantic Segmentation Pipeline Runner
# This script provides convenient commands to run different parts of the pipeline

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

# Function to check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not detected. Please activate your virtual environment first."
        print_status "You can activate it with: source venv/bin/activate"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to check if CUDA is available
check_cuda() {
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA is available"
    else
        print_warning "CUDA is not available. Training will use CPU (slow)."
    fi
}

# Function to install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies installed successfully"
}

# Function to run full pipeline
run_full_pipeline() {
    print_status "Running full pipeline..."
    python main.py
    print_success "Full pipeline completed"
}

# Function to run training only
run_training() {
    print_status "Running training only..."
    python main.py --skip-export
    print_success "Training completed"
}

# Function to run evaluation only
run_evaluation() {
    print_status "Running evaluation only..."
    python main.py --skip-training --skip-export
    print_success "Evaluation completed"
}

# Function to run export only
run_export() {
    print_status "Running export only..."
    python main.py --skip-training
    print_success "Export completed"
}

# Function to run with custom config
run_custom() {
    local config_path=$1
    print_status "Running pipeline with custom config: $config_path"
    python main.py --config "$config_path"
    print_success "Pipeline with custom config completed"
}

# Function to show help
show_help() {
    echo "Strawberry Semantic Segmentation Pipeline Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install     Install dependencies"
    echo "  full        Run full pipeline (prepare → augment → train → evaluate → export)"
    echo "  train       Run training only"
    echo "  eval        Run evaluation only (requires trained model)"
    echo "  export      Run export only (requires trained model)"
    echo "  custom      Run with custom config file"
    echo "  check       Check environment (CUDA, dependencies)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install                    # Install dependencies"
    echo "  $0 full                       # Run full pipeline"
    echo "  $0 train                      # Run training only"
    echo "  $0 custom config/my_config.yaml  # Run with custom config"
    echo ""
    echo "Environment:"
    echo "  - Make sure your virtual environment is activated"
    echo "  - Ensure dataset is available at the path specified in config"
    echo "  - For TensorRT export, ensure TensorRT is installed"
}

# Function to check environment
check_environment() {
    print_status "Checking environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1)
    print_status "Python version: $python_version"
    
    # Check PyTorch
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
        print_success "PyTorch is installed"
    else
        print_error "PyTorch is not installed"
        return 1
    fi
    
    # Check CUDA
    check_cuda
    
    # Check dataset path
    if python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataset_path = config['dataset']['root_path']
import os
if os.path.exists(dataset_path):
    print(f'Dataset found at: {dataset_path}')
else:
    print(f'Dataset not found at: {dataset_path}')
" 2>/dev/null; then
        print_success "Dataset path check completed"
    else
        print_error "Could not check dataset path"
    fi
    
    print_success "Environment check completed"
}

# Main script logic
main() {
    case "${1:-help}" in
        "install")
            check_venv
            install_deps
            ;;
        "full")
            check_venv
            check_cuda
            run_full_pipeline
            ;;
        "train")
            check_venv
            check_cuda
            run_training
            ;;
        "eval")
            check_venv
            run_evaluation
            ;;
        "export")
            check_venv
            run_export
            ;;
        "custom")
            if [[ -z "$2" ]]; then
                print_error "Please provide a config file path"
                echo "Usage: $0 custom <config_file>"
                exit 1
            fi
            check_venv
            check_cuda
            run_custom "$2"
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