# Strawberry Semantic Segmentation

A comprehensive pipeline for strawberry semantic segmentation using deep learning, with support for training, evaluation, and model export to ONNX and TensorRT formats.

## Features

- **Multi-class segmentation** for strawberries, leaves, flowers, stems, soil, containers, and background
- **Data augmentation** with Albumentations library
- **Multiple model architectures** (DeepLabV3+, UNet, etc.)
- **Training monitoring** with TensorBoard and Weights & Biases
- **Model export** to ONNX and TensorRT for deployment
- **Docker support** for reproducible environments
- **GPU acceleration** with CUDA support

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Docker (optional, for containerized execution)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd seg-det
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   - Place your dataset in the path specified in `config/config.yaml`
   - Ensure COCO format annotations are present

4. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Docker Support

### Prerequisites for Docker

1. **Install Docker:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo usermod -aG docker $USER
   
   # Log out and back in, or run:
   newgrp docker
   ```

2. **Install NVIDIA Docker Runtime (for GPU support):**
   ```bash
   # Add NVIDIA repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install NVIDIA Docker
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Using Docker

#### Quick Start with Docker

1. **Build the production image:**
   ```bash
   ./docker-build.sh build-prod
   ```

2. **Run the container:**
   ```bash
   ./docker-build.sh run-prod
   ```

#### Development with Docker

1. **Build development image:**
   ```bash
   ./docker-build.sh build-dev
   ```

2. **Run interactive development container:**
   ```bash
   ./docker-build.sh run-dev
   ```

#### Using Docker Compose

1. **Start the service:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f seg-det
   ```

3. **Stop the service:**
   ```bash
   docker-compose down
   ```

#### Custom Commands

Run the container with custom commands:

```bash
# Run training only
./docker-build.sh run-custom 'python main.py --skip-export'

# Run evaluation only
./docker-build.sh run-custom 'python main.py --skip-training --skip-export'

# Interactive shell
./docker-build.sh run-custom '/bin/bash'
```

### Docker Commands Reference

| Command | Description |
|---------|-------------|
| `./docker-build.sh build-prod` | Build production Docker image |
| `./docker-build.sh build-dev` | Build development Docker image |
| `./docker-build.sh run-prod` | Run production container |
| `./docker-build.sh run-dev` | Run interactive development container |
| `./docker-build.sh run-compose` | Run with docker-compose |
| `./docker-build.sh run-custom <cmd>` | Run with custom command |
| `./docker-build.sh cleanup` | Clean up Docker images |
| `./docker-build.sh check` | Check Docker environment |

### Volume Mounts

The Docker container mounts the following directories:

- `./data` → `/app/data` (dataset, read-only)
- `./models` → `/app/models` (model outputs)
- `./logs` → `/app/logs` (training logs)
- `./results` → `/app/results` (evaluation results)
- `./exports` → `/app/exports` (exported models)
- `./config` → `/app/config` (configuration files, read-only)

### Environment Variables

The container supports the following environment variables:

- `NVIDIA_VISIBLE_DEVICES`: GPU devices to use (default: all)
- `NVIDIA_DRIVER_CAPABILITIES`: GPU capabilities (default: compute,utility)

## NVIDIA Driver, CUDA, cuDNN, and TensorRT Installation (Ubuntu 22.04)

### System Requirements

- **OS**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **GPU**: NVIDIA GPU with compute capability 6.0+
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **Python**: 3.9+ (recommended)

### Prerequisites

Before installing CUDA, cuDNN, and TensorRT, ensure you have:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install build-essential cmake pkg-config -y

# Install Python development tools
sudo apt install python3-dev python3-pip python3-venv -y

# Install additional dependencies
sudo apt install wget curl git -y

# Install graphics libraries (for OpenCV)
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev -y
```

### 1. NVIDIA Driver Installation

#### Check GPU Compatibility
```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Check GPU model and compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

#### Check Current Driver
```bash
nvidia-smi
```

#### Install/Update NVIDIA Driver
```bash
# Method 1: Using Ubuntu's package manager (recommended for Ubuntu 22.04)
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot

# Method 2: Using NVIDIA's official driver (alternative)
# Download from: https://www.nvidia.com/Download/index.aspx
# sudo sh NVIDIA-Linux-x86_64-535.xx.xx.run

# Method 3: Using Ubuntu's additional drivers (GUI method)
# System Settings → Software & Updates → Additional Drivers
```

#### Verify Driver Installation
```bash
# Check driver version
nvidia-smi

# Check CUDA driver compatibility
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv

# Test GPU functionality
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 2. CUDA 11.8 Installation

#### Download CUDA 11.8
```bash
# Create installation directory
mkdir -p ~/cuda-install
cd ~/cuda-install

# Download CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```

#### Install CUDA 11.8
```bash
# Make installer executable
chmod +x cuda_11.8.0_520.61.05_linux.run

# Run installer
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**During installation:**
- Accept the license agreement
- Choose "Custom" installation
- Uncheck "Driver" (since we already installed it)
- Check "CUDA Toolkit" and "CUDA Samples"
- Install to default location: `/usr/local/cuda-11.8`

#### Configure Environment Variables
```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc
```

#### Verify CUDA Installation
```bash
nvcc --version
nvidia-smi
```

### 3. cuDNN 8.6.0 Installation

#### Download cuDNN 8.6.0
1. Visit [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn)
2. Sign in to your NVIDIA account (free)
3. Download: `cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz`

#### Install cuDNN
```bash
# Navigate to download directory
cd ~/Downloads  # or wherever you downloaded the file

# Extract cuDNN
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz

# Copy files to CUDA directory
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.8/lib64

# Set permissions
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

# Update library cache
sudo ldconfig
```

#### Verify cuDNN Installation
```bash
# Check cuDNN version
cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Test with PyTorch (after installing PyTorch)
python3 -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
```

### 4. TensorRT 8.6.1 Installation

#### Download TensorRT 8.6.1
1. Visit [NVIDIA TensorRT Downloads](https://developer.nvidia.com/tensorrt)
2. Sign in to your NVIDIA account
3. Download: `TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz`

#### Install TensorRT
```bash
# Navigate to download directory
cd ~/Downloads  # or wherever you downloaded the file

# Extract TensorRT
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
cd TensorRT-8.6.1.6

# Set environment variables
export TENSORRT_DIR=$(pwd)
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
export PATH=$TENSORRT_DIR/bin:$PATH

# Install Python packages (choose the correct Python version)
cd python
pip install ./tensorrt-8.6.1-cp310-none-linux_x86_64.whl
pip install ./tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl
pip install ./tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl

# Move to system directory (optional)
sudo mv $TENSORRT_DIR /usr/local/tensorrt

# Add to PATH permanently (add to ~/.bashrc)
echo 'export TENSORRT_ROOT=/usr/local/tensorrt' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/tensorrt/bin:$PATH' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc
```

#### Verify TensorRT Installation
```bash
# Check TensorRT version
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"

# Test TensorRT functionality
python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print('TensorRT installation successful')
"
```

#### Available Python Versions
TensorRT 8.6.1 supports multiple Python versions. Choose the appropriate wheel file:
- `tensorrt-8.6.1-cp36-none-linux_x86_64.whl` (Python 3.6)
- `tensorrt-8.6.1-cp37-none-linux_x86_64.whl` (Python 3.7)
- `tensorrt-8.6.1-cp38-none-linux_x86_64.whl` (Python 3.8)
- `tensorrt-8.6.1-cp39-none-linux_x86_64.whl` (Python 3.9)
- `tensorrt-8.6.1-cp310-none-linux_x86_64.whl` (Python 3.10)
- `tensorrt-8.6.1-cp311-none-linux_x86_64.whl` (Python 3.11)

### 5. Environment Setup

Create a complete environment configuration:

```bash
# Create environment setup script
cat > ~/setup_cuda_env.sh << 'EOF'
#!/bin/bash

# CUDA 11.8 Environment Variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TensorRT Environment Variables
export TENSORRT_ROOT=/usr/local/tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

# PyTorch CUDA Environment
export CUDA_VISIBLE_DEVICES=0

# Performance optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo "CUDA environment variables set"
echo "CUDA_HOME: $CUDA_HOME"
echo "TENSORRT_ROOT: $TENSORRT_ROOT"
EOF

# Make script executable
chmod +x ~/setup_cuda_env.sh

# Add to ~/.bashrc for automatic loading
echo 'source ~/setup_cuda_env.sh' >> ~/.bashrc

# Apply environment variables
source ~/setup_cuda_env.sh
```

### 6. Complete Verification

After installing all components, run these verification commands:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
"

# Check TensorRT
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"

# Test GPU computation
python3 -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print('✓ GPU computation successful')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('✗ CUDA not available')
"

# Test CUDA samples (optional)
cd /usr/local/cuda-11.8/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

### 7. PyTorch Installation with CUDA Support

After installing CUDA, cuDNN, and TensorRT, install PyTorch with CUDA support:

```bash
# Create virtual environment (recommended)
python3 -m venv ~/pytorch_env
source ~/pytorch_env/bin/activate

# Install PyTorch with CUDA 11.8 support
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies for this project
pip3 install -r requirements.txt

# Verify PyTorch CUDA installation
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### 8. Troubleshooting

#### Common Issues and Solutions

**Issue: CUDA not found**
```bash
# Check if CUDA is in PATH
echo $PATH | grep cuda

# If not, add to ~/.bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Issue: cuDNN not found**
```bash
# Check cuDNN installation
ls /usr/local/cuda-11.8/include/cudnn*.h
ls /usr/local/cuda-11.8/lib64/libcudnn*

# Update library cache
sudo ldconfig

# Verify cuDNN version
cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

**Issue: TensorRT import error**
```bash
# Check TensorRT installation
ls /usr/local/tensorrt/lib/

# Add to LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Reinstall TensorRT Python package
pip3 uninstall tensorrt -y
pip3 install /usr/local/tensorrt/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```

**Issue: Permission denied**
```bash
# Fix permissions
sudo chmod 755 /usr/local/cuda-11.8
sudo chmod 755 /usr/local/tensorrt
sudo chown -R $USER:$USER /usr/local/cuda-11.8
sudo chown -R $USER:$USER /usr/local/tensorrt
```

**Issue: PyTorch CUDA not available**
```bash
# Reinstall PyTorch with correct CUDA version
pip3 uninstall torch torchvision torchaudio -y
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Issue: Out of memory errors**
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in training configuration
# Set environment variable for memory growth
export CUDA_LAUNCH_BLOCKING=1
```

### 9. Alternative Installation Methods

#### Using Conda (Recommended for Development)
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create -n pytorch python=3.9
conda activate pytorch

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TensorRT via conda
conda install tensorrt -c nvidia
```

#### Using APT (Ubuntu Package Manager)
```bash
# Install CUDA toolkit via apt
sudo apt install nvidia-cuda-toolkit

# Install cuDNN via apt
sudo apt install libcudnn8

# Note: TensorRT is not available via apt for Ubuntu 22.04
```

### 10. Performance Optimization

#### Enable cuDNN Benchmarking
```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

#### Set GPU Memory Growth
```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### 11. Complete Installation Verification Script

Create a comprehensive verification script:

```bash
# Create verification script
cat > ~/verify_cuda_installation.sh << 'EOF'
#!/bin/bash

echo "=== CUDA Installation Verification ==="
echo

echo "1. Checking NVIDIA Driver..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found"
fi
echo

echo "2. Checking CUDA Installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA version: $(nvcc --version | grep 'release' | awk '{print $6}')"
else
    echo "✗ nvcc not found"
fi
echo

echo "3. Checking cuDNN Installation..."
if [ -f "/usr/local/cuda-11.8/include/cudnn_version.h" ]; then
    echo "✓ cuDNN version: $(cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 | grep -E 'CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL' | tr '\n' ' ' | sed 's/#define //g' | sed 's/ /./g')"
else
    echo "✗ cuDNN not found"
fi
echo

echo "4. Checking PyTorch CUDA Support..."
python3 -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ cuDNN version: {torch.backends.cudnn.version()}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('✗ CUDA not available in PyTorch')
"
echo

echo "5. Checking TensorRT Installation..."
python3 -c "
try:
    import tensorrt as trt
    print(f'✓ TensorRT version: {trt.__version__}')
except ImportError:
    print('✗ TensorRT not found')
"
echo

echo "6. Testing GPU Computation..."
python3 -c "
import torch
if torch.cuda.is_available():
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x)
        print('✓ GPU computation successful')
    except Exception as e:
        print(f'✗ GPU computation failed: {e}')
else:
    print('✗ CUDA not available for testing')
"
echo

echo "=== Verification Complete ==="
EOF

# Make script executable
chmod +x ~/verify_cuda_installation.sh

# Run verification
~/verify_cuda_installation.sh
```

### References

- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [CUDA 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN Downloads](https://developer.nvidia.com/cudnn)
- [TensorRT Downloads](https://developer.nvidia.com/tensorrt)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Ubuntu 22.04 CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## Configuration

The project uses YAML configuration files. The main configuration is in `config/config.yaml`:

```yaml
# Dataset Configuration
dataset:
  root_path: "/path/to/your/dataset"
  num_classes: 7
  class_names: ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Leaf Spot", "Powdery Mildew Fruit", "Powdery Mildew Leaf"]

# Model Configuration
model:
  name: "deeplabv3plus"
  encoder: "resnet50"
  encoder_weights: "imagenet"

# Training Configuration
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
```

## Usage

### Training

```bash
# Full pipeline
python main.py

# Training only
python main.py --skip-export

# Custom config
python main.py --config config/my_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python main.py --skip-training --skip-export
```

### Export

```bash
# Export to ONNX and TensorRT
python main.py --skip-training
```

### Using the Pipeline Script

```bash
# Install dependencies
./run_pipeline.sh install

# Run full pipeline
./run_pipeline.sh full

# Run training only
./run_pipeline.sh train

# Check environment
./run_pipeline.sh check
```

## Project Structure

```
seg-det/
├── config/
│   └── config.yaml          # Main configuration
├── src/
│   ├── data/
│   │   └── dataset.py       # Dataset and data loading
│   ├── models/
│   │   └── model_factory.py # Model creation
│   ├── training/
│   │   └── trainer.py       # Training logic
│   ├── export/
│   │   ├── onnx_export.py  # ONNX export
│   │   └── tensorrt_export.py # TensorRT export
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       └── visualization.py # Visualization utilities
├── models/                  # Trained models
├── logs/                    # Training logs
├── results/                 # Evaluation results
├── exports/                 # Exported models
├── main.py                  # Main pipeline
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── docker-build.sh         # Docker build script
└── run_pipeline.sh         # Pipeline runner script
```

## Model Export

The pipeline supports exporting trained models to:

1. **ONNX Format**: For cross-platform deployment
2. **TensorRT Format**: For optimized GPU inference
3. **Jetson Format**: For NVIDIA Jetson Xavier AGX and Orin deployment

### ONNX Export

```python
from src.export.onnx_export import export_model_to_onnx

export_model_to_onnx(
    model=model,
    output_path="models/model.onnx",
    input_shape=(1, 3, 512, 512),
    validate=True
)
```

### TensorRT Export

```python
from src.export.tensorrt_export import export_onnx_to_tensorrt

export_onnx_to_tensorrt(
    onnx_path="models/model.onnx",
    output_path="models/model.trt",
    fp16=True,
    dynamic_batch=True
)
```

### Jetson Export

```bash
# Export for Jetson Xavier AGX
python jetson_deploy.py --model models/best_metric.pth --platform xavier_agx

# Export for Jetson Orin
python jetson_deploy.py --model models/best_metric.pth --platform orin

# Auto-detect platform
python jetson_deploy.py --model models/best_metric.pth
```

The Jetson export creates optimized models for:
- **Xavier AGX**: 512MB workspace, FP16 precision, batch size 1
- **Orin**: 1GB workspace, FP16 precision, batch size 4

For detailed Jetson deployment instructions, see [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md).

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs

# Access at http://localhost:6006
```

### Weights & Biases

Enable W&B in the configuration:

```yaml
logging:
  wandb: true
  wandb_project: "strawberry-segmentation"
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Dataset not found**: Check dataset path in config
3. **TensorRT installation issues**: Ensure CUDA version compatibility
4. **Docker GPU issues**: Install NVIDIA Docker runtime

### CUDA Debugging

If you encounter CUDA errors during training, enable detailed debugging:

```bash
# Enable CUDA launch blocking for detailed error messages
export CUDA_LAUNCH_BLOCKING=1
./run_pipeline.sh train
```

This will provide more precise error locations and help identify the source of CUDA assertion failures.

### GPU Memory Optimization

```yaml
training:
  batch_size: 4  # Reduce batch size
  mixed_precision: true  # Enable mixed precision
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [PyTorch](https://pytorch.org/)
- [TensorRT](https://developer.nvidia.com/tensorrt) 