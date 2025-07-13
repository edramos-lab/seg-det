# Strawberry Semantic Segmentation Pipeline

A comprehensive pipeline for semantic segmentation of strawberry images using state-of-the-art deep learning models. This pipeline includes data preparation, augmentation, training, evaluation, and export to ONNX and TensorRT formats.

## Features

- **State-of-the-art Models**: DeepLabV3+, UNet, UNet++, PSPNet, LinkNet, FPN, PAN
- **Comprehensive Data Augmentation**: Horizontal/vertical flips, rotations, brightness/contrast adjustments, blur, noise, elastic transformations
- **Advanced Loss Functions**: Dice-CE, Focal Loss, IoU Loss, Combined Loss
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Comprehensive Metrics**: Dice coefficient, IoU, precision, recall, accuracy, per-class metrics
- **Model Export**: ONNX and TensorRT export with validation and benchmarking
- **Visualization**: Training curves, confusion matrices, per-class metrics, prediction visualizations
- **Logging**: TensorBoard and Weights & Biases integration

## Dataset

The pipeline is designed for the Strawberry Segmentation Synthetic dataset with 8 classes:
- Background
- Strawberry
- Leaf
- Flower
- Stem
- Soil
- Container
- Other

Dataset structure:
```
/home/edramos/Documents/datasets/Strawberry-segmentation-synthetic/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd seg-det
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install TensorRT** (optional, for TensorRT export):
```bash
# Follow NVIDIA's TensorRT installation guide for your platform
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```

## NVIDIA Driver, CUDA, cuDNN, and TensorRT Installation

To use GPU acceleration and export to TensorRT, you must have the correct NVIDIA driver, CUDA toolkit, cuDNN, and TensorRT installed. Instructions below cover both RTX 3050 (laptop) and A100 (datacenter) GPUs.

### 1. NVIDIA Driver

- **RTX 3050 (Laptop):**  
  Use the latest proprietary driver from Ubuntu's Additional Drivers or from NVIDIA’s website.
  ```bash
  sudo ubuntu-drivers autoinstall
  # Or, for manual install:
  # Download from https://www.nvidia.com/Download/index.aspx
  ```

- **A100 (Datacenter):**  
  Use the latest datacenter driver from NVIDIA.
  ```bash
  # Download from https://www.nvidia.com/Download/index.aspx
  # Or use your cluster's driver management tools
  ```

- **Verify installation:**
  ```bash
  nvidia-smi
  ```

### 2. CUDA Toolkit

- **Recommended:** Use the version matching your PyTorch install (see https://pytorch.org/get-started/locally/).
- **Install via apt (Ubuntu):**
  ```bash
  # Example for CUDA 12.1
  sudo apt update
  sudo apt install nvidia-cuda-toolkit
  ```
- **Or download from:**  
  https://developer.nvidia.com/cuda-downloads

- **Check CUDA version:**
  ```bash
  nvcc --version
  ```

### 3. cuDNN

- **Download from:**  
  https://developer.nvidia.com/cudnn  
  (Requires free NVIDIA Developer account)

- **Install (Ubuntu .deb):**
  ```bash
  # Example for cuDNN 8.x
  sudo dpkg -i libcudnn8*.deb
  sudo apt-get update
  sudo apt-get install libcudnn8
  ```

- **Or, for Conda environments:**
  ```bash
  conda install -c conda-forge cudnn
  ```

### 4. TensorRT

- **Download from:**  
  https://developer.nvidia.com/tensorrt  
  (Choose the version matching your CUDA and cuDNN)

- **Install (Ubuntu .deb):**
  ```bash
  sudo dpkg -i nv-tensorrt-local-repo-ubuntu*.deb
  sudo apt-get update
  sudo apt-get install tensorrt
  ```

- **Or, for Conda environments:**
  ```bash
  conda install -c conda-forge tensorrt
  ```

- **Python bindings:**
  ```bash
  pip install tensorrt
  ```

### 5. Additional Tips

- **For laptops (RTX 3050):**  
  Use the latest drivers and CUDA toolkit supported by your GPU.  
  If using Optimus (hybrid graphics), ensure you are running on the NVIDIA GPU (`prime-select nvidia`).

- **For A100:**  
  Use the datacenter driver and CUDA version recommended by your cluster admin or NVIDIA.

- **Check all installations:**
  ```bash
  nvidia-smi
  nvcc --version
  python -c "import torch; print(torch.cuda.is_available())"
  python -c "import tensorrt"
  ```

- **References:**  
  - [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)  
  - [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)  
  - [cuDNN Downloads](https://developer.nvidia.com/cudnn)  
  - [TensorRT Downloads](https://developer.nvidia.com/tensorrt)  
  - [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

---

### Installing CUDA 11.4, cuDNN 8.6.0, and TensorRT 8.5.2 (Recommended for RTX 3050)

These versions are fully compatible with the RTX 3050 and this repository.

#### 1. Install NVIDIA Driver
Make sure you have the latest NVIDIA driver (at least 470.x for CUDA 11.4):
```bash
sudo ubuntu-drivers autoinstall
nvidia-smi
```

#### 2. Install CUDA 11.4
- Download the [CUDA 11.4 runfile (local)](https://developer.nvidia.com/cuda-11.4.0-download-archive) for Ubuntu.
- Follow the official [CUDA 11.4 installation guide](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-installation-guide-linux/index.html).

**Quick install:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
sudo sh cuda_11.4.0_470.42.01_linux.run
```
- Add CUDA to your PATH:
```bash
echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
- Verify:
```bash
nvcc --version
```

#### 3. Install cuDNN 8.6.0 for CUDA 11.4
- Download cuDNN 8.6.0 for CUDA 11.x from [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
- Extract and copy files:
```bash
tar -xzvf cudnn-linux-x86_64-8.6.0.*.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.4/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.4/lib64
sudo chmod a+r /usr/local/cuda-11.4/include/cudnn*.h /usr/local/cuda-11.4/lib64/libcudnn*
```
- Verify:
```bash
cat /usr/local/cuda-11.4/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### 4. Install TensorRT 8.5.2 for CUDA 11.4
- Download TensorRT 8.5.2 tar file for Ubuntu 20.04 + CUDA 11.x from [NVIDIA TensorRT Archive](https://developer.nvidia.com/nvidia-tensorrt-archive).
- Extract and install Python bindings:
```bash
tar -xzvf TensorRT-8.5.2.*.tar.gz
cd TensorRT-8.5.2.*
export TRT_LIBPATH=$(pwd)/lib
export LD_LIBRARY_PATH=$TRT_LIBPATH:$LD_LIBRARY_PATH
pip install python/tensorrt-8.5.2-*.whl
```
- (Optional) Install `uff`, `graphsurgeon`, and `onnx-graphsurgeon` from the TensorRT package if needed.

#### 5. Verify Everything
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorrt; print(tensorrt.__version__)"
```

**References:**
- [CUDA 11.4 Downloads](https://developer.nvidia.com/cuda-11.4.0-download-archive)
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
- [TensorRT Archive](https://developer.nvidia.com/nvidia-tensorrt-archive)

## Configuration

The pipeline is configured via `config/config.yaml`. Key configuration sections:

### Dataset Configuration
```yaml
dataset:
  root_path: "/home/edramos/Documents/datasets/Strawberry-segmentation-synthetic"
  num_classes: 8
  class_names:
    - "background"
    - "strawberry"
    - "leaf"
    - "flower"
    - "stem"
    - "soil"
    - "container"
    - "other"
```

### Model Configuration
```yaml
model:
  name: "deeplabv3plus"  # Options: deeplabv3plus, unet, unetplusplus, pspnet, linknet, fpn, pan
  encoder: "resnet50"     # Options: resnet18, resnet34, resnet50, resnet101, efficientnet-b0, etc.
  encoder_weights: "imagenet"
  classes: 8
```

### Training Configuration
```yaml
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
```

## Usage

### Full Pipeline

Run the complete pipeline (prepare → augment → train → evaluate → export):

```bash
python main.py
```

### Skip Training (Use Existing Model)

```bash
python main.py --skip-training
```

### Skip Export Steps

```bash
python main.py --skip-export
```

### Custom Configuration

```bash
python main.py --config path/to/custom/config.yaml
```

## Pipeline Steps

### 1. Prepare
- Validates dataset path and required files
- Creates output directories
- Checks COCO annotation files

### 2. Augment
- Configures data augmentation transforms
- Includes horizontal/vertical flips, rotations, brightness/contrast adjustments
- Applies Gaussian blur, noise, and elastic transformations

### 3. Train
- Creates model and dataloaders
- Trains with mixed precision
- Implements early stopping and learning rate scheduling
- Saves best models based on validation metrics
- Logs metrics to TensorBoard and Weights & Biases

### 4. Evaluate
- Evaluates model on test set
- Calculates comprehensive metrics (Dice, IoU, precision, recall, accuracy)
- Generates per-class performance visualizations
- Creates confusion matrices

### 5. Export to ONNX
- Exports PyTorch model to ONNX format
- Validates ONNX model
- Compares PyTorch and ONNX outputs
- Provides model information and size

### 6. Export to TensorRT
- Converts ONNX model to TensorRT engine
- Optimizes for FP16 precision
- Benchmarks inference performance
- Compares PyTorch and TensorRT outputs

## Output Files

After running the pipeline, you'll find:

```
seg-det/
├── models/
│   ├── best_metric.pth          # Best PyTorch model
│   ├── best_loss.pth           # Best model by loss
│   └── checkpoint_epoch_*.pth  # Training checkpoints
├── results/
│   ├── training_curves.png     # Training visualization
│   ├── class_dice_scores.png   # Per-class Dice scores
│   ├── class_iou_scores.png    # Per-class IoU scores
│   └── training_history.json   # Training metrics
├── logs/                       # TensorBoard logs
├── models/
│   ├── model.onnx             # ONNX model
│   └── model.trt              # TensorRT engine
└── exports/                    # Additional export files
```

## Model Architecture

The pipeline supports multiple state-of-the-art semantic segmentation architectures:

- **DeepLabV3+**: Excellent for precise boundary segmentation
- **UNet**: Good for medical and biological image segmentation
- **UNet++**: Enhanced UNet with nested skip connections
- **PSPNet**: Pyramid Scene Parsing Network for complex scenes
- **Linknet**: Lightweight architecture for real-time inference
- **FPN**: Feature Pyramid Network for multi-scale features
- **PAN**: Pyramid Attention Network for attention-based segmentation

## Loss Functions

- **Dice-CE Loss**: Combined Dice coefficient and Cross-entropy
- **Focal Loss**: Handles class imbalance
- **IoU Loss**: Intersection over Union optimization
- **Combined Loss**: Multi-component loss with configurable weights

## Metrics

The pipeline calculates comprehensive metrics:

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Precision & Recall**: Per-class and overall
- **Pixel Accuracy**: Overall pixel-wise accuracy
- **Per-class Metrics**: Individual class performance

## Performance Optimization

- **Mixed Precision Training**: Automatic FP16 training for faster convergence
- **Data Augmentation**: Extensive augmentation for better generalization
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **TensorRT Optimization**: FP16 inference for deployment

## Deployment

### ONNX Deployment
```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('models/model.onnx')

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data})
```

### TensorRT Deployment
```python
import tensorrt as trt
import pycuda.driver as cuda

# Load TensorRT engine
with open('models/model.trt', 'rb') as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger())
engine = runtime.deserialize_cuda_engine(engine_data)

# Create execution context
context = engine.create_execution_context()

# Run inference
# (See tensorrt_export.py for complete example)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Dataset Loading Errors**:
   - Verify COCO annotation files exist
   - Check image file paths
   - Ensure proper dataset structure

3. **TensorRT Export Failures**:
   - Verify TensorRT installation
   - Check GPU compatibility
   - Ensure ONNX model is valid

### Performance Tips

- Use larger batch sizes with more GPU memory
- Enable mixed precision training
- Use appropriate model size for your hardware
- Consider model distillation for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{strawberry_segmentation_pipeline,
  title={Strawberry Semantic Segmentation Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/seg-det}
}
```

## Acknowledgments

- Segmentation Models PyTorch for model architectures
- Albumentations for data augmentation
- NVIDIA TensorRT for optimization
- PyTorch team for the deep learning framework 