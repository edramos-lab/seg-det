# Jetson Deployment Guide

This guide provides comprehensive instructions for deploying the strawberry semantic segmentation model on NVIDIA Jetson devices (Xavier AGX and Orin).

## Supported Jetson Platforms

### Jetson Xavier AGX
- **GPU**: Volta GV11B with 512 CUDA cores
- **Memory**: 32GB LPDDR4x
- **GPU Memory**: 8GB or 16GB
- **Power**: 10W, 15W, or 30W modes
- **Recommended**: 30W mode for best performance

### Jetson Orin
- **GPU**: Ampere GA10B with 2048 CUDA cores
- **Memory**: 32GB LPDDR5
- **GPU Memory**: 8GB or 16GB
- **Power**: 15W, 30W, or 60W modes
- **Recommended**: 60W mode for best performance

## Prerequisites

### 1. Jetson Device Setup

#### Flash Jetson with JetPack
```bash
# Download JetPack SDK from NVIDIA Developer website
# https://developer.nvidia.com/embedded/jetpack

# For Xavier AGX: Use JetPack 5.0 or later
# For Orin: Use JetPack 5.1 or later

# Follow the flashing guide for your specific device
```

#### Install JetPack Components
```bash
# Install CUDA, cuDNN, and TensorRT
sudo apt update
sudo apt install nvidia-cuda-toolkit
sudo apt install libcudnn8
sudo apt install python3-tensorrt

# Verify installations
nvcc --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

### 2. Development Machine Setup

#### Install Jetson Dependencies
```bash
# Install additional dependencies for Jetson deployment
pip install pycuda>=2022.1
pip install onnx-graphsurgeon>=0.3.12
pip install onnx-tf>=1.10.0
pip install tensorflow>=2.10.0
pip install tf2onnx>=1.12.0
pip install onnxsim>=0.4.0
pip install onnxoptimizer>=0.3.13
```

## Model Export for Jetson

### 1. Export from Development Machine

After training your model, export it for Jetson deployment:

```bash
# Export for auto-detected platform
python jetson_deploy.py --model models/best_metric.pth

# Export for specific platform
python jetson_deploy.py --model models/best_metric.pth --platform xavier_agx
python jetson_deploy.py --model models/best_metric.pth --platform orin

# Export with custom output directory
python jetson_deploy.py --model models/best_metric.pth --output-dir exports/jetson_xavier
```

### 2. Export Options

#### Platform-Specific Optimizations

**Xavier AGX:**
- Workspace size: 512MB
- Precision: FP16 (recommended)
- Max batch size: 1
- Memory optimization: Aggressive

**Orin:**
- Workspace size: 1GB
- Precision: FP16 (recommended)
- Max batch size: 4
- Memory optimization: Balanced

#### Precision Options
```bash
# FP32 (highest accuracy, slower)
python jetson_deploy.py --model models/best_metric.pth --precision fp32

# FP16 (recommended, good accuracy, faster)
python jetson_deploy.py --model models/best_metric.pth --precision fp16

# INT8 (fastest, may reduce accuracy)
python jetson_deploy.py --model models/best_metric.pth --precision int8
```

## Deployment Package Structure

The export creates the following files:

```
exports/jetson/
├── model_xavier_agx.onnx              # Original ONNX model
├── model_xavier_agx_optimized.onnx    # Optimized ONNX model
├── model_xavier_agx.trt               # TensorRT engine
├── deployment_info_xavier_agx.json    # Deployment metadata
└── README_deployment.md               # Deployment instructions
```

## Jetson Deployment

### 1. Transfer Files to Jetson

```bash
# Copy deployment package to Jetson
scp -r exports/jetson/ jetson@192.168.1.100:/home/jetson/deployment/

# Or use USB/SD card transfer
```

### 2. Jetson Runtime Setup

#### Install Python Dependencies
```bash
# On Jetson device
sudo apt update
sudo apt install python3-pip python3-dev

# Install ONNX Runtime for Jetson
pip3 install onnxruntime-gpu

# Install additional dependencies
pip3 install numpy opencv-python pillow
```

#### Set Performance Mode
```bash
# Set to maximum performance mode
sudo nvpmodel -m 0  # Xavier AGX: 30W mode
sudo nvpmodel -m 0  # Orin: 60W mode

# Verify power mode
sudo nvpmodel -q
```

### 3. Run Inference on Jetson

#### Using TensorRT Engine
```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2

# Load TensorRT engine
with open('model_xavier_agx.trt', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, bindings = allocate_buffers(engine)

# Load and preprocess image
image = cv2.imread('strawberry.jpg')
image = cv2.resize(image, (512, 512))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

# Run inference
do_inference(context, inputs, outputs, bindings, image)

# Process output
output = outputs[0]['host'].reshape(1, 8, 512, 512)
segmentation = np.argmax(output[0], axis=0)
```

#### Using ONNX Runtime
```python
import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession('model_xavier_agx_optimized.onnx')

# Load and preprocess image
image = cv2.imread('strawberry.jpg')
image = cv2.resize(image, (512, 512))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image})[0]

# Process output
segmentation = np.argmax(output[0], axis=0)
```

## Performance Optimization

### 1. Jetson-Specific Optimizations

#### Memory Management
```python
# Set GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### TensorRT Optimizations
```python
# Enable TensorRT optimizations
import tensorrt as trt
config = builder.create_builder_config()
config.max_workspace_size = 1 << 29  # 512MB for Xavier
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.STRICT_TYPES)
```

### 2. Power Management

#### Xavier AGX Power Modes
```bash
# 10W mode (lowest power)
sudo nvpmodel -m 1

# 15W mode (balanced)
sudo nvpmodel -m 2

# 30W mode (maximum performance)
sudo nvpmodel -m 0
```

#### Orin Power Modes
```bash
# 15W mode (lowest power)
sudo nvpmodel -m 1

# 30W mode (balanced)
sudo nvpmodel -m 2

# 60W mode (maximum performance)
sudo nvpmodel -m 0
```

### 3. Thermal Management
```bash
# Monitor temperature
tegrastats

# Set fan speed (if available)
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

## Benchmarking

### 1. Performance Benchmarking
```bash
# Run built-in benchmark
python jetson_deploy.py --model models/best_metric.pth --benchmark

# Custom benchmark script
python benchmark_jetson.py --engine model_xavier_agx.trt --iterations 1000
```

### 2. Expected Performance

#### Xavier AGX (30W mode)
- **FP32**: ~5-8 FPS
- **FP16**: ~10-15 FPS
- **INT8**: ~15-20 FPS

#### Orin (60W mode)
- **FP32**: ~15-25 FPS
- **FP16**: ~25-40 FPS
- **INT8**: ~40-60 FPS

## Troubleshooting

### 1. Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
# Use smaller input resolution
# Enable memory growth
export CUDA_LAUNCH_BLOCKING=1
```

#### TensorRT Build Failures
```bash
# Check TensorRT version compatibility
python3 -c "import tensorrt as trt; print(trt.__version__)"

# Rebuild with different precision
python jetson_deploy.py --precision fp32
```

#### ONNX Parsing Errors
```bash
# Simplify ONNX model
pip install onnxsim
python -c "import onnxsim; onnxsim.simplify('model.onnx')"
```

### 2. Performance Issues

#### Low FPS
- Check power mode: `sudo nvpmodel -q`
- Monitor temperature: `tegrastats`
- Reduce input resolution
- Use FP16 precision

#### High Latency
- Enable TensorRT optimizations
- Use batch inference
- Optimize input preprocessing

### 3. Memory Issues

#### GPU Memory
```bash
# Monitor GPU memory
nvidia-smi

# Clear GPU cache
sudo rm -rf /tmp/tensorrt*
```

#### System Memory
```bash
# Monitor system memory
free -h

# Clear system cache
sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
```

## Advanced Deployment

### 1. Docker Deployment
```dockerfile
# Dockerfile for Jetson deployment
FROM nvcr.io/nvidia/l4t-base:r35.2.1

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy deployment files
COPY exports/jetson/ /app/
COPY jetson_inference.py /app/

# Set working directory
WORKDIR /app

# Run inference
CMD ["python3", "jetson_inference.py"]
```

### 2. ROS2 Integration
```python
# ROS2 node for real-time inference
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, 'segmentation/result', 10)
        self.bridge = CvBridge()
        
        # Load TensorRT engine
        self.engine = load_tensorrt_engine('model_xavier_agx.trt')
    
    def image_callback(self, msg):
        # Process image and publish segmentation result
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        segmentation = self.inference(cv_image)
        result_msg = self.bridge.cv2_to_imgmsg(segmentation, "mono8")
        self.publisher.publish(result_msg)
```

### 3. Multi-Stream Processing
```python
# Multi-stream inference for higher throughput
import threading
import queue

class MultiStreamInference:
    def __init__(self, engine_path, num_streams=4):
        self.engines = [load_tensorrt_engine(engine_path) for _ in range(num_streams)]
        self.queues = [queue.Queue() for _ in range(num_streams)]
        self.threads = []
        
        for i in range(num_streams):
            thread = threading.Thread(target=self._inference_worker, args=(i,))
            thread.start()
            self.threads.append(thread)
    
    def _inference_worker(self, stream_id):
        while True:
            try:
                image = self.queues[stream_id].get(timeout=1)
                result = self.inference(image, self.engines[stream_id])
                # Process result
            except queue.Empty:
                continue
```

## Monitoring and Logging

### 1. Performance Monitoring
```python
import time
import psutil
import GPUtil

class JetsonMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def get_performance_metrics(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # GPU usage
        gpus = GPUtil.getGPUs()
        gpu_util = gpus[0].load * 100 if gpus else 0
        gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'gpu_utilization': gpu_util,
            'gpu_memory_percent': gpu_memory,
            'uptime': time.time() - self.start_time
        }
```

### 2. Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jetson_inference.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('jetson_inference')
```

## References

- [Jetson Xavier AGX Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit)
- [Jetson Orin Developer Kit](https://developer.nvidia.com/embedded/jetson-orin)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Jetson Performance Guide](https://docs.nvidia.com/jetson/archives/r35.2.1/DeveloperGuide/text/IN/PerformanceGuide.html) 