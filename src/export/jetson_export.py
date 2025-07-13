"""
Jetson-specific export utilities for Xavier AGX and Orin deployment.
This module provides optimized export functions for Jetson devices.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from typing import Dict, Tuple, Optional, List
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime as ort
from pathlib import Path


class JetsonExporter:
    """
    Export models optimized for Jetson Xavier AGX and Orin deployment.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Jetson exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        
        # Jetson-specific configurations
        self.jetson_configs = {
            'xavier_agx': {
                'max_workspace_size': 1 << 29,  # 512MB
                'precision': 'fp16',
                'max_batch_size': 1,
                'platform': 'xavier_agx'
            },
            'orin': {
                'max_workspace_size': 1 << 30,  # 1GB
                'precision': 'fp16',
                'max_batch_size': 4,
                'platform': 'orin'
            }
        }
    
    def detect_jetson_platform(self) -> str:
        """
        Detect the Jetson platform.
        
        Returns:
            Platform name ('xavier_agx' or 'orin')
        """
        try:
            # Check for Orin-specific features
            if os.path.exists('/sys/devices/gpu.0/devfreq/17000000.gv11b'):
                return 'orin'
            # Check for Xavier-specific features
            elif os.path.exists('/sys/devices/gpu.0/devfreq/17000000.gp10b'):
                return 'xavier_agx'
            else:
                # Fallback based on GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory >= 8 * 1024**3:  # 8GB or more
                    return 'orin'
                else:
                    return 'xavier_agx'
        except:
            return 'xavier_agx'  # Default fallback
    
    def optimize_onnx_for_jetson(
        self,
        onnx_path: str,
        output_path: str,
        platform: str = None
    ) -> str:
        """
        Optimize ONNX model for Jetson deployment.
        
        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to save optimized ONNX model
            platform: Jetson platform ('xavier_agx' or 'orin')
            
        Returns:
            Path to optimized ONNX model
        """
        if platform is None:
            platform = self.detect_jetson_platform()
        
        print(f"Optimizing ONNX model for Jetson {platform}...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Apply optimizations
        try:
            import onnxoptimizer
            optimized_model = onnxoptimizer.optimize(onnx_model)
        except ImportError:
            print("Warning: onnxoptimizer not available, skipping optimization")
            optimized_model = onnx_model
        
        # Simplify model
        try:
            import onnxsim
            simplified_model, check = onnxsim.simplify(optimized_model)
            if check:
                optimized_model = simplified_model
                print("✓ ONNX model simplified successfully")
            else:
                print("Warning: ONNX simplification failed")
        except ImportError:
            print("Warning: onnxsim not available, skipping simplification")
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        print(f"Optimized ONNX model saved to: {output_path}")
        
        return output_path
    
    def build_jetson_tensorrt_engine(
        self,
        onnx_path: str,
        output_path: str,
        platform: str = None,
        precision: str = 'fp16',
        max_batch_size: int = None
    ) -> str:
        """
        Build TensorRT engine optimized for Jetson.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            platform: Jetson platform
            precision: Precision mode ('fp16', 'fp32', 'int8')
            max_batch_size: Maximum batch size
            
        Returns:
            Path to TensorRT engine
        """
        if platform is None:
            platform = self.detect_jetson_platform()
        
        config = self.jetson_configs[platform]
        if max_batch_size is None:
            max_batch_size = config['max_batch_size']
        
        print(f"Building TensorRT engine for Jetson {platform}...")
        print(f"Precision: {precision}, Max batch size: {max_batch_size}")
        
        # Create network definition
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Parse ONNX model
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        builder_config = self.builder.create_builder_config()
        builder_config.max_workspace_size = config['max_workspace_size']
        
        # Set precision
        if precision == 'int8':
            if self.builder.platform_has_fast_int8:
                builder_config.set_flag(trt.BuilderFlag.INT8)
                builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                print("✓ INT8 precision enabled")
            else:
                print("Warning: INT8 not supported, falling back to FP16")
                precision = 'fp16'
        
        if precision == 'fp16':
            if self.builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
                print("✓ FP16 precision enabled")
            else:
                print("Warning: FP16 not supported, using FP32")
                precision = 'fp32'
        
        # Build engine
        print("Building TensorRT engine...")
        engine = self.builder.build_engine(network, builder_config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {output_path}")
        return output_path
    
    def benchmark_jetson_performance(
        self,
        engine_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Benchmark performance on Jetson device.
        
        Args:
            engine_path: Path to TensorRT engine
            input_shape: Input tensor shape
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings = self._allocate_buffers(engine, input_shape[0])
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup_iterations):
            self._do_inference(context, inputs, outputs, bindings, dummy_input)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            self._do_inference(context, inputs, outputs, bindings, dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput
        fps = 1.0 / avg_time
        
        results = {
            'avg_latency_ms': avg_time * 1000,
            'std_latency_ms': std_time * 1000,
            'min_latency_ms': min_time * 1000,
            'max_latency_ms': max_time * 1000,
            'fps': fps,
            'throughput': fps * input_shape[0],
            'input_shape': input_shape,
            'num_iterations': num_iterations
        }
        
        print(f"Benchmark Results:")
        print(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        print(f"  Throughput: {results['throughput']:.2f} images/sec")
        
        return results
    
    def _allocate_buffers(
        self,
        engine: trt.ICudaEngine,
        batch_size: int = 1
    ) -> Tuple[List, List, List]:
        """Allocate GPU buffers for input and output."""
        inputs = []
        outputs = []
        bindings = []
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings
    
    def _do_inference(
        self,
        context: trt.IExecutionContext,
        inputs: List,
        outputs: List,
        bindings: List,
        input_data: np.ndarray
    ) -> List[np.ndarray]:
        """Perform inference with TensorRT engine."""
        # Copy input data to GPU
        np.copyto(inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(inputs[0]['device'], inputs[0]['host'])
        
        # Run inference
        context.execute_v2(bindings=bindings)
        
        # Copy outputs from GPU
        for out in outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
        
        return [out['host'] for out in outputs]
    
    def create_jetson_deployment_package(
        self,
        model_path: str,
        output_dir: str,
        platform: str = None
    ) -> Dict:
        """
        Create a complete deployment package for Jetson.
        
        Args:
            model_path: Path to PyTorch model
            output_dir: Output directory for deployment package
            platform: Jetson platform
            
        Returns:
            Dictionary containing deployment package info
        """
        if platform is None:
            platform = self.detect_jetson_platform()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating Jetson {platform} deployment package...")
        
        # Load PyTorch model
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model_state_dict' in model:
            model = model['model_state_dict']
        
        # Export to ONNX
        onnx_path = output_dir / f"model_{platform}.onnx"
        self._export_to_onnx(model, str(onnx_path))
        
        # Optimize ONNX for Jetson
        optimized_onnx_path = output_dir / f"model_{platform}_optimized.onnx"
        self.optimize_onnx_for_jetson(str(onnx_path), str(optimized_onnx_path), platform)
        
        # Build TensorRT engine
        trt_path = output_dir / f"model_{platform}.trt"
        self.build_jetson_tensorrt_engine(str(optimized_onnx_path), str(trt_path), platform)
        
        # Benchmark performance
        benchmark_results = self.benchmark_jetson_performance(str(trt_path))
        
        # Create deployment info
        deployment_info = {
            'platform': platform,
            'model_files': {
                'onnx': str(onnx_path),
                'onnx_optimized': str(optimized_onnx_path),
                'tensorrt': str(trt_path)
            },
            'benchmark_results': benchmark_results,
            'deployment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        # Save deployment info
        info_path = output_dir / f"deployment_info_{platform}.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"Deployment package created in: {output_dir}")
        return deployment_info
    
    def _export_to_onnx(self, model: nn.Module, output_path: str):
        """Export PyTorch model to ONNX."""
        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"Model exported to ONNX: {output_path}")


def export_for_jetson(
    model: nn.Module,
    config: Dict,
    output_dir: str,
    platform: str = None
) -> Dict:
    """
    Export model for Jetson deployment.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        output_dir: Output directory
        platform: Jetson platform ('xavier_agx' or 'orin')
        
    Returns:
        Dictionary containing deployment package info
    """
    exporter = JetsonExporter(config)
    return exporter.create_jetson_deployment_package(model, output_dir, platform) 