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


class TensorRTExporter:
    """
    Export ONNX models to TensorRT format for optimized inference.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the TensorRT exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.parser = None
        self.engine = None
        self.context = None
    
    def build_engine_from_onnx(
        self,
        onnx_path: str,
        output_path: str,
        max_batch_size: int = 1,
        fp16: bool = True,
        int8: bool = False,
        workspace_size: int = 1 << 30  # 1GB
    ) -> str:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size
            fp16: Whether to use FP16 precision
            int8: Whether to use INT8 precision
            workspace_size: Workspace size in bytes
            
        Returns:
            Path to the TensorRT engine
        """
        # Create network definition
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Parse ONNX model
        self.parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, 'rb') as model:
            if not self.parser.parse(model.read()):
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = self.builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        # Set precision
        if int8:
            if self.builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            else:
                print("Warning: INT8 not supported on this platform")
        elif fp16:
            if self.builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("Warning: FP16 not supported on this platform")
        
        # Build engine
        print("Building TensorRT engine...")
        engine = self.builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {output_path}")
        return output_path
    
    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """
        Load TensorRT engine from file.
        
        Args:
            engine_path: Path to TensorRT engine
            
        Returns:
            TensorRT engine
        """
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        return engine
    
    def create_execution_context(self, engine: trt.ICudaEngine) -> trt.IExecutionContext:
        """
        Create execution context from engine.
        
        Args:
            engine: TensorRT engine
            
        Returns:
            Execution context
        """
        context = engine.create_execution_context()
        return context
    
    def allocate_buffers(
        self,
        engine: trt.ICudaEngine,
        batch_size: int = 1
    ) -> Tuple[List, List, List]:
        """
        Allocate GPU buffers for input and output.
        
        Args:
            engine: TensorRT engine
            batch_size: Batch size
            
        Returns:
            Tuple of (inputs, outputs, bindings)
        """
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
    
    def do_inference(
        self,
        context: trt.IExecutionContext,
        inputs: List,
        outputs: List,
        bindings: List,
        input_data: np.ndarray
    ) -> List[np.ndarray]:
        """
        Perform inference with TensorRT engine.
        
        Args:
            context: Execution context
            inputs: Input buffers
            outputs: Output buffers
            bindings: Binding indices
            input_data: Input data
            
        Returns:
            List of output arrays
        """
        # Copy input data to GPU
        np.copyto(inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(inputs[0]['device'], inputs[0]['host'])
        
        # Run inference
        context.execute_v2(bindings=bindings)
        
        # Copy outputs from GPU
        for out in outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
        
        return [out['host'] for out in outputs]
    
    def benchmark_inference(
        self,
        engine_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Benchmark TensorRT inference performance.
        
        Args:
            engine_path: Path to TensorRT engine
            input_shape: Input tensor shape
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary containing benchmark results
        """
        # Load engine
        engine = self.load_engine(engine_path)
        context = self.create_execution_context(engine)
        
        # Allocate buffers
        inputs, outputs, bindings = self.allocate_buffers(engine, input_shape[0])
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(warmup_iterations):
            self.do_inference(context, inputs, outputs, bindings, dummy_input)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            self.do_inference(context, inputs, outputs, bindings, dummy_input)
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
        throughput = fps * input_shape[0]  # images per second
        
        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'min_inference_time_ms': min_time * 1000,
            'max_inference_time_ms': max_time * 1000,
            'fps': fps,
            'throughput': throughput,
            'input_shape': input_shape,
            'num_iterations': num_iterations
        }
        
        print("Benchmark Results:")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Standard deviation: {std_time * 1000:.2f} ms")
        print(f"Min/Max time: {min_time * 1000:.2f} / {max_time * 1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"Throughput: {throughput:.2f} images/sec")
        
        return results
    
    def compare_pytorch_tensorrt(
        self,
        pytorch_model: nn.Module,
        engine_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        rtol: float = 1e-3,
        atol: float = 1e-3
    ) -> bool:
        """
        Compare PyTorch and TensorRT model outputs.
        
        Args:
            pytorch_model: PyTorch model
            engine_path: Path to TensorRT engine
            input_shape: Input tensor shape
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        # Load TensorRT engine
        engine = self.load_engine(engine_path)
        context = self.create_execution_context(engine)
        inputs, outputs, bindings = self.allocate_buffers(engine, input_shape[0])
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        dummy_input_torch = torch.from_numpy(dummy_input)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input_torch).numpy()
        
        # TensorRT inference
        tensorrt_output = self.do_inference(context, inputs, outputs, bindings, dummy_input)[0]
        
        # Reshape TensorRT output to match PyTorch output
        tensorrt_output = tensorrt_output.reshape(pytorch_output.shape)
        
        # Compare outputs
        is_close = np.allclose(pytorch_output, tensorrt_output, rtol=rtol, atol=atol)
        
        if is_close:
            print("PyTorch and TensorRT outputs match within tolerance!")
            print(f"Max difference: {np.max(np.abs(pytorch_output - tensorrt_output))}")
        else:
            print("PyTorch and TensorRT outputs do not match!")
            print(f"Max difference: {np.max(np.abs(pytorch_output - tensorrt_output))}")
        
        return is_close
    
    def get_engine_info(self, engine_path: str) -> Dict:
        """
        Get information about the TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine
            
        Returns:
            Dictionary containing engine information
        """
        engine = self.load_engine(engine_path)
        
        # Get engine size
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
        
        # Get layer information
        num_layers = engine.num_layers
        
        # Get binding information
        bindings = []
        for i in range(engine.num_bindings):
            binding_info = {
                'name': engine.get_binding_name(i),
                'shape': engine.get_binding_shape(i),
                'dtype': str(engine.get_binding_dtype(i)),
                'is_input': engine.binding_is_input(i)
            }
            bindings.append(binding_info)
        
        return {
            'engine_size_mb': engine_size,
            'num_layers': num_layers,
            'bindings': bindings,
            'max_batch_size': engine.max_batch_size
        }


def export_onnx_to_tensorrt(
    onnx_path: str,
    config: Dict,
    output_path: str,
    fp16: bool = True,
    benchmark: bool = True,
    compare: bool = True,
    pytorch_model: Optional[nn.Module] = None
) -> Dict:
    """
    Export ONNX model to TensorRT format with validation and benchmarking.
    
    Args:
        onnx_path: Path to ONNX model
        config: Configuration dictionary
        output_path: Path to save TensorRT engine
        fp16: Whether to use FP16 precision
        benchmark: Whether to benchmark the engine
        compare: Whether to compare with PyTorch model
        pytorch_model: PyTorch model for comparison
        
    Returns:
        Dictionary containing export results
    """
    exporter = TensorRTExporter(config)
    
    # Build TensorRT engine
    engine_path = exporter.build_engine_from_onnx(
        onnx_path=onnx_path,
        output_path=output_path,
        fp16=fp16
    )
    
    # Get engine info
    engine_info = exporter.get_engine_info(engine_path)
    
    results = {
        'engine_path': engine_path,
        'engine_info': engine_info,
        'benchmark_results': None,
        'comparison_passed': False
    }
    
    # Benchmark engine
    if benchmark:
        results['benchmark_results'] = exporter.benchmark_inference(engine_path)
    
    # Compare with PyTorch model
    if compare and pytorch_model is not None:
        results['comparison_passed'] = exporter.compare_pytorch_tensorrt(
            pytorch_model, engine_path
        )
    
    return results 