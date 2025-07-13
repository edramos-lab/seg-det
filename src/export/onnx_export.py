import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, Optional
import os


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize the ONNX exporter.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.model.eval()
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        dynamic_batch: bool = True,
        opset_version: int = 11
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (batch_size, channels, height, width)
            dynamic_batch: Whether to use dynamic batch size
            opset_version: ONNX opset version
            
        Returns:
            Path to the exported ONNX model
        """
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Set dynamic axes if requested
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def validate_onnx(self, onnx_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512)) -> bool:
        """
        Validate the exported ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            input_shape: Input tensor shape
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            ort_session = ort.InferenceSession(onnx_path)
            
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            print("ONNX model validation passed!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {ort_outputs[0].shape}")
            
            return True
            
        except Exception as e:
            print(f"ONNX model validation failed: {e}")
            return False
    
    def compare_pytorch_onnx(
        self,
        onnx_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> bool:
        """
        Compare PyTorch and ONNX model outputs.
        
        Args:
            onnx_path: Path to the ONNX model
            input_shape: Input tensor shape
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if outputs match within tolerance, False otherwise
        """
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = self.model(dummy_input)
        
        # ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        pytorch_output_np = pytorch_output.numpy()
        
        # Check if shapes match
        if pytorch_output_np.shape != onnx_output.shape:
            print(f"Shape mismatch: PyTorch {pytorch_output_np.shape} vs ONNX {onnx_output.shape}")
            return False
        
        # Check if values match within tolerance
        is_close = np.allclose(pytorch_output_np, onnx_output, rtol=rtol, atol=atol)
        
        if is_close:
            print("PyTorch and ONNX outputs match within tolerance!")
            print(f"Max difference: {np.max(np.abs(pytorch_output_np - onnx_output))}")
        else:
            print("PyTorch and ONNX outputs do not match!")
            print(f"Max difference: {np.max(np.abs(pytorch_output_np - onnx_output))}")
        
        return is_close
    
    def get_onnx_model_info(self, onnx_path: str) -> Dict:
        """
        Get information about the ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            
        Returns:
            Dictionary containing model information
        """
        onnx_model = onnx.load(onnx_path)
        
        # Get model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        
        # Get input/output information
        inputs = []
        outputs = []
        
        for input in onnx_model.graph.input:
            inputs.append({
                'name': input.name,
                'shape': [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                         for dim in input.type.tensor_type.shape.dim]
            })
        
        for output in onnx_model.graph.output:
            outputs.append({
                'name': output.name,
                'shape': [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                         for dim in output.type.tensor_type.shape.dim]
            })
        
        return {
            'model_size_mb': model_size,
            'inputs': inputs,
            'outputs': outputs,
            'opset_version': onnx_model.opset_import[0].version,
            'ir_version': onnx_model.ir_version
        }


def export_model_to_onnx(
    model: nn.Module,
    config: Dict,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    validate: bool = True,
    compare: bool = True
) -> Dict:
    """
    Export a PyTorch model to ONNX format with validation.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        output_path: Path to save the ONNX model
        input_shape: Input tensor shape
        validate: Whether to validate the ONNX model
        compare: Whether to compare PyTorch and ONNX outputs
        
    Returns:
        Dictionary containing export results
    """
    exporter = ONNXExporter(model, config)
    
    # Export to ONNX
    onnx_path = exporter.export_to_onnx(
        output_path=output_path,
        input_shape=input_shape,
        dynamic_batch=config['export']['dynamic_batch']
    )
    
    # Get model info
    model_info = exporter.get_onnx_model_info(onnx_path)
    
    results = {
        'onnx_path': onnx_path,
        'model_info': model_info,
        'validation_passed': False,
        'comparison_passed': False
    }
    
    # Validate ONNX model
    if validate:
        results['validation_passed'] = exporter.validate_onnx(onnx_path, input_shape)
    
    # Compare PyTorch and ONNX outputs
    if compare:
        results['comparison_passed'] = exporter.compare_pytorch_onnx(onnx_path, input_shape)
    
    return results 