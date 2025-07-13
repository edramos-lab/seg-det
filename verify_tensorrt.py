#!/usr/bin/env python3
"""
TensorRT Installation Verification Script
This script verifies that TensorRT is properly installed and working.
"""

import sys
import os

def check_tensorrt_installation():
    """Check TensorRT installation and functionality."""
    print("=" * 60)
    print("TENSORRT INSTALLATION VERIFICATION")
    print("=" * 60)
    
    # Check if TensorRT can be imported
    try:
        import tensorrt as trt
        print(f"✓ TensorRT imported successfully")
        print(f"✓ TensorRT version: {trt.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import TensorRT: {e}")
        return False
    
    # Check TensorRT functionality
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        print(f"✓ TensorRT Builder created successfully")
        
        # Check if FP16 is supported
        if builder.platform_has_fast_fp16:
            print(f"✓ FP16 precision supported")
        else:
            print(f"⚠ FP16 precision not supported")
        
        # Check if INT8 is supported
        if builder.platform_has_fast_int8:
            print(f"✓ INT8 precision supported")
        else:
            print(f"⚠ INT8 precision not supported")
            
    except Exception as e:
        print(f"✗ TensorRT functionality test failed: {e}")
        return False
    
    # Check environment variables
    print("\nEnvironment Variables:")
    tensorrt_root = os.environ.get('TENSORRT_ROOT', 'Not set')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    
    print(f"  TENSORRT_ROOT: {tensorrt_root}")
    print(f"  LD_LIBRARY_PATH: {ld_library_path}")
    
    # Check if TensorRT libraries are accessible
    if 'tensorrt' in ld_library_path.lower():
        print("✓ TensorRT libraries in LD_LIBRARY_PATH")
    else:
        print("⚠ TensorRT libraries not found in LD_LIBRARY_PATH")
    
    # Test ONNX parser
    try:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        print("✓ ONNX parser created successfully")
    except Exception as e:
        print(f"✗ ONNX parser test failed: {e}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    return True

def test_tensorrt_build():
    """Test TensorRT engine building capability."""
    print("\nTesting TensorRT engine building...")
    
    try:
        import tensorrt as trt
        import numpy as np
        
        # Create a simple network
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Add a simple layer
        input_tensor = network.add_input("input", trt.DataType.FLOAT, (1, 3, 224, 224))
        identity = network.add_identity(input_tensor)
        network.mark_output(identity.get_output(0))
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 20  # 1MB
        
        # Try to build engine
        engine = builder.build_engine(network, config)
        
        if engine is not None:
            print("✓ TensorRT engine building test successful")
            return True
        else:
            print("✗ TensorRT engine building failed")
            return False
            
    except Exception as e:
        print(f"✗ TensorRT engine building test failed: {e}")
        return False

if __name__ == "__main__":
    print("TensorRT Installation Verification")
    print("=" * 40)
    
    # Check basic installation
    if check_tensorrt_installation():
        print("✓ TensorRT installation verified successfully!")
        
        # Test engine building
        if test_tensorrt_build():
            print("✓ TensorRT engine building verified successfully!")
            print("\n🎉 TensorRT is ready for use!")
        else:
            print("⚠ TensorRT installation complete but engine building failed")
    else:
        print("✗ TensorRT installation verification failed")
        sys.exit(1) 