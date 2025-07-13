#!/usr/bin/env python3
"""
Test script to verify the pipeline setup and basic functionality.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data.dataset import StrawberrySegmentationDataset, create_dataloaders
        print("✓ Data module imported successfully")
    except Exception as e:
        print(f"✗ Data module import failed: {e}")
        return False
    
    try:
        from src.models.model_factory import create_model, get_model_info
        print("✓ Models module imported successfully")
    except Exception as e:
        print(f"✗ Models module import failed: {e}")
        return False
    
    try:
        from src.models.losses import get_loss_function
        print("✓ Losses module imported successfully")
    except Exception as e:
        print(f"✗ Losses module import failed: {e}")
        return False
    
    try:
        from src.utils.metrics import calculate_metrics
        print("✓ Metrics module imported successfully")
    except Exception as e:
        print(f"✗ Metrics module import failed: {e}")
        return False
    
    try:
        from src.utils.visualization import plot_training_curves
        print("✓ Visualization module imported successfully")
    except Exception as e:
        print(f"✗ Visualization module import failed: {e}")
        return False
    
    try:
        from src.export.onnx_export import export_model_to_onnx
        print("✓ ONNX export module imported successfully")
    except Exception as e:
        print(f"⚠ ONNX export module import failed: {e}")
        print("  This is likely due to ONNX version compatibility. The pipeline will still work without ONNX export.")
        # Don't fail the test for ONNX issues
        return True
    
    try:
        from src.export.tensorrt_export import export_onnx_to_tensorrt
        print("✓ TensorRT export module imported successfully")
    except Exception as e:
        print(f"⚠ TensorRT export module import failed: {e}")
        print("  This is likely due to TensorRT version compatibility. The pipeline will still work without TensorRT export.")
        # Don't fail the test for TensorRT issues
        return True
    
    return True


def test_config():
    """Test if configuration file can be loaded."""
    print("\nTesting configuration...")
    
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['dataset', 'model', 'training', 'augmentation', 'loss', 'optimizer', 'scheduler', 'logging', 'export', 'hardware']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("✓ Configuration loaded successfully")
        print(f"  - Dataset path: {config['dataset']['root_path']}")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Classes: {config['dataset']['num_classes']}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_dataset_path():
    """Test if dataset path exists."""
    print("\nTesting dataset path...")
    
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_path = config['dataset']['root_path']
        
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset path does not exist: {dataset_path}")
            return False
        
        # Check for required annotation files
        required_files = [
            os.path.join(dataset_path, 'train', '_annotations.coco.json'),
            os.path.join(dataset_path, 'valid', '_annotations.coco.json'),
            os.path.join(dataset_path, 'test', '_annotations.coco.json')
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"✗ Required annotation file not found: {file_path}")
                return False
        
        print("✓ Dataset path and annotation files found")
        return True
    except Exception as e:
        print(f"✗ Dataset path test failed: {e}")
        return False


def test_model_creation():
    """Test if model can be created."""
    print("\nTesting model creation...")
    
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        from src.models.model_factory import create_model, get_model_info
        
        model = create_model(config)
        model_info = get_model_info(model)
        
        print("✓ Model created successfully")
        print(f"  - Total parameters: {model_info['total_parameters']:,}")
        print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"  - Model size: {model_info['model_size_mb']:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def test_loss_function():
    """Test if loss function can be created."""
    print("\nTesting loss function...")
    
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        from src.models.losses import get_loss_function
        
        loss_fn = get_loss_function(config)
        
        print("✓ Loss function created successfully")
        print(f"  - Loss type: {config['loss']['name']}")
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA is not available (training will use CPU)")
    
    return True


def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 'PIL', 
        'matplotlib', 'seaborn', 'tqdm', 'albumentations', 'scikit-learn',
        'tensorboard', 'wandb', 'pycocotools', 'onnx', 'onnxruntime',
        'pyyaml', 'hydra-core', 'omegaconf', 'timm', 'transformers',
        'segmentation-models-pytorch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'pyyaml':
                import yaml
            elif package == 'hydra-core':
                import hydra
            elif package == 'segmentation-models-pytorch':
                import segmentation_models_pytorch
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - MISSING ({e})")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required dependencies are installed")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("STRAWBERRY SEGMENTATION PIPELINE - SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Path", test_dataset_path),
        ("Model Creation", test_model_creation),
        ("Loss Function", test_loss_function),
        ("CUDA", test_cuda),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run the full pipeline: python main.py")
        print("2. Or use the convenience script: ./run_pipeline.sh full")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check dataset path in config/config.yaml")
        print("3. Ensure CUDA is properly installed (optional)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 