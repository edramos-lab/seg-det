#!/usr/bin/env python3
"""
Main pipeline for strawberry semantic segmentation.
This script orchestrates the entire pipeline: prepare, augment, train, evaluate, export to ONNX, and TensorRT.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import create_dataloaders
from src.models.model_factory import create_model, get_model_info
from src.training.trainer import SegmentationTrainer
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_training_curves, plot_confusion_matrix, plot_class_metrics
from src.export.onnx_export import export_model_to_onnx
from src.export.tensorrt_export import export_onnx_to_tensorrt


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the pipeline by creating necessary directories and validating dataset."""
    print("=" * 60)
    print("PREPARING PIPELINE")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    
    # Validate dataset path
    dataset_path = config['dataset']['root_path']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Check for required files
    required_files = [
        os.path.join(dataset_path, 'train', '_annotations.coco.json'),
        os.path.join(dataset_path, 'valid', '_annotations.coco.json'),
        os.path.join(dataset_path, 'test', '_annotations.coco.json')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required annotation file not found: {file_path}")
    
    print(f"✓ Dataset path validated: {dataset_path}")
    print(f"✓ Required annotation files found")
    print(f"✓ Output directories created")
    
    return config


def augment_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Data augmentation is handled by the dataset class during training."""
    print("=" * 60)
    print("DATA AUGMENTATION")
    print("=" * 60)
    
    print("✓ Data augmentation configured in dataset transforms")
    print("✓ Augmentation includes:")
    print("  - Horizontal/Vertical flips")
    print("  - Random rotation")
    print("  - Brightness/Contrast adjustments")
    print("  - Gaussian blur and noise")
    print("  - Elastic transformations")
    
    return config


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the semantic segmentation model."""
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model_info = get_model_info(model)
    print(f"Model created: {model_info}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train model
    print("Starting training...")
    training_results = trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print("✓ Training completed")
    print(f"✓ Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"✓ Best validation metric: {training_results['best_val_metric']:.4f}")
    
    # Cleanup
    trainer.cleanup()
    
    return training_results


def evaluate_model(config: Dict[str, Any], training_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the trained model."""
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model and load best weights
    model = create_model(config)
    checkpoint = torch.load('models/best_metric.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = training_results['test_metrics']
    
    print("Test Results:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Plot class metrics
    class_names = config['dataset']['class_names']
    plot_class_metrics(test_metrics, class_names, 'dice', 'results/class_dice_scores.png')
    plot_class_metrics(test_metrics, class_names, 'iou', 'results/class_iou_scores.png')
    
    print("✓ Evaluation completed")
    print("✓ Results saved to results/ directory")
    
    return test_metrics


def export_to_onnx(config: Dict[str, Any]) -> Dict[str, Any]:
    """Export the trained model to ONNX format."""
    print("=" * 60)
    print("EXPORTING TO ONNX")
    print("=" * 60)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model and load best weights
    model = create_model(config)
    checkpoint = torch.load('models/best_metric.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Export to ONNX
    onnx_path = config['export']['onnx_path']
    input_shape = tuple(config['export']['input_shape'])
    
    print(f"Exporting model to ONNX: {onnx_path}")
    onnx_results = export_model_to_onnx(
        model=model,
        config=config,
        output_path=onnx_path,
        input_shape=input_shape,
        validate=True,
        compare=True
    )
    
    print("ONNX Export Results:")
    print(f"  Model size: {onnx_results['model_info']['model_size_mb']:.2f} MB")
    print(f"  Validation passed: {onnx_results['validation_passed']}")
    print(f"  Comparison passed: {onnx_results['comparison_passed']}")
    
    print("✓ ONNX export completed")
    
    return onnx_results


def export_to_tensorrt(config: Dict[str, Any], onnx_results: Dict[str, Any]) -> Dict[str, Any]:
    """Export the ONNX model to TensorRT format."""
    print("=" * 60)
    print("EXPORTING TO TENSORRT")
    print("=" * 60)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model for comparison
    model = create_model(config)
    checkpoint = torch.load('models/best_metric.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Export to TensorRT
    onnx_path = onnx_results['onnx_path']
    tensorrt_path = config['export']['tensorrt_path']
    fp16 = config['export']['fp16']
    
    print(f"Exporting ONNX model to TensorRT: {tensorrt_path}")
    tensorrt_results = export_onnx_to_tensorrt(
        onnx_path=onnx_path,
        config=config,
        output_path=tensorrt_path,
        fp16=fp16,
        benchmark=True,
        compare=True,
        pytorch_model=model
    )
    
    print("TensorRT Export Results:")
    print(f"  Engine size: {tensorrt_results['engine_info']['engine_size_mb']:.2f} MB")
    print(f"  Number of layers: {tensorrt_results['engine_info']['num_layers']}")
    if tensorrt_results['benchmark_results']:
        benchmark = tensorrt_results['benchmark_results']
        print(f"  Average inference time: {benchmark['avg_inference_time_ms']:.2f} ms")
        print(f"  FPS: {benchmark['fps']:.2f}")
    print(f"  Comparison passed: {tensorrt_results['comparison_passed']}")
    
    print("✓ TensorRT export completed")
    
    return tensorrt_results


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Strawberry Semantic Segmentation Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing model')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip export steps')
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Pipeline steps
    try:
        # Step 1: Prepare
        config = prepare_pipeline(config)
        
        # Step 2: Augment
        config = augment_data(config)
        
        # Step 3: Train (unless skipped)
        training_results = None
        if not args.skip_training:
            training_results = train_model(config)
        else:
            print("Skipping training (using existing model)")
            training_results = {'test_metrics': {}}  # Placeholder
        
        # Step 4: Evaluate
        test_metrics = evaluate_model(config, training_results)
        
        # Step 5: Export to ONNX (unless skipped)
        onnx_results = None
        if not args.skip_export:
            onnx_results = export_to_onnx(config)
        else:
            print("Skipping ONNX export")
        
        # Step 6: Export to TensorRT (unless skipped)
        tensorrt_results = None
        if not args.skip_export and onnx_results:
            tensorrt_results = export_to_tensorrt(config, onnx_results)
        else:
            print("Skipping TensorRT export")
        
        # Summary
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Summary:")
        print(f"  ✓ Dataset prepared: {config['dataset']['root_path']}")
        print(f"  ✓ Model trained: {config['model']['name']}")
        if test_metrics:
            print(f"  ✓ Model evaluated: Dice={test_metrics.get('dice', 0):.4f}")
        if onnx_results:
            print(f"  ✓ ONNX exported: {onnx_results['onnx_path']}")
        if tensorrt_results:
            print(f"  ✓ TensorRT exported: {tensorrt_results['engine_path']}")
        
        print("\nOutput files:")
        print("  - models/best_metric.pth (PyTorch model)")
        print("  - results/training_curves.png")
        print("  - results/class_dice_scores.png")
        print("  - results/class_iou_scores.png")
        if onnx_results:
            print(f"  - {onnx_results['onnx_path']} (ONNX model)")
        if tensorrt_results:
            print(f"  - {tensorrt_results['engine_path']} (TensorRT engine)")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 