#!/usr/bin/env python3
"""
Jetson Deployment Script for Strawberry Semantic Segmentation
This script exports trained models for deployment on Jetson Xavier AGX and Orin devices.
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

from src.models.model_factory import create_model
from src.export.jetson_export import export_for_jetson


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def detect_jetson_platform() -> str:
    """Detect the Jetson platform."""
    try:
        # Check for Orin-specific features
        if os.path.exists('/sys/devices/gpu.0/devfreq/17000000.gv11b'):
            return 'orin'
        # Check for Xavier-specific features
        elif os.path.exists('/sys/devices/gpu.0/devfreq/17000000.gp10b'):
            return 'xavier_agx'
        else:
            # Fallback based on GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory >= 8 * 1024**3:  # 8GB or more
                    return 'orin'
                else:
                    return 'xavier_agx'
            else:
                return 'xavier_agx'  # Default fallback
    except:
        return 'xavier_agx'  # Default fallback


def export_for_jetson_platform(
    model_path: str,
    config: Dict[str, Any],
    output_dir: str,
    platform: str = None
) -> Dict[str, Any]:
    """
    Export model for specific Jetson platform.
    
    Args:
        model_path: Path to trained PyTorch model
        config: Configuration dictionary
        output_dir: Output directory for deployment files
        platform: Jetson platform ('xavier_agx' or 'orin')
        
    Returns:
        Dictionary containing deployment information
    """
    print("=" * 60)
    print("JETSON DEPLOYMENT EXPORT")
    print("=" * 60)
    
    # Detect platform if not specified
    if platform is None:
        platform = detect_jetson_platform()
    
    print(f"Target platform: {platform}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model architecture
    model = create_model(config)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Export for Jetson
    print("Exporting model for Jetson deployment...")
    deployment_info = export_for_jetson(
        model=model,
        config=config,
        output_dir=output_dir,
        platform=platform
    )
    
    print("✓ Jetson deployment export completed!")
    print(f"✓ Deployment package created in: {output_dir}")
    
    return deployment_info


def main():
    """Main function for Jetson deployment."""
    parser = argparse.ArgumentParser(description="Export model for Jetson deployment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_metric.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/jetson",
        help="Output directory for deployment files"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=['xavier_agx', 'orin'],
        help="Target Jetson platform (auto-detected if not specified)"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available Jetson platforms"
    )
    
    args = parser.parse_args()
    
    if args.list_platforms:
        print("Available Jetson platforms:")
        print("  - xavier_agx: NVIDIA Jetson Xavier AGX")
        print("  - orin: NVIDIA Jetson Orin")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Export for Jetson
    try:
        deployment_info = export_for_jetson_platform(
            model_path=args.model,
            config=config,
            output_dir=args.output_dir,
            platform=args.platform
        )
        
        print("\n" + "=" * 60)
        print("DEPLOYMENT SUMMARY")
        print("=" * 60)
        print(f"Platform: {deployment_info['platform']}")
        print(f"Deployment date: {deployment_info['deployment_date']}")
        print("\nModel files:")
        for file_type, file_path in deployment_info['model_files'].items():
            print(f"  {file_type}: {file_path}")
        
        print("\nPerformance benchmark:")
        benchmark = deployment_info['benchmark_results']
        print(f"  Average latency: {benchmark['avg_latency_ms']:.2f} ms")
        print(f"  FPS: {benchmark['fps']:.2f}")
        print(f"  Throughput: {benchmark['throughput']:.2f} images/sec")
        
        print(f"\nDeployment package ready in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during Jetson export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 