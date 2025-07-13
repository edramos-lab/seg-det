import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional
import cv2


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Metric curves
    axes[0, 1].plot(history['train_metric'], label='Train Dice', color='blue')
    axes[0, 1].plot(history['val_metric'], label='Val Dice', color='red')
    axes[0, 1].set_title('Training and Validation Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate curve
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
    
    # Loss ratio
    if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
        loss_ratio = [t/v for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 1].plot(loss_ratio, color='purple')
        axes[1, 1].set_title('Train/Val Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Plot predictions vs ground truth.
    
    Args:
        images: Input images (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        predictions: Predicted masks (B, C, H, W)
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Path to save the plot
    """
    num_classes = len(class_names)
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, images.size(0))):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        target = torch.argmax(targets[i], dim=0).cpu().numpy()
        target_colored = np.zeros((*target.shape, 3))
        for j in range(num_classes):
            target_colored[target == j] = colors[j][:3]
        axes[i, 1].imshow(target_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
        pred_colored = np.zeros((*pred.shape, 3))
        for j in range(num_classes):
            pred_colored[pred == j] = colors[j][:3]
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=class_names[i])
                      for i in range(num_classes)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    metric_type: str = 'dice',
    save_path: Optional[str] = None
):
    """
    Plot per-class metrics.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
        metric_type: Type of metric to plot ('dice', 'iou', 'precision', 'recall')
        save_path: Path to save the plot
    """
    class_metrics = []
    for i in range(len(class_names)):
        key = f'class_{i}_{metric_type}'
        if key in metrics:
            class_metrics.append(metrics[key])
        else:
            class_metrics.append(0.0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_metrics, color='skyblue', edgecolor='navy')
    plt.title(f'Per-Class {metric_type.upper()} Scores')
    plt.xlabel('Classes')
    plt.ylabel(f'{metric_type.upper()} Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, class_metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class metrics plot saved to {save_path}")
    
    plt.show()


def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[List] = None
) -> np.ndarray:
    """
    Create overlay image with segmentation mask.
    
    Args:
        image: Input image (H, W, C)
        mask: Segmentation mask (H, W)
        alpha: Transparency factor
        colors: List of colors for each class
        
    Returns:
        Overlay image
    """
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, mask.max() + 1))[:, :3]
    
    overlay = image.copy()
    for i in range(1, mask.max() + 1):
        mask_i = (mask == i)
        color = colors[i]
        overlay[mask_i] = overlay[mask_i] * (1 - alpha) + color * alpha
    
    return overlay


def plot_sample_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str],
    num_samples: int = 6,
    save_path: Optional[str] = None
):
    """
    Plot sample predictions from a dataloader.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run inference on
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Path to save the plot
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)
            
            predictions = model(images)
            
            for i in range(min(images.size(0), num_samples - sample_count)):
                # Original image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                axes[sample_count, 0].imshow(img)
                axes[sample_count, 0].set_title('Original Image')
                axes[sample_count, 0].axis('off')
                
                # Ground truth
                target = torch.argmax(targets[i], dim=0).cpu().numpy()
                target_colored = np.zeros((*target.shape, 3))
                for j in range(len(class_names)):
                    target_colored[target == j] = plt.cm.Set3(j / len(class_names))[:3]
                axes[sample_count, 1].imshow(target_colored)
                axes[sample_count, 1].set_title('Ground Truth')
                axes[sample_count, 1].axis('off')
                
                # Prediction
                pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
                pred_colored = np.zeros((*pred.shape, 3))
                for j in range(len(class_names)):
                    pred_colored[pred == j] = plt.cm.Set3(j / len(class_names))[:3]
                axes[sample_count, 2].imshow(pred_colored)
                axes[sample_count, 2].set_title('Prediction')
                axes[sample_count, 2].axis('off')
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show() 