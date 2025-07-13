import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Tuple
from sklearn.metrics import confusion_matrix


def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    if target.dim() == 4:
        target = torch.argmax(target, dim=1)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    if target.dim() == 4:
        target = torch.argmax(target, dim=1)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> Tuple[float, float]:
    """
    Calculate precision and recall.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        smooth: Smoothing factor
        
    Returns:
        Tuple of (precision, recall)
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    if target.dim() == 4:
        target = torch.argmax(target, dim=1)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    precision = (intersection + smooth) / (pred_flat.sum() + smooth)
    recall = (intersection + smooth) / (target_flat.sum() + smooth)
    
    return precision.item(), recall.item()


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel accuracy.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        
    Returns:
        Pixel accuracy
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)
    if target.dim() == 4:
        target = torch.argmax(target, dim=1)
    
    correct = (pred == target).sum()
    total = pred.numel()
    
    accuracy = correct / total
    return accuracy.item()


def calculate_class_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Calculate per-class metrics.
    
    Args:
        pred: Predicted masks (B, C, H, W)
        target: Ground truth masks (B, C, H, W)
        num_classes: Number of classes
        
    Returns:
        Dictionary of per-class metrics
    """
    pred_indices = torch.argmax(pred, dim=1)
    target_indices = torch.argmax(target, dim=1)
    
    metrics = {}
    
    for class_id in range(num_classes):
        pred_class = (pred_indices == class_id)
        target_class = (target_indices == class_id)
        
        intersection = (pred_class & target_class).sum()
        union = (pred_class | target_class).sum()
        
        # Dice coefficient
        dice = (2 * intersection) / (pred_class.sum() + target_class.sum() + 1e-6)
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # Precision and Recall
        precision = intersection / (pred_class.sum() + 1e-6)
        recall = intersection / (target_class.sum() + 1e-6)
        
        metrics[f'class_{class_id}_dice'] = dice.item()
        metrics[f'class_{class_id}_iou'] = iou.item()
        metrics[f'class_{class_id}_precision'] = precision.item()
        metrics[f'class_{class_id}_recall'] = recall.item()
    
    return metrics


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, metric_type: str = 'all') -> Union[float, Dict[str, float]]:
    """
    Calculate segmentation metrics.
    
    Args:
        pred: Predicted masks (B, C, H, W)
        target: Ground truth masks (B, C, H, W)
        metric_type: Type of metric to calculate ('dice', 'iou', 'precision', 'recall', 'accuracy', 'all')
        
    Returns:
        Metric value or dictionary of metrics
    """
    if metric_type == 'dice':
        return calculate_dice_coefficient(pred, target)
    elif metric_type == 'iou':
        return calculate_iou(pred, target)
    elif metric_type == 'precision':
        precision, _ = calculate_precision_recall(pred, target)
        return precision
    elif metric_type == 'recall':
        _, recall = calculate_precision_recall(pred, target)
        return recall
    elif metric_type == 'accuracy':
        return calculate_accuracy(pred, target)
    elif metric_type == 'all':
        dice = calculate_dice_coefficient(pred, target)
        iou = calculate_iou(pred, target)
        precision, recall = calculate_precision_recall(pred, target)
        accuracy = calculate_accuracy(pred, target)
        
        # Calculate per-class metrics
        num_classes = pred.size(1)
        class_metrics = calculate_class_metrics(pred, target, num_classes)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            **class_metrics
        }
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def calculate_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        pred: Predicted masks (B, C, H, W)
        target: Ground truth masks (B, C, H, W)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix
    """
    pred_indices = torch.argmax(pred, dim=1)
    target_indices = torch.argmax(target, dim=1)
    
    pred_flat = pred_indices.view(-1).cpu().numpy()
    target_flat = target_indices.view(-1).cpu().numpy()
    
    return confusion_matrix(target_flat, pred_flat, labels=range(num_classes))


def calculate_mean_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Calculate mean metrics across all classes.
    
    Args:
        pred: Predicted masks (B, C, H, W)
        target: Ground truth masks (B, C, H, W)
        num_classes: Number of classes
        
    Returns:
        Dictionary of mean metrics
    """
    class_metrics = calculate_class_metrics(pred, target, num_classes)
    
    mean_dice = np.mean([class_metrics[f'class_{i}_dice'] for i in range(num_classes)])
    mean_iou = np.mean([class_metrics[f'class_{i}_iou'] for i in range(num_classes)])
    mean_precision = np.mean([class_metrics[f'class_{i}_precision'] for i in range(num_classes)])
    mean_recall = np.mean([class_metrics[f'class_{i}_recall'] for i in range(num_classes)])
    
    return {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall
    } 