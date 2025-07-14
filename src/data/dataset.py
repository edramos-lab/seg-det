import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional


class StrawberrySegmentationDataset(Dataset):
    """
    Custom dataset for strawberry semantic segmentation using COCO format annotations.
    """
    
    def __init__(
        self,
        root_path: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (512, 512),
        num_classes: int = 8
    ):
        """
        Initialize the dataset.
        
        Args:
            root_path: Path to the dataset root directory
            split: Dataset split ('train', 'valid', 'test')
            transform: Albumentations transforms
            target_size: Target image size (height, width)
            num_classes: Number of classes including background
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        
        # Load COCO annotations
        self.annotations_file = os.path.join(root_path, split, "_annotations.coco.json")
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image to annotations mapping
        self.image_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)
        
        # Create image list
        self.images = self.coco_data['images']
        
        # Create category mapping
        self.category_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        self.id_to_category = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Create mapping from dataset categories to our 7 disease classes
        # Map 'fruit' and 'Healthy Strawberry' to background (class 0)
        # Map disease classes to their respective indices (1-6)
        self.category_mapping = {
            'fruit': 0,  # Background
            'Healthy Strawberry': 0,  # Background
            'Angular Leafspot': 1,
            'Anthracnose Fruit Rot': 2,
            'Blossom Blight': 3,
            'Gray Mold': 4,
            'Leaf Spot': 5,
            'Powdery Mildew Fruit': 6,
            'Powdery Mildew Leaf': 7
        }
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Original categories: {list(self.category_to_id.keys())}")
        print(f"Category mapping: {self.category_mapping}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image and mask tensors
        """
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.root_path, self.split, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Get annotations for this image
        image_id = img_info['id']
        if image_id in self.image_to_annotations:
            for ann in self.image_to_annotations[image_id]:
                # Create polygon mask for this annotation
                segmentation = ann['segmentation'][0]  # COCO format
                points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                
                # Get category name and map to our class indices
                category_name = self.id_to_category[ann['category_id']]
                mapped_class_id = self.category_mapping.get(category_name, 0)  # Default to background
                
                # Fill polygon with mapped category ID
                cv2.fillPoly(mask, [points], mapped_class_id)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default transforms
            transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(),
                ToTensorV2()
            ])
            transformed = transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask values are within valid range (0 to num_classes-1)
        if isinstance(mask, torch.Tensor):
            mask = torch.clamp(mask, 0, self.num_classes - 1).long()
        else:
            mask = np.clip(mask, 0, self.num_classes - 1)
            mask = torch.from_numpy(mask).long()
        
        # Debug: Check mask values (only for first few samples)
        if idx < 5:  # Only debug first 5 samples
            if isinstance(mask, torch.Tensor):
                unique_values = torch.unique(mask).cpu().numpy()
                mask_min, mask_max = mask.min().item(), mask.max().item()
            else:
                unique_values = np.unique(mask)
                mask_min, mask_max = mask.min(), mask.max()
            print(f"Sample {idx}: Mask unique values: {unique_values}, range: [{mask_min}, {mask_max}], num_classes: {self.num_classes}")

        return {
            'image': image,
            'mask': mask,
            'image_id': image_id,
            'file_name': img_info['file_name']
        }



def get_transforms(config: Dict, split: str = "train") -> A.Compose:
    """
    Get Albumentations transforms based on configuration.
    
    Args:
        config: Configuration dictionary
        split: Dataset split
        
    Returns:
        Albumentations compose object
    """
    if split == "train":
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=config['augmentation']['train']['horizontal_flip']),
            A.VerticalFlip(p=config['augmentation']['train']['vertical_flip']),
            A.RandomRotate90(p=config['augmentation']['train']['random_rotate']),
            A.RandomBrightnessContrast(
                brightness_limit=config['augmentation']['train']['random_brightness_contrast'],
                contrast_limit=config['augmentation']['train']['random_brightness_contrast'],
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(1.0, 1.0 + config['augmentation']['train']['random_gamma']),
                p=0.5
            ),
            A.GaussianBlur(
                blur_limit=3,
                p=config['augmentation']['train']['blur']
            ),
            # A.GaussNoise(
            #     var_limit=10.0,
            #     p=config['augmentation']['train']['noise']
            # ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=config['augmentation']['train']['elastic_transform']
            ),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2()
        ])


def create_dataloaders(config: Dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of train, validation, and test dataloaders
    """
    # Create datasets
    train_dataset = StrawberrySegmentationDataset(
        root_path=config['dataset']['root_path'],
        split=config['dataset']['train_split'],
        transform=get_transforms(config, "train"),
        num_classes=config['dataset']['num_classes']
    )
    
    val_dataset = StrawberrySegmentationDataset(
        root_path=config['dataset']['root_path'],
        split=config['dataset']['val_split'],
        transform=get_transforms(config, "val"),
        num_classes=config['dataset']['num_classes']
    )
    
    test_dataset = StrawberrySegmentationDataset(
        root_path=config['dataset']['root_path'],
        split=config['dataset']['test_split'],
        transform=get_transforms(config, "val"),
        num_classes=config['dataset']['num_classes']
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader 