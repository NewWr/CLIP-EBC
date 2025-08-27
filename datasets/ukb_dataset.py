import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
import os
import pandas as pd
from PIL import Image
import numpy as np
from typing import Optional, Callable, Union, Tuple


class UKBDataset(Dataset):
    def __init__(
        self,
        excel_path: str,
        data_root: str,
        target_column: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        UKB Dataset for regression tasks, compatible with CLIP-EBC data processing.
        
        Args:
            excel_path: Path to Excel file with metadata
            data_root: Root directory for images
            target_column: Column name for regression target
            split: Dataset split ('train', 'val', 'test', 'all')
            transforms: Optional transform to be applied on images
            return_filename: Whether to return filename
            num_crops: Number of crops per image (for training augmentation)
        """
        assert split in ["train", "val", "test", "all"], f"Split {split} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."
        
        self.excel_path = excel_path
        self.data_root = data_root
        self.target_column = target_column
        self.split = split
        self.transforms = transforms
        self.return_filename = return_filename
        self.num_crops = num_crops
        
        # Load and process metadata
        self.__load_metadata__()
        self.__process_split__()
        
        # Initialize transforms (following CLIP-EBC pattern)
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        print(f"Loaded {len(self.metadata)} samples for {split} split (target: {target_column})")
        if len(self.targets) > 0:
            print(f"Target range: [{self.targets.min():.2f}, {self.targets.max():.2f}], mean: {self.targets.mean():.2f}")
    
    def __load_metadata__(self) -> None:
        """Load metadata from Excel file"""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        # Load metadata
        self.metadata = pd.read_excel(self.excel_path)
        
        # Filter out rows with missing target values
        initial_size = len(self.metadata)
        self.metadata = self.metadata[self.metadata[self.target_column].notna()]
        print(f"Dataset loaded: {len(self.metadata)}/{initial_size} samples (removed {initial_size-len(self.metadata)} missing '{self.target_column}' values)")
        
        # Create full image paths
        self.metadata['full_path'] = self.metadata['processed_path'].apply(
            lambda x: os.path.join(self.data_root, x.replace('\\', '/')) if not pd.isna(x) else None
        )
        
        # Filter out non-existent files
        before_filter = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata['full_path'].apply(lambda x: os.path.exists(x) if x else False)
        ]
        missing_files = before_filter - len(self.metadata)
        if missing_files > 0:
            print(f"Removed {missing_files} samples with missing image files")
    
    def __process_split__(self) -> None:
        """Process data split following UKB pattern"""
        if self.split in ['test', 'all']:
            # For evaluation, use all available data
            print(f"Using all {len(self.metadata)} samples for evaluation")
        else:
            # Random split into train and validation (80% train, 20% validation)
            np.random.seed(42)
            total_len = len(self.metadata)
            
            # Create random indices
            indices = np.random.permutation(total_len)
            train_size = int(0.9 * total_len)
            
            if self.split == 'train':
                train_indices = indices[:train_size]
                self.metadata = self.metadata.iloc[train_indices].reset_index(drop=True)
            elif self.split == 'val':
                val_indices = indices[train_size:]
                self.metadata = self.metadata.iloc[val_indices].reset_index(drop=True)
        
        # Get targets
        self.targets = self.metadata[self.target_column].values.astype('float32')
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, str]]:
        """Get item following CLIP-EBC pattern but adapted for regression"""
        row = self.metadata.iloc[idx]
        img_path = row['full_path']
        target = self.targets[idx]
        
        # Load image
        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        image = self.to_tensor(image)
        
        # Create target tensor (regression target instead of density map)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        # Apply transforms if provided (following CLIP-EBC pattern)
        if self.transforms is not None:
            # For regression, we don't need label transformation like in crowd counting
            # We'll apply transforms to image only and replicate for num_crops
            images_targets = []
            for _ in range(self.num_crops):
                # Apply transforms to image copy
                transformed_image = image.clone()
                # Note: CLIP-EBC transforms expect (image, label) but we only transform image
                # We'll need to adapt this based on the actual transform implementation
                if hasattr(self.transforms, '__call__'):
                    try:
                        # Try to call with both image and dummy label
                        dummy_label = torch.tensor([], dtype=torch.float)
                        transformed_image, _ = self.transforms(transformed_image, dummy_label)
                    except:
                        # If that fails, try with just image
                        transformed_image = self.transforms(transformed_image)
                images_targets.append((transformed_image, target_tensor.clone()))
            
            images, targets = zip(*images_targets)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            targets = [target_tensor.clone() for _ in range(self.num_crops)]
        
        # Normalize images (following CLIP-EBC pattern)
        images = [self.normalize(img) for img in images]
        
        # Stack images
        images = torch.stack(images, 0)
        targets = torch.stack(targets, 0)
        
        if self.return_filename:
            filename = os.path.basename(img_path)
            return images, targets, filename
        else:
            return images, targets


# Available regression targets (from original UKB implementation)
REGRESSION_TARGETS = [
    'diabp',    # Diastolic blood pressure
    'sysbp',    # Systolic blood pressure
    'hba1c',    # HbA1c level
    'fbg',      # Fasting blood glucose
    'ldl',      # LDL cholesterol
    'hdl'       # HDL cholesterol
]