import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import re

from sklearn.model_selection import train_test_split

# Utility function: Extracts natural sort keys for filenames (e.g., sorts "file10" after "file2")
def natural_sort_key(filename):
    """
    Key for natural sorting of filenames.
    Splits the filename into numeric and non-numeric parts for intuitive ordering.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Custom Dataset class for Breast Ultrasound Images and Masks
class BreastUltrasoundDataset(Dataset):
    """
    PyTorch Dataset for breast ultrasound images and their corresponding segmentation masks.

    Args:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks.
        transform (callable, optional): Transformations to apply to the images and masks.

    Attributes:
        image_filenames (list): List of sorted filenames for images.
        mask_filenames (list): List of sorted filenames for masks.
        transform (callable): Transformation function.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir), key = natural_sort_key)
        self.mask_filenames = sorted(os.listdir(mask_dir), key = natural_sort_key)
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Transformed image and mask as tensors.
        """
        # Paths to image and mask files
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask values to binary (0 or 1)
        mask = np.array(mask)
        mask = np.where(mask > 1, 1, 0).astype(np.float32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension to mask

        return image, mask

# Function to split dataset into train, validation, and test sets
def split_dataset(dataset, test_ratio=0.1, val_ratio=0.2, random_seed=42):
    """
    Splits the dataset into train, validation, and test subsets.

    Args:
        dataset (Dataset): The full dataset to split.
        test_ratio (float): Proportion of the dataset to use as test set.
        val_ratio (float): Proportion of the dataset to use as validation set.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: Datasets for training, validation, and testing.
    """
    # Determine dataset sizes
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)

    # Generate indices for all samples
    all_indices = list(range(total_size))
    
    # Split indices into train+val and test
    train_val_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_seed)

    # Further split train+val into train and validation
    train_size = int(total_size * (1 - val_ratio - test_ratio))
    train_indices, val_indices = train_test_split(train_val_indices, test_size=len(train_val_indices) - train_size, random_state=random_seed)

    # Create Subsets for PyTorch DataLoader
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset