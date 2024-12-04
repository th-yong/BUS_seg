import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import re

from sklearn.model_selection import train_test_split


def natural_sort_key(filename):

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]


class BreastUltrasoundDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir), key = natural_sort_key)
        self.mask_filenames = sorted(os.listdir(mask_dir), key = natural_sort_key)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask values to [0, 1]
        mask = np.array(mask)
        mask = np.where(mask > 1, 1, 0).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        return image, mask

def split_dataset(dataset, test_ratio=0.1, val_ratio=0.2, random_seed=42):
    # Shuffle and split indices
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)

    # Fixed test indices
    all_indices = list(range(total_size))
    train_val_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_seed)

    # Split Train and Validation
    train_size = int(total_size * (1 - val_ratio - test_ratio))
    train_indices, val_indices = train_test_split(train_val_indices, test_size=len(train_val_indices) - train_size, random_state=random_seed)

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset