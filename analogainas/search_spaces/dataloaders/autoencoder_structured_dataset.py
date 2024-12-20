import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class AutoEncoderStructuredDataset(Dataset):
    def __init__(self, original_dataset):
        """
        Args:
            original_dataset: A dataset that returns (image, label) pairs, e.g., an MNIST dataset instance.
        """
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the image and label from the original dataset
        img, label = self.dataset[idx]
        # Return (image, image) instead of (image, label)
        return img, img
