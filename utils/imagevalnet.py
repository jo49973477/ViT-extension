import logging
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ValidationImageNet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.img_directories = [os.path.join(root, "ILSVRC2011_val_{:08d}.JPEG".format(i)) for i in range(1, 50001)]
        val_directory = os.path.join(root, "ILSVRC2011_validation_ground_truth.txt")

        with open(val_directory, 'r') as f:
            self.labels = [int(line.strip())-1 for line in f]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_directories[idx])
        image = self.transform(image)
        
        return image, self.labels[idx]