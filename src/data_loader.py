import os
import numpy as np
from skimage import io, img_as_float
from torch.utils.data import Dataset, DataLoader

class DenoisingDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_images = [os.path.join(low_dir, img) for img in os.listdir(low_dir)]
        self.high_images = [os.path.join(high_dir, img) for img in os.listdir(high_dir)]
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_image = img_as_float(io.imread(self.low_images[idx]))
        high_image = img_as_float(io.imread(self.high_images[idx]))
        return low_image, high_image
    def get_dataloader(low_dir, high_dir, batch_size=16, shuffle=True):
        dataset = DenoisingDataset(low_dir, high_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)