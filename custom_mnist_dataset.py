import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_test=False):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        item = self.data_frame.iloc[index]
        
        if self.is_test:
            image = item.values.reshape(28, 28).astype(np.uint8)
            label = None
        else:
            image = item[1:].values.reshape(28, 28).astype(np.uint8)
            label = item.iloc[0]
            
        image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.is_test:
            return image
        else:
            return image, label
