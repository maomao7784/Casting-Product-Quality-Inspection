import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

# Data Loader
class CastingDataset(Dataset):
    # Initialize the dataset
    def __init__(self, def_path, ok_path, x=None, y=None):
        super(CastingDataset, self).__init__()
        self.def_files = sorted([os.path.join(def_path, x) 
                                 for x in os.listdir(def_path) if x.endswith(".jpeg")])
        self.ok_files = sorted([os.path.join(ok_path, x) 
                                for x in os.listdir(ok_path) if x.endswith(".jpeg")])
        # Store files and their labels
        self.files = [(file, 1) for file in self.def_files] \
            + [(file, 0) for file in self.ok_files]
        # Preprocess and store images and labels as tensors
        images = []
        labels = []
        for file, label in self.files:
            im = Image.open(file).convert("L")  # read the img as grayscale 
            im = im.resize((224, 224))  # Resize the image to 224x224
            im = np.array(im)/255.0  # Normalize the image to [0, 1] range
            images.append(im)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
        self._len = len(self.files)

    # Get a single sample from the dataset
    def __getitem__(self, idx):
        # Add a channel dimension (1) to the features
        # Return the features and label at the given index   
        return self.images[idx].unsqueeze(0), self.labels[idx]

    # Return the length of the dataset
    def __len__(self):
        return self._len
