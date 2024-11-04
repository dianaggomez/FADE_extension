import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

class IncomeDataset(Dataset):
    BASE_DIR = os.path.join(DATA_DIR, "adult")
    PATHS = {
        "train_basis": os.path.join(BASE_DIR, "train_basis.csv"),
        "train_betahats": os.path.join(BASE_DIR, "train_betahats.csv"),
        "test": os.path.join(BASE_DIR, "test.csv"),
        "validate": os.path.join(BASE_DIR, "validate.csv"),
    }
    
    FEATURES = ["education_num", "age"]  # Define the features we want to use
    TARGET = "income"
    
    def __init__(self, split="train", sensitive_attribute="sex", features=None, to_tensor=True):
        """
        Args:
            split: The dataset split, one of "train_basis", "train_betahats", "test", or "validate"
            sensitive_attribute: The sensitive attribute to use as the protected class
            features: The features to use in the dataset
        """
        if split not in self.PATHS:
            raise ValueError(f"Invalid split. Must be one of {list(self.PATHS.keys())}")
        
        if features is None:
            features = self.FEATURES
            
        self.S_name = sensitive_attribute
        self.X_names = features
        self.Y_name = self.TARGET
        
        data = pd.read_csv(self.PATHS[split])
        self.S = data[[self.S_name]].values
        self.X = data[self.X_names].values
        self.Y = data[[self.Y_name]].values
        
        if to_tensor:
            self.S = torch.tensor(self.S, dtype=torch.float32)
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.Y = torch.tensor(self.Y, dtype=torch.float32)
            
        self.is_tensor = to_tensor

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample from the dataset at index `idx`."""
        x = self.X[idx]
        y = self.Y[idx]
        s = self.S[idx]
        return x, y, s
