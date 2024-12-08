import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

class IncomeDataset(Dataset):
    BASE_DIR = os.path.join(DATA_DIR, "adult")
    PATHS = {
        "train_basis": os.path.join(BASE_DIR, "train_basis.csv"),
        "train_betahats": os.path.join(BASE_DIR, "train_betahats.csv"),
        "test": os.path.join(BASE_DIR, "test.csv"),
        "validate": os.path.join(BASE_DIR, "validate.csv"),
    }
    
    FEATURES = ["education_num", "age"]  # Define the features we want to use
    SENSITIVE = "sex"
    TARGET = "income"
    DECISION = "loan_approved"
    
    def __init__(self, split="train", features=None, sensitive_attribute=None, 
                 target=None, decision_variable=None, to_tensor=True):
        """
        Args:
            split: The dataset split, one of "train_basis", "train_betahats", "test", or "validate"
            features: The features to use in the dataset
            sensitive_attribute: The sensitive attribute to use as the protected class
            target: The target variable to use in the dataset
            decision_variable: The decision variable to use in the dataset
        """
        if split not in self.PATHS:
            raise ValueError(f"Invalid split. Must be one of {list(self.PATHS.keys())}")
        
        self.X_names = features if features is not None else self.FEATURES
        self.A_name = sensitive_attribute if sensitive_attribute is not None else self.SENSITIVE
        self.Y_name = target if target is not None else self.TARGET
        self.D_name = decision_variable if decision_variable is not None else self.DECISION
        
        data = pd.read_csv(self.PATHS[split])
        self.X = data[self.X_names].values
        self.A = data[[self.A_name]].values
        self.Y = data[[self.Y_name]].values
        self.D = data[[self.D_name]].values
        
        if to_tensor:
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.A = torch.tensor(self.A, dtype=torch.float32)
            self.Y = torch.tensor(self.Y, dtype=torch.float32)
            self.D = torch.tensor(self.D, dtype=torch.float32)
            
        self.is_tensor = to_tensor

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample from the dataset at index `idx`."""
        x = self.X[idx]
        a = self.A[idx]
        y = self.Y[idx]
        d = self.D[idx]
        return x, a, y, d
