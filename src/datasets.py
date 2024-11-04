import torch
import pandas as pd
from torch.utils.data import Dataset


class IncomeDataset(Dataset):
    
    URLS = {
        "train": 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        "test": 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    }
    
    COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                'marital_status', 'occupation', 'relationship', 'race', 
                'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
                'native_country', 'income']
    
    FEATURES = []  # Define the features we want to use
    
    def __init__(self, split="train", sensitive_attribute="sex"):
        """
        Args:
            X: Input features
            Y: Labels
            S: Sensitive attributes
        """
        if split not in ["train", "test"]:
            raise ValueError("Invalid split. Must be either 'train' or 'test'")
            
        data = pd.read_csv(self.URLS[split], names=self.COLUMNS, sep=',', skipinitialspace=True)
        # self.X = data.drop(columns=['income', sensitive_attribute]).values
        self.X = data[self.FEATURES].values
        self.Y = (data['income'] > 50_000).astype(float).values
        self.S = data[sensitive_attribute].values

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample from the dataset at index `idx`."""
        x = self.X[idx]
        y = self.Y[idx]
        s = self.S[idx]
        return x, y, s

