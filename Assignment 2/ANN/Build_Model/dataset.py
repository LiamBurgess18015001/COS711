import pandas as pd
import torch
from torch.utils.data import Dataset


class CosDataset(Dataset):
    def __init__(self, datafile: str, label_file: str, transform=None, target_transform=None):
        self.data_set = pd.read_csv(datafile, encoding="utf-8", skiprows=1, header=None)
        self.data_labels = pd.read_csv(label_file, encoding="utf-8", skiprows=1, header=None)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        row = torch.tensor(self.data_set.iloc[idx].transpose(), dtype=torch.float64)
        label = torch.tensor([self.data_labels.iloc[idx][1]], dtype=torch.float64)
        if self.transform:
            row = self.transform(row)
        if self.target_transform:
            label = self.target_transform(label)
        return row, label
