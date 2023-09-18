import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CosDataset(Dataset):
    def __init__(self, datafile: str, label_file: str, transform=None, target_transform=None):
        self.data_set = pd.read_csv(datafile, encoding="utf-8")
        self.data_labels = pd.read_csv(label_file, encoding="utf-8", header=None)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        row = self.fetch_row(self.data_labels.iloc[idx, 0])
        label = self.data_labels.iloc[idx, 1]
        if self.transform:
            row = self.transform(row)
        if self.target_transform:
            label = self.target_transform(label)
        return row, label

    def fetch_row(self, row_index: int):
        row = self.data_set.iloc[row_index]
        row = pd.DataFrame(row).transpose()
        return torch.tensor(row.values.astype('float32'))
