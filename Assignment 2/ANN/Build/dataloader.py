import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CosDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data_set = pd.read_csv("./Build/files/preprocessed_data.csv", encoding="utf-8")
        self.data_labels = pd.read_csv("./Build/files/label_names.csv", encoding="utf-8")
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

    def fetch_row(self, row_index: str):
        country = row_index[:row_index.find("_")]
        year = row_index[row_index.find("_") + 1:]
        result = (self.data_set["Year"] == year) & (self.data_set["Entity"] == country)
        row = self.data_set.loc[result.index[0]]
        row = pd.DataFrame(row).transpose().drop(columns=["Entity", "Year"])
        return torch.tensor(row.values.astype(np.float64))
