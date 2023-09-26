import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from global_vars import columns

data = pd.read_csv('./pre_processing/files/cleaned_data.csv')


def minMax_scaling(grp, debug=True):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(grp.values.reshape(-1, 1))
    if debug:
        plt.hist(normalized_data, bins=np.arange(-4, 4, 0.1), edgecolor="black")
        plt.show()
    return normalized_data


def z_score_non_zero(grp, debug=False):
    non_zeros = grp  # [grp != 0]
    mean = non_zeros.mean()
    std_dev = non_zeros.std()
    threshold = 3
    z_scores = (grp - mean) / std_dev
    z_scores = z_scores.mask((abs(z_scores) > threshold), 0)
    if debug:
        plt.hist(z_scores, bins=np.arange(-4, 4, 0.1), edgecolor="black")
        plt.show()
        if abs(z_scores.mean()) * 100 > 20:
            print("Adjust Z")
        print()
        print(z_scores.std())
    return z_scores


def drop_columns(data: pd.DataFrame):
    return data.drop(columns=['Latitude', 'Longitude', 'Value_co2_emissions_kt_by_country', 'Entity', 'Year'])


def make_labels(rows, filename="./Build/files/test_labels.csv"):
    labels = rows['Value_co2_emissions_kt_by_country'].values.astype(np.float64) / 1000000
    ins = []
    for i, lab in enumerate(labels):
        ins.append([i, lab])
    pd.DataFrame(ins).to_csv(f"{filename}", encoding="utf-8", index=False, header=False)


def train_test_split(data):
    random_indices = random.sample(range(0, len(data)), math.ceil(len(data)/4))
    test = data.iloc[random_indices]
    train = data.drop(random_indices)
    return train, test


data[columns[2:13]] = data[columns[2:13]].apply(z_score_non_zero)
data[columns[15:-2]] = data[columns[15:-2]].apply(z_score_non_zero)

train, test = train_test_split(data)
make_labels(test, "./Build_Model/files/test_labels.csv")
make_labels(train, "./Build_Model/files/train_labels.csv")
train = drop_columns(train)
test = drop_columns(test)
train.to_csv("Build_Model/files/train_data.csv", index=False)
test.to_csv("Build_Model/files/test_data.csv", index=False)
