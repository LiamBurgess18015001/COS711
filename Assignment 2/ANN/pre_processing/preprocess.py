import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

from global_vars import columns

data = pd.read_csv('Data_Cleaning/co2_emissions/emission_data.csv')


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
    return data.drop(columns=['Latitude', 'Longitude'])


def make_labels(grps):
    labels = []
    label_names = []
    for i, grp in enumerate(grps):
        labels.append([grp[0], i])
        for year in grp[1]['Year']:
            label_names.append([f'{grp[0]}_{year}', i])

    with open("Build/files/labels.csv", "w+", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(labels)

    with open("Build/files/label_names.csv", "w+", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(label_names)


data[columns[2:-2]] = data[columns[2:-2]].apply(z_score_non_zero)
data = drop_columns(data)
make_labels(data.groupby("Entity"))
data.to_csv("Build/files/preprocessed_data.csv")
