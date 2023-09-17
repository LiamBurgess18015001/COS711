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


def make_labels(rows):
    labels = rows['Value_co2_emissions_kt_by_country'].values.astype(np.float64)
    ins = []
    for i, lab in enumerate(labels):
        ins.append([i, lab])
    pd.DataFrame(ins).to_csv("./Build/files/labels.csv", encoding="utf-8", index=False, header=False)


data[columns[2:-2]] = data[columns[2:-2]].apply(z_score_non_zero)
make_labels(data)
data = drop_columns(data)
data.to_csv("Build/files/preprocessed_data.csv", index=False)
