import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from global_vars import columns

figs, ax = plt.subplots(4, 4, figsize=(12, 4))
ax = ax.flatten()
axs_count = 0

def min_max_scaling(grp, debug=False):
    scaler = MinMaxScaler()
    max = grp.max()
    min = grp.min()
    normalized_data = (grp - min) / (max - min)
    # normalized_data = scaler.fit_transform(grp.values.reshape(0, 1))
    if debug:
        plt.hist(normalized_data, bins=np.arange(-4, 4, 0.1), edgecolor="black")
        plt.show()
    return normalized_data


def z_score_non_zero(grp, debug=True):
    global ax
    global axs_count
    non_zeros = grp[grp != 0]
    mean = non_zeros.mean()
    std_dev = non_zeros.std()
    threshold = 3
    z_scores = (grp - mean) / std_dev
    # z_scores = z_scores.mask((abs(z_scores) > threshold), 0)
    if debug:
        ax[axs_count].hist(z_scores, bins=np.arange(-4, 4, 0.1), edgecolor="black")
        ax[axs_count].set_title(columns[axs_count+2])
        print(z_scores.mean())
        print(z_scores.std())
        axs_count += 1
    return z_scores


def drop_columns(data: pd.DataFrame):
    return data.drop(columns=['Latitude', 'Longitude', 'Value_co2_emissions_kt_by_country', 'Entity', 'Year'])
    # return data.drop(columns=['Latitude', 'Longitude', 'Value_co2_emissions_kt_by_country', 'Entity', 'Year',
    #                           'Electricity from nuclear (TWh)', 'Renewables (% equivalent primary energy)'])


def make_labels(rows, filename="./Build/files/test_labels.csv"):
    labels = rows['Value_co2_emissions_kt_by_country'].values.astype(np.float64) / 1000000
    ins = []
    # for i, lab in enumerate(labels):
    #     ins.append([i, lab])
    pd.DataFrame(labels).to_csv(f"{filename}", encoding="utf-8", index=False, header=False)


def train_test_validate_split(data):
    random_indices = random.sample(range(0, len(data)), math.ceil(len(data) / 10))
    test = data.iloc[random_indices]
    random_indices = random.sample(range(0, len(data)), math.ceil(len(data) / 20))
    validate = data.iloc[random_indices]
    train = data.drop(random_indices)
    return train, test, validate


data = pd.read_csv('./pre_processing/files/cleaned_data.csv')

# Scale
# data[columns[2:13]] = data[columns[2:13]].apply(min_max_scaling)
# data[columns[14:-2]] = data[columns[14:-2]].apply(min_max_scaling)

# Normalize
data[columns[2:13]] = data[columns[2:13]].apply(z_score_non_zero)
data[columns[14:-2]] = data[columns[14:-2]].apply(z_score_non_zero)

plt.tight_layout()
plt.show()

# Split and Label
train, test, validate = train_test_validate_split(data)

make_labels(test, "./Build_Model/files/test_labels.csv")
make_labels(train, "./Build_Model/files/train_labels.csv")
make_labels(validate, "./Build_Model/files/validate_labels.csv")

train = drop_columns(train)
test = drop_columns(test)
validate = drop_columns(validate)

train.to_csv("Build_Model/files/train_data.csv", index=False)
test.to_csv("Build_Model/files/test_data.csv", index=False)
validate.to_csv("Build_Model/files/validate_data.csv", index=False)
