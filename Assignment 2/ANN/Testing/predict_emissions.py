import pandas as pd
from Build_Model.Neural_Network import NeuralNetwork, build_model
import torch
import matplotlib.pyplot as plt

from global_vars import columns

data = pd.read_csv("./pre_processing/files/empty_labels.csv")


def z_score_non_zero(grp, debug=True):
    non_zeros = grp[grp != 0]
    mean = non_zeros.mean()
    std_dev = non_zeros.std()
    z_scores = (grp - mean) / std_dev
    return z_scores


data[columns[2:13]] = data[columns[2:13]].apply(z_score_non_zero)
data[columns[14:-2]] = data[columns[14:-2]].apply(z_score_non_zero)


def drop_columns(data: pd.DataFrame):
    return data.drop(columns=['Latitude', 'Longitude', 'Value_co2_emissions_kt_by_country', 'Entity', 'Year'])


# data = drop_columns(data)

# model = NeuralNetwork(build_model("pyramid", "prelu", "large"))
# model.load_state_dict(torch.load("./Run/files/Optim_Model_1_2_3"))

south_africa = drop_columns(data[data['Entity'] == "South Africa"])
input_data = torch.tensor(south_africa.iloc[0].values, dtype=torch.float64)

rsa_year = []
# for i in range(5):
#     rsa_year.append()


denmark = data[data['Entity'] == "Denmark"]
den_year = []
# for i in range(5):
#     den_year.append()

portugal = data[data['Entity'] == "Portugal"]
port_year = []
# for i in range(5):
#     port_year.append()

fig, ax = plt.subplots()
x = [2023, 2024, 2025, 2026, 2027]
ax.plot(x, rsa_year, marker='o', label='South Africa')
ax.plot(x, den_year, marker='o', label='Denmark')
ax.plot(x, port_year, marker='o', label='Portugal')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
plt.show()