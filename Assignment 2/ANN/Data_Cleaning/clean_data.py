# its been 5 years lol
import numpy as np
import pandas as pd
import scipy as sp
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.impute import KNNImputer

from global_vars import columns


def convert_to_float(value):
    if isinstance(value, str) and ',' in value:
        return float(value.replace(',', '.'))
    return value


def interpolate_group(group):
    for col in group.columns:
        if col == "Value_co2_emissions_kt_by_country":
            continue

        if group[col].isna().sum() == 0:
            continue

        x_values = np.arange(len(group[col]))
        # Filter NaN values and interpolate using cubic spline
        mask = group[col].isna()

        if group[col].isna().sum() > len(group[col]) * 0.65:
            group.loc[mask, col] = 0
            continue

        x = x_values[~mask]
        y = group[col].dropna().values
        try:
            cs = sp.interpolate.CubicSpline(x, y, bc_type='not-a-knot', extrapolate=True)

            max = y.max() * 1.5
            min = y.min() * 0.5

            # Interpolate missing values and replace them in the group
            group.loc[mask, col] = cs(x_values[mask])
            for i, val in enumerate(group[col]):
                if val > max:
                    group[col][i] = max
                elif val < min:
                    group[col][i] = min

        except Exception as e:
            group.loc[mask, col] = 0
            print("pain")

    return group


data = pd.read_csv('Data_Cleaning/co2_emissions/emission_data.csv')

data_no_countries = pd.read_table('Data_Cleaning/co2_emissions/emission_data_modified.csv', sep=',')
mt = MCARTest(method="little")
if mt.little_mcar_test(data_no_countries) > 0.05:
    print('Is MCAR')
else:
    print('Not MCAR')

knn_imputer = KNNImputer(n_neighbors=1)
imputed_data = knn_imputer.fit_transform(data_no_countries)
pd.DataFrame(imputed_data, columns=columns[2:])

empty_labels = data[data['Value_co2_emissions_kt_by_country'].isna()]
groups_with_all_nans = empty_labels.groupby("Entity").apply(lambda grp: grp.isna().all())
axis_a, axis_b = groups_with_all_nans.axes
for axi_a in axis_a:
    for axi_b in axis_b[1:]:
        if groups_with_all_nans.loc[axi_a, axi_b]:
            empty_labels.loc[data["Entity"] == axi_a, axi_b] = 0
empty_labels = empty_labels.groupby("Entity").apply(interpolate_group).reset_index(drop=True)
empty_labels.to_csv("pre_processing/files/empty_labels.csv", index=False)

data = data[~data['Value_co2_emissions_kt_by_country'].isna()]
groups_with_all_nans = data.groupby("Entity").apply(lambda grp: grp.isna().all())
axis_a, axis_b = groups_with_all_nans.axes
for axi_a in axis_a:
    for axi_b in axis_b[1:]:
        if groups_with_all_nans.loc[axi_a, axi_b]:
            data.loc[data["Entity"] == axi_a, axi_b] = 0

data = data.groupby("Entity").apply(interpolate_group).reset_index(drop=True)
data.to_csv("pre_processing/files/cleaned_data.csv", index=False)

nan_counts = data.isna().sum() * 100 / len(data)
print(nan_counts)
#################################################
