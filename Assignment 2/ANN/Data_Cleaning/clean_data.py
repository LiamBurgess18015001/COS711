import numpy as np
import pandas as pd
import scipy as sp
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.impute import KNNImputer

columns = ['Entity', 'Year', 'Access to electricity (% of population)', 'Access to clean fuels for cooking',
           'Renewable-electricity-generating-capacity-per-capita', 'Financial flows to developing countries (US $)',
           'Renewable energy share in the total final energy consumption (%)', 'Electricity from fossil fuels (TWh)',
           'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)',
           'Low-carbon electricity (% electricity)', 'Primary energy consumption per capita (kWh/person)',
           'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 'Value_co2_emissions_kt_by_country',
           'Renewables (% equivalent primary energy)', 'gdp_growth', 'gdp_per_capita', 'Density\\n(P/Km2)',
           'Land Area(Km2)', 'Latitude', 'Longitude']


def convert_to_float(value):
    if isinstance(value, str) and ',' in value:
        return float(value.replace(',', '.'))
    return value


data = pd.read_csv('./co2_emissions/emission_data.csv')

nan_counts = data.isna().sum() * 100 / len(data)

data_no_countries = pd.read_table('./co2_emissions/emission_data_modified.csv', sep=',')
mt = MCARTest(method="little")
if mt.little_mcar_test(data_no_countries) > 0.05:
    print('Is MCAR')
else:
    print('Not MCAR')

knn_imputer = KNNImputer(n_neighbors=1)
imputed_data = knn_imputer.fit_transform(data_no_countries)
pd.DataFrame(imputed_data, columns=columns[2:])

groups_with_all_nans = data.groupby("Entity").apply(lambda grp: grp.isna().all())
pd.DataFrame(groups_with_all_nans)
axis_a, axis_b = groups_with_all_nans.axes
for axi_a in axis_a:
    for axi_b in axis_b[1:]:
        if groups_with_all_nans.loc[axi_a, axi_b]:
            data.loc[data["Entity"] == axi_a, axi_b] = 0


def interpolate_group(group):
    for col in group.columns:
        if group[col].isna().sum() == 0:
            continue
        x_values = np.arange(len(group[col]))
        # Filter NaN values and interpolate using cubic spline
        mask = group[col].isna()
        x = x_values[~mask]
        y = group[col].dropna().values
        try:
            cs = sp.interpolate.CubicSpline(x, y, bc_type='not-a-knot', extrapolate=True)

            # Interpolate missing values and replace them in the group
            group.loc[mask, col] = cs(x_values[mask])
        except Exception as e:
            print("pain")

    return group


data = data.groupby("Entity").apply(interpolate_group).reset_index(drop=True)
df = pd.DataFrame(data)
print(1)
