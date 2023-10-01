import pandas as pd
from scipy import stats

data = pd.read_csv("./Testing/files/training_data.csv", encoding="utf-8")
# large = data[data["size"] == "l"]['test'].values
# small = data[data["size"].isin(["m", "s"])]['test'].values
#
# t_stat, p_value = stats.ttest_ind(large, small)
#
# # Set the desired confidence level
# confidence_level = 0.95
#
# # Calculate the significance level (alpha)
# alpha = 1 - confidence_level
#
# # Interpret the results
# if p_value <= alpha:
#     print(f"Reject the null hypothesis. p-value = {p_value:.4f}")
# else:
#     print(f"Fail to reject the null hypothesis. p-value = {p_value:.4f}")
#
# print(p_value)

prelu = data[data["actv"] == "p"]['gen_error'].values
not_prelu = data[data["actv"] == "o"]['gen_error'].values

t_stat, p_value = stats.ttest_ind(prelu, not_prelu)

# Set the desired confidence level
confidence_level = 0.95

# Calculate the significance level (alpha)
alpha = 1 - confidence_level

# Interpret the results
if p_value <= alpha:
    print(f"Reject the null hypothesis. p-value = {p_value:.4f}")
else:
    print(f"Fail to reject the null hypothesis. p-value = {p_value:.4f}")

print(p_value)
