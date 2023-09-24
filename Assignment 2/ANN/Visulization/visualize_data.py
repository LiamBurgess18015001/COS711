import pandas as pd

test_data = pd.read_csv("./Build/files/train_data.csv", encoding="utf-8").iloc[0:20]
label_data = pd.read_csv("./Build/files/train_labels.csv", encoding="utf-8").iloc[0:20]

print(test_data)
