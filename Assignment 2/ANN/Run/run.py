import sys

import torch
from torch.utils.data import DataLoader

from Build_Model.Neural_Network import NeuralNetwork, build_model
from Build_Model.dataset import CosDataset
from Run.Train_Test import train_test

sys.setrecursionlimit(1000000)
torch.set_default_tensor_type(torch.DoubleTensor)

# torch.manual_seed(69)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

training_data = CosDataset("./Build_Model/files/train_data.csv", "./Build_Model/files/train_labels.csv")
test_data = CosDataset("./Build_Model/files/test_data.csv", "./Build_Model/files/test_labels.csv")

loss_fn = torch.nn.L1Loss()
learning_rate = 0.001
momentum = 0.99
# learning_rate = 0.001
# momentum = 0.78
batch_size = 128
batch_passes = 1
fix = True
epochs = 250
training_target = 63

model = NeuralNetwork(build_model("pyramid", "sigmoid", "large")).to(device)
print(model)
model_name = "L1_SGD_Pyramid_Large_Sigmoid"

train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.001)

total_train_avg = 0
total_test_avg = 0

for i in range(1, 11):
    # batch_size = int(batch_size / i)

    # learning_rate = 0.9
    training_target += 1
    train_avg, test_avg = train_test(model_name,
                                     train_dataloader,
                                     test_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     epochs,
                                     batch_passes,
                                     fix,
                                     training_target,
                                     )

print(f'Final Training Average: {total_train_avg / 10}')
print(f'Final Testing Average: {total_test_avg / 10}')

with open("./Run/files/config_res.txt", "a+", encoding="utf-8") as file:
    file.write(f"{model_name}")
    file.write(f'Final Training Average: {total_train_avg / 10}')
    file.write(f'Final Testing Average: {total_test_avg / 10}')

print("Done!")
