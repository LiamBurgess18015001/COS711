import sys

import torch
from torch.utils.data import DataLoader

from Build_Model.Neural_Network import NeuralNetwork
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

model = NeuralNetwork().to(device)
print(model)

validation_data = CosDataset("./Build_Model/files/validate_data.csv", "./Build_Model/files/validate_labels.csv")
test_data = CosDataset("./Build_Model/files/test_data.csv", "./Build_Model/files/test_labels.csv")

model = NeuralNetwork()

loss_fn = torch.nn.L1Loss()
learning_rates = [0.00001, 0.0001, 0.0005, 0.0008, 0.001, 0.01225, 0.0225, 0.05, 0.08, 0.1]
momentums = [0.99, 0.95, 0.9, 0.88, 0.83, 0.8, 0.78, 0.75]
batch_sizes = [64, 128, 256]
batch_passes = [1]
fix = True
epochs = 200
training_target = 69

# with open("./Run/files/optim_results_static.txt", 'a+', encoding="utf-8") as file:
#     for learning_rate in learning_rates:
#         for momentum in momentums:
#             for batch_size in batch_sizes:
#                 validation_dataset = DataLoader(validation_data, batch_size)
#                 test_dataloader = DataLoader(test_data, batch_size)
#                 for batch_passe in batch_passes:
#                     maximum = 0
#                     for _ in range(10):
#                         model.load_state_dict(torch.load("./Run/files/SGD_L1_Test"))
#                         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
#                         maximum += train_test("Optim_Model", validation_dataset, test_dataloader, model,
#                                               loss_fn, optimizer, epochs, batch_passe, fix, training_target)
#
#                     file.write(f"""
# Average: {maximum / 10}
# Learning Rate: {learning_rate}
# Momentum: {momentum}
# Batch Size: {batch_size}
# Batch Passes: {batch_passe}""")

validation_dataset = DataLoader(validation_data, 128)
test_dataloader = DataLoader(test_data, 128)

with open("./Run/files/optim_results_dynamic.txt", 'a+', encoding="utf-8") as file:
    learning_rate = 0.9
    for _ in range(20):
        learning_rate -= 0.011
        momentum = 0.999
        for _ in range(25):
            momentum -= 0.01
            for _ in range(10):
                maximum = 0
                model.load_state_dict(torch.load("./Run/files/SGD_L1_Test"))
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                maximum += train_test("Optim_Model", validation_dataset, test_dataloader, model,
                                 loss_fn, optimizer, epochs, 1, fix, training_target)

            file.write(f"""
Average: {maximum/10}
Learning Rate: {learning_rate}
Momentum: {momentum}
""")

print("Done!")
