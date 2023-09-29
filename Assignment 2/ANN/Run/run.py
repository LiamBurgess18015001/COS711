import sys

import torch
from torch.utils.data import DataLoader

from Build_Model.Neural_Network import NeuralNetwork
from Build_Model.dataset import CosDataset
from Run.Train_Test import train_test, do_test

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

training_data = CosDataset("./Build_Model/files/train_data.csv", "./Build_Model/files/train_labels.csv")
test_data = CosDataset("./Build_Model/files/test_data.csv", "./Build_Model/files/test_labels.csv")

model = NeuralNetwork()
# model.load_state_dict(torch.load("./Run/files/SGD_L1"))

loss_fn = torch.nn.MSELoss()
learning_rate = 0.0001
momentum = 0.78
batch_size = 256
batch_passes = 100
fix = True
epochs = 1
training_target = 70

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for i in range(10):
    training_target += 1
    train_test("L1_SGD", train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, batch_passes,
               fix, training_target)

# do_test(model, test_dataloader, loss_fn)

print("Done!")
