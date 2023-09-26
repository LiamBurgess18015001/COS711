import torch
import sys
from torch.utils.data import DataLoader

from Build_Model.dataset import CosDataset
from Build_Model.build import train_loop
from Build_Model.Neural_Network import NeuralNetwork
from Run.Train_Test import train_test

import matplotlib.pyplot as plt

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
loss_fn = torch.nn.L1Loss()
learning_rate = 0.001
momentum = 0.999
batch_size = 128
batch_passes = 1
fix = True
epochs = 2500

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# train_loop(train_dataloader, model, loss_fn, optimizer, epochs, batch_passes)

# test_loop(test_dataloader, model, loss_fn)

train_test("SGD_L1", train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, batch_passes, fix)

print("Done!")

torch.save(model.state_dict(), "./Run/files/SGD_L1")

model = NeuralNetwork()
loss_fn = torch.nn.L1Loss()
learning_rate = 0.001
momentum = 0.999
batch_size = 128
batch_passes = 1
fix = True
epochs = 2500

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_test("ADAM_L1", train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, batch_passes, fix)

print("Done!")

torch.save(model.state_dict(), "./Run/files/ADAM_L1")

model = NeuralNetwork()
loss_fn = torch.nn.MSELoss()
learning_rate = 0.001
momentum = 0.999
batch_size = 128
batch_passes = 1
fix = True
epochs = 2500

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_test("SGD_MSE", train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, batch_passes, fix)

print("Done!")

torch.save(model.state_dict(), "./Run/files/SGD_MSE")

model = NeuralNetwork()
loss_fn = torch.nn.MSELoss()
learning_rate = 0.001
momentum = 0.999
batch_size = 128
batch_passes = 1
fix = True
epochs = 2500

train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_test("ADAM_MSE", train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs, batch_passes, fix)

print("Done!")

torch.save(model.state_dict(), "./Run/files/ADAM_MSE")
