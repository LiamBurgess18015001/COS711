import torch
import sys
from torch.utils.data import DataLoader

from Build.dataset import CosDataset
from Build.build import NeuralNetwork, train_loop, test_loop

sys.setrecursionlimit(1000000)

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

training_data = CosDataset("./Build/files/train_data.csv", "./Build/files/train_labels.csv")
test_data = CosDataset("./Build/files/test_data.csv", "./Build/files/test_labels.csv")

model = NeuralNetwork()
loss_fn = torch.nn.L1Loss()
learning_rate = 0.01
momentum = 0.5
batch_size = 64
epochs = 12

train_dataloader = DataLoader(training_data, batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_loop(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)

# test_loop(test_dataloader, model, loss_fn)

print("Done!")
