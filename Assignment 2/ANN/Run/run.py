import torch
from torch.utils.data import DataLoader

from Build.build import NeuralNetwork
from Build.dataset import CosDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

training_data = CosDataset("./Build/files/train_data.csv", "./Build/files/train_labels.csv")
test_data = CosDataset("./Build/files/test_data.csv", "./Build/files/test_labels.csv")

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

neural_network = NeuralNetwork()
neural_network.init_hyperparameters(0.001, 64, 12)
neural_network.train_model(train_dataloader, test_dataloader)
