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
learning_rates = [0.00001, 0.0001, 0.0005, 0.0008, 0.001, 0.01, 0.1]
momentums = [0.99, 0.9, 0.88, 0.83, 0.8, 0.78, 0.75]
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
batch_passes = [1, 2, 3]
fix = True
epochs = 100
training_target = 70

with open("./Run/files/optim_results.txt", 'a+', encoding="utf-8") as file:
    for learning_rate in learning_rates:
        for momentum in momentums:
            for batch_size in batch_sizes:
                validation_dataset = DataLoader(validation_data, batch_size)
                test_dataloader = DataLoader(test_data, batch_size)
                for batch_passe in batch_passes:
                    model.load_state_dict(torch.load("./Run/files/SGD_L1_1p"))
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                    avg = train_test("Optim_Model", validation_dataset, test_dataloader, model,
                                     loss_fn, optimizer, epochs, batch_passe, fix, training_target)

                    file.write(f"""
                    Average: {avg}
                    Learning Rate: {learning_rate}
                    Momentum: {momentum}
                    Batch Size: {batch_size}
                    Batch Passes: {batch_passe}""")

print("Done!")
