import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Optimizer


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_loop(dataloader, test_data, model, loss_fn, optimizer: Optimizer, epochs):
    loss_eval = []
    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X = X.float()
            y = y.float()

            # forward pass
            pred = model(X)

            # perform error calculation
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.zero_grad()

            # Record learning
            loss_eval.append(loss.item())

        # evaluate performance after each iteration
        test_loop(test_data, model, loss_fn, 0.002)

        # current_loss += loss.item()
        # if batch % 10 == 0:
        #     print('Loss after mini-batch %5d: %.6f' %
        #           (batch + 1, current_loss / 64))

        # current_loss = 0.0
    indices = list(range(len(loss_eval)))
    plt.figure(figsize=(10, 6))
    plt.plot(indices, loss_eval, label='Training Loss', color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Batches')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_loop(dataloader, model, loss_fn, threshold=0.02):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred - y) <= threshold).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
