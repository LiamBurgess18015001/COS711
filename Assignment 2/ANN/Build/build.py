import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    loss_fn = nn.MSELoss()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = None
        self.epochs = None
        self.optimizer = None
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def init_hyperparameters(self, learning_rate: float, batch_size: int, epochs: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def train_loop(self, dataloader):
        size = len(dataloader.dataset)

        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_model(self, train_dataloader, test_dataloader):
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(train_dataloader)
            self.test_loop(test_dataloader)
        print("Done!")

    def save_model(self, model_name):
        torch.save(self.state_dict(), f'./Build/files/{model_name}.pth')

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(f'./Build/files/{model_name}.pth'))
        self.model.eval()
