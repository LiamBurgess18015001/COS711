import torch
from torch import nn


class NeuralNetwork(nn.Module):
    loss_fn = nn.MSELoss()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            self.float()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    # def init_hyperparameters(self, learning_rate: float, batch_size: int, epochs: int):
    #     self.epochs = epochs
    #     self.batch_size = batch_size
    #     self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def _train_loop(self, dataloader, model, optimizer):
        size = len(dataloader.dataset)

        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test_loop(self, dataloader):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_model(self, train_dataloader, test_dataloader):
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self._train_loop(train_dataloader)
            self._test_loop(test_dataloader)
        print("Done!")

    def save_model(self, model_name):
        torch.save(self.state_dict(), f'./Build/files/{model_name}.pth')

    def load_model(self, model_name):
        self.load_state_dict(torch.load(f'./Build/files/{model_name}.pth'))
        self.eval()
