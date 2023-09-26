import matplotlib.pyplot as plt
import torch

from torch.optim import Optimizer


def train_loop(model_name:str, dataloader, model, loss_fn, optimizer: Optimizer, epochs: int, batch_runs=1):
    loss_eval = []
    model.train()
    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        for batch, (X, y) in enumerate(dataloader):
            X = X.float()
            y = y.float()
            run_loss = []

            for _ in range(batch_runs):
                # forward pass
                pred = model(X)

                # perform error calculation
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()

                # Update weights
                optimizer.zero_grad()
                optimizer.step()

                # Record learning
                run_loss.append(loss.item())

            # graph_res(run_loss)

            loss_eval.append(loss.item())

        # evaluate performance after each iteration
        # test_loop(test_data, model, loss_fn, 0.02)

            current_loss += loss.item()
            if batch % 10 == 0:
                print('Loss after mini-batch %5d: %.6f' %
                      (batch + 1, current_loss / 64))

        # current_loss = 0.0
    graph_res(loss_eval)


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
            X = X.float()
            y = y.float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (abs(pred - y) <= threshold).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    model.train()


def graph_res(data):
    indices = list(range(len(data)))
    plt.figure(figsize=(10, 6))
    plt.plot(indices, data, label='Training Loss', color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Batches')
    plt.legend()
    plt.grid(True)
    plt.show()
