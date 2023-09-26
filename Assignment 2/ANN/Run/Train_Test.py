import random
import matplotlib.pyplot as plt
import torch

file = open("./Run/files/run_info.txt", 'a+', encoding="utf-8")


def train_test(model_name: str, train_loader, test_loader, model: torch.nn, loss_fn, optimizer: torch.optim.Optimizer,
               epochs: int, batch_runs=1, fix=True):
    file.write(f"{model_name}\n")
    loss_eval = []
    model.train()
    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0
        train_avg = 0.00
        for batch, (X, y) in enumerate(train_loader):
            # X = X.float()
            # y = y.float()
            # run_loss = []

            if not fix:
                batch_runs_rand = random.randint(1, batch_runs)
            else:
                batch_runs_rand = batch_runs
            # print(f"Run batch {batch_runs_rand}")

            for _ in range(batch_runs_rand):
                # forward pass
                pred = model(X)

                # perform error calculation
                train_error = loss_fn(pred, y)

                # Backpropagation
                train_error.backward()
                optimizer.step()

                # Update weights
                optimizer.zero_grad()

                # Record learning
                # run_loss.append(loss.item())
            train_avg += ((abs(pred - y) <= 0.002).sum().item())

            # loss_eval.append(loss.item())

            current_loss += train_error.item()
        train_avg = train_avg * 100 / len(train_loader.dataset)
        print(f"Train Average: {train_avg}")

        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            avg = 0.00
            for X, y in test_loader:
                pred = model(X)
                val_epoch_loss += loss_fn(pred, y).item()
                avg += ((abs(pred - y) <= 0.02).sum().item())
                print(
                    f'Epoch {epoch + 0:03}: | Train Loss: {current_loss / len(train_loader):.5f} | Test Loss: {val_epoch_loss / len(test_loader):.5f}')
            avg = avg * 100 / len(test_loader.dataset)
            print(f"Test Average: {avg}")
            # fig, ax = plt.subplots()
            # ax.plot(y, 'ro', label='ref')
            # ax.plot(pred, 'bo', label='pred')
            # plt.show()


            file.write(f'epoch: {epoch}\nTrain Acccuracy: {train_avg}\nTest Accuracy: {avg}\n')
    file.write("###############################################################\n")