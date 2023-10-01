import random

import torch

file = open("./Run/files/run_info.txt", 'a+', encoding="utf-8")


def train_test(model_name: str,
               train_loader,
               test_loader,
               model: torch.nn,
               loss_fn,
               optimizer: torch.optim.Optimizer,
               epochs: int,
               batch_runs=1,
               fix=True,
               training_target=64,
               threshold=0.02,
               learning_decay=0,
               momentum_decay=0) -> (float, float):
    train_maximum = 65
    test_max = 65
    file.write(f"{model_name}\n")
    model.train()
    for epoch in range(epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        if epoch % 20 == 0:
            print(
                f"Learning Rate Old: {optimizer.param_groups[0]['lr']}\nNew: {optimizer.param_groups[0]['lr'] - learning_decay}")
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] - learning_decay

            print(
                f"Momentum Old: {optimizer.param_groups[0]['momentum']}\nNew: {optimizer.param_groups[0]['momentum'] - momentum_decay}")
            optimizer.param_groups[0]['momentum'] = optimizer.param_groups[0]['momentum'] - momentum_decay

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
            train_avg += ((abs(pred - y) <= threshold).sum().item())

            # loss_eval.append(loss.item())

            # print(f'Train Loss: {current_loss / len(train_loader):.5f}')

            current_loss += train_error.item()
        train_avg = train_avg * 100 / len(train_loader.dataset)

        if train_avg > train_maximum:
            train_maximum = train_avg

        print(f"Train Average: {train_avg}")

        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            avg = 0.00
            for X, y in test_loader:
                pred = model(X)
                val_epoch_loss += loss_fn(pred, y).item()
                avg += ((abs(pred - y) <= threshold).sum().item())
                print(
                    f'Epoch {epoch + 0:03}: | Train Loss: {current_loss / len(train_loader):.5f} | Test Loss: {val_epoch_loss / len(test_loader):.5f}')
            avg = avg * 100 / len(test_loader.dataset)
            print(f"Test Average: {avg}")

            if avg > test_max:
                test_max = avg

            file.write(f'epoch: {epoch}\nTrain Acccuracy: {train_avg}\nTest Accuracy: {avg}\n')

            if avg >= training_target:
                print("Training Target Hit")
                torch.save(model.state_dict(), f"./Run/files/{model_name}")
                # torch.save(model.state_dict(), "./Run/files/SGD_L1_optim")
                file.write(f'epoch: {epoch}\nTrain Acccuracy: {train_avg}\nTest Accuracy: {avg}\n')
                break
    file.write("###############################################################\n")
    return train_maximum, test_max


def do_test(model, test_loader, loss_fn):
    with torch.no_grad():
        val_epoch_loss = 0
        model.eval()
        avg = 0.00
        for X, y in test_loader:
            pred = model(X)
            val_epoch_loss += loss_fn(pred, y).item()
            avg += ((abs(pred - y) <= 0.02).sum().item())
        avg = avg * 100 / len(test_loader.dataset)
        print(f"Test Average: {avg}")
