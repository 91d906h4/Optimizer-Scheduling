import torch

from torch import nn
from network import CNN
from optimizer import Optimizer
from hyperparameters import BATCH_SIZE
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: CNN, optimizer: Optimizer, loss_fn: nn.CrossEntropyLoss, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> None:
        self.model              = model
        self.optimizer          = optimizer
        self.loss_fn            = loss_fn
        self.train_loader       = train_loader
        self.test_loader        = test_loader
        self.device             = device

        # Set default value.
        self.current_epoch      = 0
        self.train_loss_history = []
        self.train_acc_history  = []
        self.test_loss_history  = []
        self.test_acc_history   = []

    def train(self, epochs: int) -> None:
        # Set model to training mode.
        self.model.train()

        for _ in range(epochs):
            # Set default value.
            counter     = 0
            total_loss  = 0
            total_acc   = 0
            total       = len(self.train_loader)
            
            for x, y in self.train_loader:
                # Set defualt data type.
                x       : torch.Tensor
                y       : torch.Tensor
                loss    : torch.Tensor

                # Move data to device.
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Get prediction.
                y_pred = self.model(x)

                # Calculate loss.
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()

                # Calculate accuracy.
                acc = sum(y == torch.argmax(y_pred, dim=1)).item() / BATCH_SIZE
                total_acc += acc

                # Save accuracy and loss history.
                self.train_loss_history.append(loss)
                self.train_acc_history.append(acc)

                # Select optimizer.
                self.optimizer.select(epoch=self.current_epoch)

                # Update model.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update counter.
                counter += 1

                # Print training message.
                print(
                    f"Epoch {self.current_epoch} | "
                    f"Progress: {counter / total * 100:.3f}% | "
                    f"Loss: {total_loss / counter:.3f} | ",
                    f"Accuracy: {total_acc / counter * 100:.3f}%",
                    end="\r"
                )

            # Print new line.
            print()

            # Test model.
            self.test()

            # Update current epoch.
            self.current_epoch += 1

        # Set model to evaluation mode.
        self.model.eval()

    @torch.no_grad()
    def test(self) -> None:
        # Set defualt values.
        counter     = 0
        total_loss  = 0
        total_acc   = 0
        total       = len(self.test_loader)

        for x, y in self.test_loader:
            # Set defualt data types.
            x       : torch.Tensor
            y       : torch.Tensor
            loss    : torch.Tensor

            # Move data to device.
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            # Get prediction.
            y_pred = self.model(x)

            # Calculate loss.
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()

            # Calculate accuracy.
            acc = sum(y == torch.argmax(y_pred, dim=1)).item() / BATCH_SIZE
            total_acc += acc

            # Save accuracy and loss history.
            self.test_loss_history.append(loss)
            self.test_acc_history.append(acc)

            # Update counter.
            counter += 1

            # Print training message.
            print(
                f"Test | "
                f"Progress: {counter / total * 100:.3f}% | "
                f"Loss: {total_loss / counter:.3f} | ",
                f"Accuracy: {total_acc / counter * 100:.3f}%",
                end="\r"
            )

        # Print new line.
        print()

    def get_train_history(self) -> tuple[list, list]:
        # Move loss data to cpu.
        train_loss_history = torch.Tensor(self.train_loss_history).cpu()
        train_acc_history = torch.Tensor(self.train_acc_history).cpu()
    
        # Conver to list.    
        train_loss_history = train_loss_history.tolist()
        train_acc_history = train_acc_history.tolist()

        return train_loss_history, train_acc_history

    def get_test_history(self) -> tuple[list, list]:
        # Move loss data to cpu.
        test_loss_history = torch.Tensor(self.test_loss_history).cpu()
        test_acc_history = torch.Tensor(self.test_acc_history).cpu()
    
        # Conver to list.    
        test_loss_history = test_loss_history.tolist()
        test_acc_history = test_acc_history.tolist()

        return test_loss_history, test_acc_history