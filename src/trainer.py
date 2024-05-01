import torch

from torch import nn
from network import CNN
from optimizer import Optimizer
from hyperparameters import BATCH_SIZE
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: CNN, optimizer: Optimizer, loss_fn: nn.CrossEntropyLoss, train_loader: DataLoader, device: torch.device) -> None:
        self.model          = model
        self.optimizer      = optimizer
        self.loss_fn        = loss_fn
        self.train_loader   = train_loader
        self.device         = device

        # Set default value.
        self.current_epoch  = 0
        self.loss_history   = []

    def train(self, epochs: int) -> None:
        # Set model to training mode.
        self.model.train()

        for _ in range(epochs):
            # Set default value.
            counter     = 0
            total_loss  = 0
            total_acc   = 0
            total       = len(self.train_loader)
            
            for image, label in self.train_loader:
                # Set defualt data type.
                image   : torch.Tensor
                label   : torch.Tensor
                loss    : torch.Tensor

                # Move data to device.
                image = image.to(device=self.device)
                label = label.to(device=self.device)

                # Get prediction.
                output = self.model(image)

                # Calculate loss.
                loss = self.loss_fn(output, label)
                total_loss += loss.item()

                # Calculate accuracy.
                acc = sum(label == torch.argmax(output, dim=1)).item() / BATCH_SIZE
                total_acc += acc

                # Save loss history.
                self.loss_history.append(loss)

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

            # Update current epoch.
            self.current_epoch += 1

        # Set model to evaluation mode.
        self.model.eval()

    def get_loss_accuracy(self) -> list:
        # Move loss data to cpu.
        history = torch.Tensor(self.loss_history).cpu()
    
        # Conver to list.    
        history = history.tolist()

        return history