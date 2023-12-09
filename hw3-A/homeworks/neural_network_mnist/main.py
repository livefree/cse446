# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        a = torch.sqrt(torch.tensor(1.0 / d))

        self.W0 = Parameter(torch.Tensor(d, h).uniform_(-a, a))
        self.b0 = Parameter(torch.Tensor(h).uniform_(-a, a))
        self.W1 = Parameter(torch.Tensor(h, k).uniform_(-a, a))
        self.b1 = Parameter(torch.Tensor(k).uniform_(-a, a))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # Apply the first linear transformation manually.
        x = torch.mm(x, self.W0) + self.b0
        # Apply the ReLU activation function
        x = relu(x)
        # Apply the second linear transformation manually.
        x = torch.mm(x, self.W1) + self.b1
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        a = torch.sqrt(torch.tensor(1.0 / d))

        self.W0 = Parameter(torch.Tensor(d, h0).uniform_(-a, a))
        self.b0 = Parameter(torch.Tensor(h0).uniform_(-a, a))
        self.W1 = Parameter(torch.Tensor(h0, h1).uniform_(-a, a))
        self.b1 = Parameter(torch.Tensor(h1).uniform_(-a, a))
        self.W2 = Parameter(torch.Tensor(h1, k).uniform_(-a, a))
        self.b2 = Parameter(torch.Tensor(k).uniform_(-a, a))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # Apply the first linear transformation manually.
        x = torch.mm(x, self.W0) + self.b0
        # Apply the first ReLU activation function
        x = relu(x)
        # Apply the second linear transformation manually.
        x = torch.mm(x, self.W1) + self.b1
        # Apply the second ReLU activation function
        x = relu(x)
        # Apply the third linear transformation manually.
        x = torch.mm(x, self.W2) + self.b2
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()  #Training mode
    loss_fn = torch.nn.functional.cross_entropy
    epoch_losses = []  # List to record average loss per epoch

    for epoch in range(100):  # Adjust range
        batch_losses = []  # List to store losses for each batch
        correct = 0
        total = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(x)  # Forward pass
            loss = loss_fn(outputs, y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            batch_losses.append(loss.item())  # Record batch loss
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        epoch_loss = sum(batch_losses) / len(batch_losses)  # Average loss for the epoch
        epoch_losses.append(epoch_loss)  # Record epoch loss
        epoch_accuracy = correct / total  # Calculate accuracy for the epoch

        # Check accuracy
        if epoch_accuracy >= 0.99:
            break  # Stop training if goal accuracy is achieved

    return epoch_losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # Define hyperparameters
    h = 64  # Hidden units for F1
    h0 = h1 = 32 # Hidden units for F2
    d = x.shape[1]  # Input dimension
    k = len(torch.unique(y))  # Number of classes

    # Initialize F1 and F2 models
    model_f1 = F1(h, d, k)
    model_f2 = F2(h0, h1, d, k) 

    # Define optimizers for F1 and F2
    optimizer_f1 = torch.optim.Adam(model_f1.parameters())
    optimizer_f2 = torch.optim.Adam(model_f2.parameters())

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    # test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    # F1
    losses_f1 = train(model_f1, optimizer_f1, train_loader)
    accuracy_f1, loss_f1 = evaluate(model_f1, x_test, y_test)
    print(f'F1 Test Accuracy: {accuracy_f1}, F1 Test Loss: {loss_f1}')

    # F2
    losses_f2 = train(model_f2, optimizer_f2, train_loader)
    accuracy_f2, loss_f2 = evaluate(model_f2, x_test, y_test)
    print(f'F2 Test Accuracy: {accuracy_f2}, F2 Test Loss: {loss_f2}')

    # Report number of parameters
    print(f'F1 Total Parameters: {count_parameters(model_f1)}')
    print(f'F2 Total Parameters: {count_parameters(model_f2)}')

    # Plot
    plt.figure(1)
    plt.plot(losses_f1)
    plt.title('Training Loss for F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.figure(2)
    plt.plot(losses_f2)
    plt.title('Training Loss for F2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.show()

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        loss = torch.nn.functional.cross_entropy(outputs, y_test)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / total
    return accuracy, loss.item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    main()
