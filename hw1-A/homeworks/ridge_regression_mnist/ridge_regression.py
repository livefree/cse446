import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import load_dataset, problem


@problem.tag("hw1-A")
def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), targets (`y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `x` and `y` with hyperparameter `_lambda`.
    """
    n, d = x.shape

    # Normalize the columns
    # norms = np.linalg.norm(x, axis=0) + 1.0
    norms = np.zeros(x.shape[1])
    norms[:] = 1.0  # test purpose
    x_normalized = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_normalized[:, i] = x[:, i] / norms[i]

    # xTx = x_normalized.T.dot(x_normalized)
    xTx = np.matmul(x_normalized.T, x_normalized)
    regularization_matrix = _lambda * np.identity(d)

    # Solve the linear system of equations
    # weight = scipy.linalg.solve(xTx + regularization_matrix, x.T.dot(y))
    weight = scipy.linalg.solve(xTx + regularization_matrix, np.matmul(x.T, y))

    return weight


@problem.tag("hw1-A")
def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), and weight matrix (`w`) to generate predicated class for each observation in x.

    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        w (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    norms = np.zeros(x.shape[1])
    norms[:] = 1.0  # test purpose
    x_normalized = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_normalized[:, i] = x[:, i] / norms[i]
    return np.argmax(np.matmul(x_normalized, w), axis=1)


@problem.tag("hw1-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot


def plotImage(x: np.ndarray, y: np.ndarray, y_read: np.ndarray):
    n = x.shape[0]
    size = 28
    # Define the limit for the number of plots
    plot_limit = 10
    plots_done = 0
    # Create a subplot with 2 rows and 5 columns
    plt.figure(figsize=(10, 4))
    for i in range(n):
        row = x[i, :]

        # Condition: Only plot if
        if y[i] != y_read[i]:
            plt.subplot(2, 5, plots_done + 1)
            # Reshape the row into a square matrix
            square_matrix = row.reshape(size, size)

            # Use imshow to plot the small image
            plt.imshow(square_matrix, cmap='gray')
            plt.title(f'N:{y[i]} R:{y_read[i]}')
            plt.axis('off')
            plots_done += 1

        if plots_done >= plot_limit:
            break

    plt.tight_layout()
    plt.show()


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)
    print("Ridge Regression Problem")

    print(f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%")
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")
    plotImage(x_test, y_test, y_test_pred)


if __name__ == "__main__":
    main()
