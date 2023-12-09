from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * (x ** 2))


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # kpoly(x, z) = (1 + xi.T xj)^d where d ∈ N is a hyperparameter,
    return (1 + np.multiply.outer(x_i, x_j)) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma * (np.subtract.outer(x_i, x_j) ** 2))


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    # Compute the kernel matrix
    K = kernel_function(x, x, kernel_param)

    # Compute the alpha vector
    n = len(x)
    alpha = np.linalg.solve(K + _lambda * np.eye(n), y)
    return alpha


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    total_error = 0
    for i in range(num_folds):
        # Split data into training and validation sets
        start, end = i * fold_size, (i + 1) * fold_size
        x_val, y_val = x[start:end], y[start:end]
        x_train = np.concatenate([x[:start], x[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        # Train the model and predict on the validation set
        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        # print("alpha: ", alpha.shape, kernel_function(x_train, x_val, kernel_param).shape)
        y_pred = np.dot(kernel_function(x_train, x_val, kernel_param).T, alpha)

        # Compute mean squared error
        total_error += np.mean((y_val - y_pred) ** 2)

    # Return the average error across folds
    return total_error / num_folds


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """

    best_score = float('inf')
    best_gamma = gamma_search(x)
    best_lambda = None

    # Define ranges for lambda
    lambdas = 10**(np.linspace(-5, -1, num=20))  # range for lambda
   
    for _lambda in lambdas:
        score = cross_validation(x, y, rbf_kernel, best_gamma, _lambda, num_folds)
        if score < best_score:
            best_score = score
            best_lambda = _lambda

    return best_gamma, best_lambda


def gamma_search(x: np.ndarray) -> float:

    # Compute the squared distances between each pair of points
    squared_distances = np.subtract.outer(x, x) ** 2

    # Flatten the matrix to a vector
    upper_triangle_indices = np.triu_indices_from(squared_distances, k=1)

    # take the median of the squared distances
    median_squared_distance = np.median(squared_distances[upper_triangle_indices])

    # Calculate gamma as the reciprocal of the median
    return 1 / median_squared_distance


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem. 
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    best_lambda = None
    best_d = None
    best_score = float('inf')

    # Define ranges for lambda and d
    lambdas = 10**(np.linspace(-5, -1, num=20))  # Adjust here
    degrees = np.arange(5, 26)  # Polynomial degrees from 5 to 25

    for _lambda in lambdas:
        for d in degrees:
            # Perform cross-validation
            score = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)

            # Update the best parameters if the current score is better
            if score < best_score:
                best_score = score
                best_lambda = _lambda
                best_d = d

    return best_lambda, best_d
    

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    # RBF kernel
    optimal_gamma, optimal_lambda_rbf = rbf_param_search(x_30, y_30, num_folds=len(x_30))
    alpha_rbf = train(x_30, y_30, rbf_kernel, optimal_gamma, optimal_lambda_rbf)
    print("RBF kernel: gamma=",optimal_gamma, " lambda=", optimal_lambda_rbf)

    # Polynomial kernel
    optimal_lambda_poly, optimal_d = poly_param_search(x_30, y_30, num_folds=len(x_30))
    alpha_poly = train(x_30, y_30, poly_kernel, optimal_d, optimal_lambda_poly)
    print("Poly kernel: d=", optimal_d, " lambda=", optimal_lambda_poly)

    # Generate a fine grid for predictions
    fine_grid = np.linspace(0, 1, num=100)
    # Predict using RBF kernel
    y_pred_rbf = (np.dot(rbf_kernel(fine_grid, x_30, optimal_gamma), alpha_rbf))
    
    # Predict using Polynomial kernel
    y_pred_poly = (np.dot(poly_kernel(fine_grid, x_30, optimal_d), alpha_poly))

    f = f_true(fine_grid)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot for RBF kernel
    plt.subplot(1, 2, 1)
    plt.scatter(x_30, y_30, label="Data")
    plt.plot(fine_grid, y_pred_rbf, label="RBF Predictions", color="red")
    plt.plot(fine_grid, f, label="True function")
    plt.title("RBF Kernel Predictions")
    plt.ylim(-6, 6)
    plt.legend()

    # Plot for Polynomial kernel
    plt.subplot(1, 2, 2)
    plt.scatter(x_30, y_30, label="Data")
    plt.plot(fine_grid, y_pred_poly, label="Polynomial Predictions", color="green")
    plt.title("Polynomial Kernel Predictions")
    plt.plot(fine_grid, f, label="True function")
    plt.ylim(-6, 6)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
