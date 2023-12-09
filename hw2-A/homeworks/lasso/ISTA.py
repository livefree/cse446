from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    # b(t+1) ← b(t) − 2η sum((X w)+b(t) -y )
    bias_next = bias - 2 * eta * np.sum(np.matmul(X, weight) + bias - y)
    # w(t+1) ← w(t) − 2η (XT (X w(t)+b−yi))
    weight_next = weight - 2 * eta * np.matmul(X.T, np.matmul(X, weight) + bias - y)

    # create masks for conditions [-2 * eta * _lambda, 2 * eta * _lambda]
    lessthan = weight_next < -2 * eta * _lambda
    morethan = weight_next > 2 * eta * _lambda
    otherwise = ~(lessthan | morethan)

    weight_next[lessthan] += 2 * eta * _lambda
    weight_next[morethan] -= 2 * eta * _lambda
    weight_next[otherwise] = 0 # Lasso set small feathers to 0

    return [weight_next, bias_next]


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # calculate loss function
    Xw = np.matmul(X, weight)
    l = (np.linalg.norm(y - Xw - bias, 2)**2 + _lambda * np.linalg.norm(weight, 1))
    return l


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.0001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None
    while (convergence_criterion(start_weight, old_w, start_bias, old_b, convergence_delta) is False):
        old_w = start_weight
        old_b = start_bias
        [start_weight, start_bias] = step(X, y, old_w, old_b, _lambda, eta)
    return [start_weight, start_bias]


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if((old_b is None) or (old_w is None)):
        return False

    max_weight_delta = np.max(np.abs(weight - old_w))
    bias_delta = np.abs(bias - old_b)
    if(max(max_weight_delta, bias_delta) < convergence_delta):
        return True
    else:
        return False


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # n = 500,d = 1000,k = 100, and σ = 1
    n = 500
    d = 1000
    k = 100
    mu, sigma = 0, 1
    noise = np.random.normal(mu, sigma, size=(n, d))
    start_weight = np.array([i/k if i < k else 0 for i in range(d)])
    X = (np.random.rand(n, d) * 5)**2 
    y = np.matmul(X, start_weight) + noise
    
    avg_y = np.mean(y)

    # standarize X
    X_ = X / np.linalg.norm(X, 2)
    _lambda_max = 2 * max(np.matmul(X_.T, (y - avg_y)))
    _lambda = _lambda_max
    nonzeroes = np.array([]) # empty
    lambdas = np.array([])
    weight_record = np.array([])
    count = 0
    eta = 0.00008
    delta = 0.001
    while(count < k):
        [weight, bias] = train(X_, y, _lambda, eta=eta, convergence_delta=delta)
        nonzeroes = np.append(nonzeroes, np.count_nonzero(weight))
        lambdas = np.append(lambdas, _lambda)
        weight_record = np.append(weight_record, weight)
        _lambda /= 2
        count = nonzeroes[-1]
        print('>>', count, ' lambda ', lambdas, ' ? ', nonzeroes)

    weight_record = np.reshape(weight_record, (len(nonzeroes), len(weight)))
    # calculate FDR start_w_j ~= 0 while w_j = 0
    mask = (start_weight != 0) & (weight_record == 0)

    fdr = np.count_nonzero(mask, axis=1) / k
    print(fdr)

    # calculate TPR
    mask = (start_weight != 0) & (weight_record != 0)
    tpr = np.count_nonzero(mask, axis=1) / k
    print(tpr)
    # Now, let's plot the histogram of this data
    plt.figure(1)
    plt.plot(lambdas, nonzeroes)
    # Add titles and labels
    plt.title(f'Plot 1 lambda v.s non zero weights with eta={eta} delta={delta}')
    plt.xlabel('lambda')
    plt.ylabel('num of nonzeros features')
    # plt.xscale('log')
    plt.grid(True)

    plt.figure(2)
    plt.plot(fdr, tpr)
    plt.title(f'Plot 2 FDR vs TDR with eta={eta} delta={delta}')
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
