if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    Xw = np.matmul(X, weight)
    l = np.linalg.norm(y - Xw - bias, 2)**2 
    return l

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    print(df_train.head())

    # Assign the first column to 'y'
    y_train = df_train.iloc[:, 0].values
    y_test = df_test.iloc[:, 0].values

    # Assign the rest of the columns to 'X'
    X_train = df_train.iloc[:, 1:].values
    X_test = df_test.iloc[:, 1:].values

    # Initial weight
    n, d = X_train.shape
    start_weight = np.zeros(d)
    
    # Initial parameters
    avg_y = np.mean(y_train)
    _lambda = 2 * max(np.matmul(X_train.T, (y_train - avg_y)))
    eta = 0.00001
    delta = 0.000001

    nonzeroes = np.array([]) # empty
    lambdas = np.array([])
    weight_record = np.array([])
    train_loss = np.array([])
    test_loss = np.array([])
    count = 0

    [start_weight, start_bias] = train(X_train, y_train, _lambda, eta=eta, convergence_delta=delta, start_weight=start_weight, start_bias=0)
    nonzeroes = np.append(nonzeroes, np.count_nonzero(start_weight))
    lambdas = np.append(lambdas, _lambda)
    weight_record = np.append(weight_record, start_weight)
    _lambda /= 2
    count = nonzeroes[-1]
    # calculate error
    train_loss = np.append(train_loss, loss(X_train, y_train, start_weight, start_bias, _lambda))
    test_loss = np.append(test_loss, loss(X_test, y_test, start_weight, start_bias, _lambda))
    print('>>', count, ' lambda ', lambdas.size, ' ? ', nonzeroes.size)
    while(count < d):
        [weight, bias] = train(X_train, y_train, _lambda, eta=eta, convergence_delta=delta, start_weight=start_weight, start_bias=start_bias)
        nonzeroes = np.append(nonzeroes, np.count_nonzero(weight))
        lambdas = np.append(lambdas, _lambda)
        weight_record = np.append(weight_record, weight)
        _lambda /= 2
        count = nonzeroes[-1]

        # calculate error
        train_loss = np.append(train_loss, loss(X_train, y_train, weight, bias, _lambda))
        test_loss = np.append(test_loss, loss(X_test, y_test, weight, bias, _lambda))
        
        print('>>', count, ' lambda ', lambdas[-1], ' nonzeroes ', nonzeroes)
    

    labels = np.array(['agePct12t29','pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'])
    loc = [df_train.columns.get_loc(label) for label in labels]

    weight_record = np.reshape(weight_record, (len(nonzeroes), len(weight)))
    extracted_weights = weight_record[:, loc]

    # train use lambda = 30
    [weight, bias] = train(X_train, y_train, _lambda=30, eta=eta, convergence_delta=delta, start_weight=start_weight, start_bias=start_bias)
    max_loc = np.argmax(weight)
    min_loc = np.argmin(weight)
    
    print(df_train.columns[max_loc + 1], weight[max_loc], df_train.columns[min_loc + 1], weight[min_loc])

    # retrain with lambda = 0
    [weight, bias] = train(X_train, y_train, _lambda=0, eta=eta, convergence_delta=delta, start_weight=weight, start_bias=bias)
    max_loc = np.argmax(weight)
    min_loc = np.argmin(weight)
    print(df_train.columns[max_loc + 1], weight[max_loc], df_train.columns[min_loc + 1], weight[min_loc])

    plt.figure(1)
    plt.plot(lambdas, nonzeroes)
    # Add titles and labels
    plt.title(f'Plot 1 lambda v.s non zero weights with eta={eta} delta={delta}')
    plt.xlabel('Lambda')
    plt.ylabel('num of nonzeros features')
    plt.xscale('log')
    plt.grid(True)

    plt.figure(2)
    # Loop through each column in the extracted data
    for i in range(extracted_weights.shape[1]):
        plt.plot(lambdas, extracted_weights[:, i], label=f'{labels[i]}')

    # Add title and labels
    plt.title('Line Plots for Extracted Columns')
    plt.xlabel('Lambda')
    plt.xscale('log')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.legend()  # Show the legend

    plt.figure(3)
    plt.title('Training error vs Test error')
    plt.plot(lambdas, train_loss, label = 'Training error')
    plt.plot(lambdas, test_loss, label = 'Test error')
    plt.xscale('log')
    plt.ylabel('Coefficient')
    plt.grid(True)
    plt.legend()  # Show the legend
    plt.show()

if __name__ == "__main__":
    main()
