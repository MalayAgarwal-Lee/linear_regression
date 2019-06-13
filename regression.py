import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path, header=None)
    m = data.shape[0]

    # Getting values of features
    X = np.array(data.iloc[:, 0:-1])

    # Appending (m, 1) matrix of ones for extra feature x_0
    # Each row of matrix corresponds to a training example
    # Each column corresponds to a feature
    X = np.append(np.ones((m, 1)), X, axis=1)

    # Getting y values as (m, 1) matrix
    y = np.reshape(np.array(data.iloc[:, -1]), (m, 1))

    # Initializing theta to a (n, 1) zero matrix
    # n is the number of features
    n = X.shape[1]
    theta = np.zeros((n, 1))

    return X, y, theta, m, n


def cost_function(X, y, theta, m):

    # Predicting y with current theta values
    predictions = X @ theta

    # Getting (h(x) - y) ^ 2
    sqr_err = np.square(predictions - y)

    # Computing cost as (1/2 * m) * sum((h(x) - y) ^ 2)
    return 1 / (2 * m) * sum(sqr_err)


def plot_cost(costs):
    plt.plot(costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("J(theta)")
    plt.title("Iterations vs Cost")
    plt.show()


def gradient_descent(X, y, theta, alpha, num_iters, m):

    # Array to store cost values at each iteration
    j_vals = np.zeros((num_iters, 1))

    for i in range(num_iters):
        # Vectorized gradient descent
        # delta = ((X * theta - y)' * X)'
        # => theta = theta - (alpha / m) * delta
        diff_transpose = np.transpose((X @ theta - y))
        delta = np.transpose(diff_transpose @ X)
        theta = theta - (alpha / m) * delta
        j_vals[i][0] = cost_function(X, y, theta, m)

    plot_cost(j_vals)
    return theta
