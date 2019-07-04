import numpy as np
import matplotlib.pyplot as plt


def cost_function(X, y, theta, m):
    '''
    Calculates the value of the cost function
    For a given X, y, theta and m

    The formula for the cost function is as follows:

        cost, J = (1/2 * m) * sum((X * theta - y) ^ 2)

    Arguments:
        X: ndarray, (m, n) matrix consisting of the features
        y: ndarray, (m, 1) matrix with y-values
        theta: ndarray, (n, 1) matrix with parameter values
        m: int, number of training examples

    Returns:
        J: ndarray, (1,1) matrix with value of cost function
    '''
    predictions = X @ theta
    sqr_err = np.square(predictions - y)
    return 1 / (2 * m) * sum(sqr_err)


def plot_cost(costs):
    '''
    Plots the values of the cost function
    Against number of iterations
    If gradient descent has converged, graph flattens out
    And becomes constant near the final iterations
    Otherwise, it shows a different trend
    '''
    plt.plot(costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("J(theta)")
    plt.title("Iterations vs Cost")
    plt.show()


def gradient_descent(X, y, theta, alpha, num_iters, m):
    '''
    Runs gradient descent num_iters times
    To get the optimum values of the parameters
    The algorithm can be looked at here:
    https://en.wikipedia.org/wiki/Gradient_descent

    It can be vectorized as follows:
        theta = theta - (alpha / m) * ((X * theta - y)' * X)'

    Arguments:
        X: ndarray, (m, n) matrix consisting of the features
        y: ndarray, (m, 1) matrix with y-values
        theta: ndarray, (n, 1) matrix with initial parameter values
        alpha: float, the learning rate
        num_iters: int, the number of times algorithm is to be run
        m: int, the number of training examples

    Returns:
        theta: ndarray, (n, 1) matrix with optimum param values
    '''

    # Array to store cost values at each iteration
    # Will be used to check convergence of the algorithm
    j_vals = np.zeros((num_iters, 1))

    for i in range(num_iters):
        # (X * theta - y)'
        difference = np.transpose((X @ theta - y))
        # ((X * theta - y)' * X)'
        delta = np.transpose(difference @ X)
        theta = theta - (alpha / m) * delta
        j_vals[i][0] = cost_function(X, y, theta, m)

    # Plotting the cost values
    plot_cost(j_vals)
    return theta
