from regression import *


def main():
    X, y, theta, m, n = load_data("data/ex1data2.csv")

    # Learning rate and number of iterations
    alpha, num_iters = 0.01, 1500

    # Running gradient descent
    theta = gradient_descent(X, y, theta, alpha, num_iters, m)


if __name__ == '__main__':
    main()
