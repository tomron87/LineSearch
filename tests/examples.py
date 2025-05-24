import numpy as np

def func1(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    Q = np.array([[2, 0], [0, 2]])
    f_x = 0.5 * x @ Q @ x.T

    return [f_x, Q @ x, Q]

def func2(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    Q = np.array([[2, 0], [0, 200]])
    f_x = 0.5 * x @ Q @ x.T

    return [f_x, Q @ x, Q]

def func3(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    a = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    b = np.array([[200, 0], [0, 2]])
    Q = a.T @ b @ a
    f_x = 0.5 * x @ Q @ x.T

    return [f_x, Q @ x, Q]

def rosenbrock(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    f_x = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    grad_x = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
    hessian_x = np.array([[2 - 400 * x[1] + 1200 * x[0]**2, -400 * x[0]], [-400 * x[0], 200]], dtype=float)

    return [f_x, grad_x, hessian_x]

def linear(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    a = np.array([3, 5])
    f_x = a @ x.T

    return [f_x, a.T, None]

def func4(x):
    if (len(x) != 2):
        raise ValueError("x must be a 2D array")
    f_x = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    grad_x = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1), 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    hessian_x = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1), 3 * np.exp(x[0] + 3 * x[1] - 0.1) + 3 * np.exp(x[0] - 3 * x[1] - 0.1)], [3 * np.exp(x[0] + 3 * x[1] - 0.1) + 3 * np.exp(x[0] - 3 * x[1] - 0.1), 9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])

    return [f_x, grad_x, hessian_x]