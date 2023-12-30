import numpy as np


def decision_function(K: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return w @ K + b


def slack_margin(K: np.ndarray, w: np.ndarray, b: float, y: np.ndarray) -> np.ndarray:
    return y * decision_function(K, w, b)


def compute_kernel_matrix(X: np.ndarray, kernel: str, kwargs: dict) -> np.ndarray:
    if kernel == 'linear':
        K = linear_kernel(X, X)
    elif kernel == 'poly':
        K = poly_kernel(X, X, kwargs['degree'], kwargs['poly_bias'])
    elif kernel == 'rbf':
        K = rbf_kernel(X, X, kwargs['sigma'])
    else:
        raise KeyError(f'Unknown kernel {kernel}, supported kernel types are {["linear", "poly", "rbf"]}')

    return K


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(1 / sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[:, np.newaxis], axis=2) ** 2)


def poly_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, poly_bias: float) -> np.ndarray:
    return poly_bias + X1.dot(X2) ** degree


def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return X1.dot(X2)
