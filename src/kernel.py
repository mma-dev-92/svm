import numpy as np


def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return X1.dot(X2.T)


def poly_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, translation: float) -> np.ndarray:
    return (X1.dot(X2.T) + translation) ** degree


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    return mu * np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)
