import numpy as np


def decision_function(K: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return w.dot(K) + b


def slack_margin(y: np.ndarray, K: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return y * decision_function(K, w, b)


def predict(K: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.sign(decision_function(K, w, b))


def accuracy(y: np.ndarray, K: np.ndarray, w: np.ndarray, b: float) -> float:
    return np.mean(y == predict(K, w, b))
