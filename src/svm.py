import numpy as np

from src.utils import compute_kernel_matrix, slack_margin, decision_function


def train(
        X: np.ndarray,
        y: np.ndarray,
        C: float,
        epochs: int = 1000,
        lr: float = 1e-3,
        kernel: str = 'linear',
        **kwargs,
) -> tuple[np.ndarray, float]:
    """
    Train SVM classifier
    :param X: input features
    :param y: classification vector
    :param C: regularization constant hyperparameter
    :param lr: learning rate for gradient descent
    :param epochs: number of epochs
    :param kernel: kernel type, can be 'linear', 'poly' or 'rbf'
    :param kwargs: kernel function parameters
    :return: trained normal vector w and bias b
    """
    K = compute_kernel_matrix(X, kernel, **kwargs)

    b, w = 0, np.random.randn(X.shape[0])

    for epoch in range(epochs):
        margin = slack_margin(K, w, b, y)
        bad_guesses = np.where(margin < 1)[0]

        dw = K @ w - C * y[bad_guesses] @ K[bad_guesses]
        w = w - lr * dw

        db = C * y[bad_guesses].sum()
        b = b - db * lr

        obj = 0.5 * w @ K @ w + C * np.maximum(0, 1 - margin).sum()
        if epoch % 100 == 0:
            print('epoch | ', epoch, ' | obj value ', obj)

    return w, b


def predict(K: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.sign(decision_function(K, w, b))


def score(K: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.mean(predict(K, w, b) == y)
