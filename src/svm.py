import numpy as np

from src.utils import slack_margin


def fit(
        K: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 100,
        C: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    SVM classifier training

    :param K: kernel matrix
    :param y: training set classification (given apriori)
    :param lr: learning rate for the gradient descent algorithm
    :param epochs: number of epochs
    :param C: regularization constant hyperparameter
    :return: support_vectors, w, bias, obj_values
    """
    obj_values = []
    datasize = y.shape[0]
    w, b = np.random.randn(datasize), 0

    for epoch in range(epochs):

        margin = slack_margin(y, K, w, b)
        bad_guess = np.where(margin < 1)[0]

        dw = K.dot(w) - C * y[bad_guess].dot(K[bad_guess])
        w = w - lr * dw

        db = -C * y[bad_guess].sum()
        b = b - lr * db

        primal_obj = w.dot(K.dot(w)) + C * np.sum(np.maximum(0, 1 - margin))
        obj_values.append(primal_obj)

        if epoch % 100 == 0:
            print('epoch | ', epoch, ' | obj ', primal_obj)

    return np.array(obj_values), w, b
