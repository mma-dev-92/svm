import numpy as np


class PrimalKernelSVM:
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            C: float = 1.0,
            kernel: str = 'linear',
            sigma: float = 0.1,
            degree: int = 2,
            poly_bias: float = 1.0,
    ) -> None:

        if kernel == 'linear':
            self.K = self._linear_kernel(X, X)
        elif kernel == 'poly':
            self.K = self._poly_kernel(X, X, degree, poly_bias)
        elif kernel == 'rbf':
            self.kernel = self._rbf_kernel(X, X, sigma)
        else:
            raise KeyError(f'Unknown kernel {kernel}, supported kernel types are {["linear", "poly", "rbf"]}')

        self.X, self.y = X, y
        self.b = 0
        self.w = np.random.randn(X.shape[0])
        self.C = C

    def __decision_function(self) -> np.ndarray:
        """ if 1 - support vector, if > 1 - good guess, if < 1 - bad guess """
        return self.w @ self.K + self.b

    def __slack_margin(self) -> np.ndarray:
        """ Slack variable to track decision boundary violation. """
        return self.y * self.__decision_function()

    def predict(self) -> np.ndarray:
        return np.sign(self.__decision_function())

    def score(self) -> np.ndarray:
        return np.mean(self.predict() == self.y)

    def fit(self, lr: float = 1e-3, epochs=1000) -> None:

        obj_values = []
        for epoch in range(epochs):
            margin = self.__slack_margin()
            bad_guesses = np.where(margin < 1)[0]
            dw = self.K @ self.w - self.C * self.y[bad_guesses] @ self.K[bad_guesses]
            db = self.C * self.y[bad_guesses].sum()

            self.w, self.b = self.w - lr * dw, self.b - lr * db
            obj = 0.5 * self.w @ self.K @ self.w + self.C * np.maximum(0, 1 - margin).sum()
            obj_values.append(obj)

            if epoch % 100 == 0:
                print('epoch | ', epoch, ' | obj value ', obj)

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-(1 / sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[:, np.newaxis], axis=2) ** 2)

    @staticmethod
    def _poly_kernel(X1: np.ndarray, X2: np.ndarray, degree: int, poly_bias: float) -> np.ndarray:
        return poly_bias + X1.dot(X2) ** degree

    @staticmethod
    def _linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return X1.dot(X2)
