import numpy as np


class DualKernelSVM:

    def __init__(
            self,
            C: float = 1.0,
            kernel: str = 'linear',
            sigma: float = 0.1,
            degree: int = 2,
            poly_bias: float = 1.0,
    ) -> None:

        if kernel == 'linear':
            self.kernel = self._linear_kernel
        elif kernel == 'poly':
            self.kernel = self._poly_kernel
            self.degree = degree
            self.poly_bias = poly_bias
        elif kernel == 'rbf':
            self.kernel = self._rbf_kernel
            self.sigma = sigma
        else:
            raise KeyError(f'Unknown kernel {kernel}, supported kernel types are {["linear", "poly", "rbf"]}')

        self.C = C
        self.alpha: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-3, epochs: int = 1000) -> None:

        # set random initial value for optimized parameter and bias to 0
        self.alpha, self.b = np.random.random(X.shape[0]), 0
        # precompute sum(y_i * y_j * K_ij) expression for faster computation
        yi_yj_kij = np.outer(y, y) * self.kernel(X, X)
        unit_vector = np.ones(X.shape[0])
        dual_obj_vals = []

        for epoch in range(epochs):
            gradient = unit_vector - yi_yj_kij.dot(self.alpha)
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < 1e-6:
                break
            self.alpha = self.alpha + lr * gradient

            # we have to apply dual problem constraint (0 <= alpha <= C)
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0

            # calculate dual obj
            dual_obj = self.alpha.sum() - 0.5 * np.sum(np.outer(y, y) * yi_yj_kij)
            dual_obj_vals.append(dual_obj)

            # print the objective every 1000 iterations
            if epoch % 1000 == 0:
                print('epoch ', epoch, ' | grad ', gradient_norm, ' | dual_obj ', dual_obj)

        sv_index = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        self.b = np.mean(y[sv_index] - (self.alpha * y).dot(self.kernel(X, X[sv_index])))

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sign(self._decision_function(X, y))

    def _decision_function(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (self.alpha * y).dot(self.kernel(X, X)) + self.b

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[:, np.newaxis], axis=2) ** 2)

    def _poly_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.poly_bias + X1.dot(X2) ** self.degree

    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return X1.dot(X2)
