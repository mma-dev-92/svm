import numpy as np


class SVM:
    """
    Class that implements SVM classifier. No parallelization yet.
    """

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_epochs: int,
            learning_rate: float,
            scaling_constant: float
    ) -> None:
        """
        :param X: training data input examples
        :param y: classification labels of training data input examples
        :param n_epochs: number of epochs
        :param learning_rate: learning rate for gradient descent
        :param scaling_constant: scaling factor for each individual support vector on the objective function
        """

        self._n_epochs: int = n_epochs
        self._learning_rate: float = learning_rate
        self._scaling_constant: float = scaling_constant

        # additional column in X and in w is added to handle the "b" vector more convenient
        self._X: np.ndarray = np.column_stack((np.ones(X.shape[0]), X))
        self._w: np.ndarray = np.ones(self._X.shape[1])

        self._y: np.ndarray = y

    def fit(self) -> None:
        for i in range(self._n_epochs):
            dw, L = self.compute_dw_L(self._X, self._w, self._y)
            self._w = self._w - self._learning_rate * dw
            if i % 10000 == 0:
                print(i, ' | ', L)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classify data using trained vector w
        :param X: data to classify
        :return: estimated y vector
        """
        X = np.column_stack((np.ones(X.shape[0]), X))
        return np.sign(X @ self._w)

    def distances(self, w: np.ndarray) -> np.ndarray:
        """
        Distances of each data point to the hyperplane defined by the normal vector w = (b, v), where b is a scalar and
        v is N-dimensional vector.

        Note: all distances greater than 1, can be set to 0, since the optimal solution is a combination of a
        support vectors (vectors that lie on the hyperplane defined by the vector w). It is a consequence of Lagrange
        theorem.

        :param w: normal vector w = (b, v), where b is a scalar and v is N-dimensional vector.
        :return: vector of distances between each datapoint and the hyperplane defined by the given normal vector w and
        constant b (included in the vector w as first coordinate)
        """
        dist = self._y * np.dot(self._X, w) - 1
        dist[dist > 0] = 0
        return dist

    def compute_dw_L(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = self.distances(w)

        # Get current cost
        L = 1 / 2 * np.dot(w, w) - self._scaling_constant * np.sum(distances)

        dw = np.zeros(len(w))

        for ind, d in enumerate(distances):
            if d == 0:  # if sample is not on the support vector
                di = w  # (alpha * y[ind] * X[ind]) = 0
            else:
                # (alpha * y[ind] * X[ind]) = y[ind] * X[ind]
                di = w - (self._scaling_constant * y[ind] * X[ind])
            dw += di
        return dw / len(X), L

    @property
    def w(self) -> np.ndarray:
        return self._w[1:]

    @property
    def b(self) -> float:
        return self._w[0]
