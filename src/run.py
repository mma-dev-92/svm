from src.data import get_moon, plot_data_boundary, get_donut
from src.kernel import rbf_kernel, poly_kernel, linear_kernel
from src.svm import fit
from src.utils import accuracy


def moon_data_example() -> None:
    X, y = get_moon(n_samples=10000, noise=0.1)
    y[y == 0] = -1

    K = rbf_kernel(X, X, gamma=2.04, mu=1)
    obj, w, b = fit(K, y, lr=1e-4, epochs=500)
    print('Accuracy: ', accuracy(y, K, w, b))

    plot_data_boundary(X, y, w, b, lambda m: rbf_kernel(X, m, gamma=2.04, mu=1))


def donut_data_example() -> None:
    X, y = get_donut(n_samples=200, noise=0.05)
    y[y == 0] = -1

    K = poly_kernel(X, X, degree=5, translation=1.0)
    obj, w, b = fit(K, y, lr=1e-5, epochs=1000, C=0.8)
    print('Accuracy: ', accuracy(y, K, w, b))
    plot_data_boundary(X, y, w, b, lambda m: poly_kernel(X, m, degree=5, translation=1.0))


if __name__ == '__main__':
    moon_data_example()
    donut_data_example()
