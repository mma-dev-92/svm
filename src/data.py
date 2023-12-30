from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from src.utils import decision_function


def get_moon(n_samples, noise=0.05):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return noisy_moons[0], noisy_moons[1]


def get_donut(n_samples, noise=0.05, factor=0.5):
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=2)
    return noisy_circles[0], noisy_circles[1]


def plot_input_data(X: np.ndarray, y: np.ndarray) -> None:
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.show()


def plot_data_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, kernel_function: Callable) -> None:
    ax = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    ax = plt.gca()

    x_axis_range = ax.get_xlim()
    y_axis_range = ax.get_ylim()

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_axis_range[0], x_axis_range[1], 50),
        np.linspace(y_axis_range[0], y_axis_range[1])
    )

    grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    z = decision_function(kernel_function(grid), w, b).reshape(x_grid.shape)

    ax.contour(x_grid, y_grid, z, colors=['green', 'blue', 'purple'], levels=[-1, 0, 1], linestyles=['--', '-', '--'], linewidths=2.0)
    plt.show()
