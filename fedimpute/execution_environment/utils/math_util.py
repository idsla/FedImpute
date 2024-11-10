import numpy as np


def max_squared_sum(X: np.ndarray) -> float:
    """
    Calculate the maximum squared sum of a matrix X
    """
    row_norm = np.sum(np.multiply(X, X), axis=1)
    return np.max(row_norm)