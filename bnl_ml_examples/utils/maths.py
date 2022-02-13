import numpy as np


def min_max_normalize(x, axis=-1):
    """
    Min max normalization
    Parameters
    ----------
    x: array
    axis: int

    Returns
    -------

    """
    return (np.array(x) - np.min(x, axis=axis)) / (
        np.max(x, axis=axis) - np.min(x, axis=axis)
    )
