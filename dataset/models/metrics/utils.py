""" Contains utility function for metrics evaluation """
from numba import njit
import numpy as np


@njit(nogil=True)
def binarize(mask, threshold=.5):
    """ Create a binary mask from probabilities with a given threshold.

    Parameters
    ----------
    mask : np.array
        input mask with probabilities
    threshold : float
        where probability is above the threshold, the output mask will have 1, otherwise 0.

    Returns
    -------
    np.array
        binary mask of the same shape as the input mask
    """
    return mask >= threshold


@njit(nogil=True)
def sigmoid(arr):
    return 1. / (1. + np.exp(-arr))