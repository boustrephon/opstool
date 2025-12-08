from abc import ABC, abstractmethod

import numpy as np


class ResponseBase(ABC):
    @abstractmethod
    def initialize(self):
        """initialize response data."""

    @abstractmethod
    def reset(self):
        """Reset response data."""

    @abstractmethod
    def add_data_one_step(self, *args):
        """Add data at each analysis step."""

    @abstractmethod
    def get_data(self):
        """Get responses data"""

    @abstractmethod
    def update_data(self, *args):
        """Update response data."""

    @abstractmethod
    def get_track(self):
        """Get track tag."""

    @abstractmethod
    def add_to_datatree(self, *args):
        """To xarray DataTree."""

    @abstractmethod
    def read_datatree(self, *args):
        """Read response data from a xarray DataTree."""


def _expand_to_uniform_array(array_list, dtype=None):
    """
    Convert a list of NumPy arrays with varying dimensions and shapes into a single
    uniform NumPy array, padding with NaN where dimensions do not match.

    Parameters:
        array_list (list): List of NumPy arrays with different dimensions and shapes
        dtype: Optional, data type of the returned array

    Returns:
        np.ndarray: A padded NumPy array with uniform shape
    """
    if not array_list:
        return np.array([])

    # Ensure all elements are numpy arrays
    array_list = [np.asarray(arr) for arr in array_list]

    # Find the maximum number of dimensions
    max_ndim = max(arr.ndim for arr in array_list)

    # Find the maximum size for each dimension
    max_shape = []
    for dim in range(max_ndim):
        max_size = 0
        for arr in array_list:
            if dim < arr.ndim:
                max_size = max(max_size, arr.shape[dim])
        max_shape.append(max_size)

    # Create result array, first dimension is the number of arrays
    result = np.full((len(array_list), *max_shape), np.nan)

    # Copy each array into the result
    for i, arr in enumerate(array_list):
        # Create slices for each dimension of the current array
        slices = tuple(slice(0, dim) for dim in arr.shape)

        # If array has fewer dimensions than max, need to pad higher dimensions
        if arr.ndim < max_ndim:
            # Add slices for missing dimensions (take first position)
            full_slices = slices + tuple(slice(0, 1) for _ in range(max_ndim - arr.ndim))
            result[i][full_slices] = arr.reshape(arr.shape + (1,) * (max_ndim - arr.ndim))
        else:
            result[i][slices] = arr

    if dtype is not None:
        result = result.astype(dtype)

    return result
