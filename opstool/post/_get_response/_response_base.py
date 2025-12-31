from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import xarray as xr


class ResponseBase(ABC):
    """
    Abstract base class for response recorders.
    """

    def __init__(self, *, model_update: bool = False, dtype: dict | None = None):
        self.kargs = {"model_update": model_update, "dtype": dtype}
        self.initialize(model_update=model_update, dtype=dtype)

    def initialize(self, *, dtype: dict | None = None, model_update: bool = False):
        self._resp_step_data = None
        self.resp_step_data_list = []  # for temporary storage, model update handling
        self.resp_step_data_dict = {}  # for temporary storage, non-model-update handling
        self._dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self._dtype.update(dtype)
        self._resp_name = None
        self._resp_types = None
        self._model_update = model_update

        self._step_track = 0
        self._times = [0.0]
        self._have_initial_state_done = False

    def reset(self):
        self.initialize(**self.kargs)

    def check_dataset_empty(self):
        cond = True
        if len(self.resp_step_data_list) > 0:
            for ds in self.resp_step_data_list:
                if ds is not None and ds.data_vars:
                    cond = False
                    return cond
        if self.resp_step_data_dict:
            for data_list in self.resp_step_data_dict.values():
                if len(data_list) > 0:
                    cond = False
                    return cond
        return cond

    def append_time(self, time):
        self._times.append(time)

    def append_step_track(self):
        self._step_track += 1

    def reset_resp_step_data(self):
        self.times = []
        self.resp_step_data = None
        self.resp_step_data_list.clear()
        if self._resp_types is not None:
            for key in self._resp_types:
                self.resp_step_data_dict[key].clear()
        else:
            self.resp_step_data_dict.clear()

    def initialize_resp_step_data_dict(self, keys: Sequence[str]):
        for key in keys:
            self.resp_step_data_dict[key] = []

    def move_one_step(self, time_value: float = 0.0):
        if not self._have_initial_state_done:
            self._times = [0.0]
            self._step_track = 0
            self._have_initial_state_done = True
        else:
            self.append_time(time_value)
            self.append_step_track()

    def add_resp_data_to_datatree(self, dt: xr.DataTree):
        if self.resp_step_data is not None:
            dt[f"/{self.resp_name}"] = self.resp_step_data

    # -----------------------------
    # some properties
    # -----------------------------
    @property
    def resp_step_data(self):
        if self._resp_step_data is None:
            self.add_resp_data_to_dataset()
        return self._resp_step_data

    @resp_step_data.setter
    def resp_step_data(self, data):
        self._resp_step_data = data

    @property
    def current_time(self):
        return self._times[-1]

    @property
    def current_step(self):
        return self._step_track

    @property
    def resp_name(self):
        if self._resp_name is None:
            raise ValueError("resp_name is not set yet.")  # noqa: TRY003
        return self._resp_name

    @resp_name.setter
    def resp_name(self, name: str):
        self._resp_name = name

    @property
    def resp_types(self):
        if self._resp_types is None:
            raise ValueError("resp_types is not set yet.")  # noqa: TRY003
        return self._resp_types

    @resp_types.setter
    def resp_types(self, types: Sequence[str]):
        self._resp_types = types
        self.initialize_resp_step_data_dict(types)

    @property
    def dtype(self):
        return self._dtype

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times: Sequence[float]):
        self._times = list(times)

    @property
    def model_update(self):
        return self._model_update

    @model_update.setter
    def model_update(self, mu: bool):
        self._model_update = mu

    @property
    def have_initial_state_done(self):
        return self._have_initial_state_done

    @have_initial_state_done.setter
    def have_initial_state_done(self, done: bool):
        self._have_initial_state_done = done

    # -----------------------------
    # Abstract API
    # -----------------------------

    @abstractmethod
    def add_resp_data_one_step(self, *args, **kwargs):
        """Add data at each analysis step."""

    @abstractmethod
    def add_resp_data_to_dataset(self, *args, **kwargs):
        """Add data to xarray Dataset."""

    @staticmethod
    @abstractmethod
    def read_response(*args, **kwargs):
        """Read response data."""

    # -----------------------------------------------------------------------------------
    # Utility methods
    @staticmethod
    def _select_node_tags(ds: xr.Dataset, node_tags=None) -> xr.Dataset:
        if node_tags is None or "nodeTags" not in ds.dims:
            return ds
        # slice -> sel (label-based)
        if isinstance(node_tags, slice):
            return ds.sel(nodeTags=node_tags)
        # integer index array -> isel (fast)
        if isinstance(node_tags, (list, tuple, np.ndarray)) and len(node_tags) > 0:
            arr = np.asarray(node_tags)
            if np.issubdtype(arr.dtype, np.integer):
                return ds.isel(nodeTags=arr)
        # otherwise treat as labels
        return ds.sel(nodeTags=node_tags)

    @staticmethod
    def _select_ele_tags(ds: xr.Dataset, ele_tags=None) -> xr.Dataset:
        if ele_tags is None or "eleTags" not in ds.dims:
            return ds
        # slice -> sel (label-based)
        if isinstance(ele_tags, slice):
            return ds.sel(eleTags=ele_tags)
        # integer index array -> isel (fast)
        if isinstance(ele_tags, (list, tuple, np.ndarray)) and len(ele_tags) > 0:
            arr = np.asarray(ele_tags)
            if np.issubdtype(arr.dtype, np.integer):
                return ds.isel(eleTags=arr)
        # otherwise treat as labels
        return ds.sel(eleTags=ele_tags)


def expand_to_uniform_array(array_list, dtype=None):
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
