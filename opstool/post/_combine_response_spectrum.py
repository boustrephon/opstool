import numpy as np
import xarray as xr


def _cqc_correlation_matrix(lambdas, damping, scale=None):
    """
    Compute the CQC (Complete Quadratic Combination) correlation matrix.

    Parameters
    ----------
    lambdas : array-like
        Modal frequencies (or circular frequencies). Length = n_modes.
    damping : float or array-like
        Modal damping ratios. Can be a scalar (same damping for all modes)
        or an array of length n_modes.
    scale : array-like or None
        Modal scaling factors (e.g., modal participation factors).
        If None, all scaling factors are set to 1.

    Returns
    -------
    rho : ndarray (n_modes, n_modes)
        The CQC correlation matrix, including scale_i * scale_j.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    n = lambdas.size

    # Damping
    damping = np.full(n, float(damping)) if np.isscalar(damping) else np.asarray(damping, dtype=float)

    # Scaling factors
    scale = np.ones(n, dtype=float) if scale is None else np.asarray(scale, dtype=float)

    # Broadcast shapes for pairwise computation
    li = lambdas.reshape(-1, 1)
    lj = lambdas.reshape(1, -1)
    di = damping.reshape(-1, 1)
    dj = damping.reshape(1, -1)

    # Frequency ratio
    bij = li / lj

    # CQC correlation factor rho_ij
    num = 8.0 * np.sqrt(di * dj) * (di + bij * dj) * (bij**1.5)
    den = (1.0 - bij**2.0) ** 2.0 + 4.0 * di * dj * bij * (1.0 + bij**2.0) + 4.0 * (di**2.0 + dj**2.0) * bij**2.0

    rho = num / den

    # Apply scaling factors scale[i] * scale[j]
    si = scale.reshape(-1, 1)
    sj = scale.reshape(1, -1)
    rho = rho * (si * sj)

    return rho


def _combine_da(
    da,
    method="srss",
    lambdas=None,
    damping=0.05,
    scale=None,
    time_dim="time",
):
    """
    Combine modal responses in a DataArray along the given dimension.

    Note
    ----
    This function assumes *all* entries along `time_dim` are modal
    responses. It does not know about the "index 0 = non-modal" convention.
    That convention is handled outside, in `combine_response_spectrum`.

    Parameters
    ----------
    da : xarray.DataArray
        Modal response data. The `time_dim` corresponds to modal index.
    method : {"srss", "cqc"}
        Combination method:
        - "srss": sqrt(sum(mu_i^2))
        - "cqc" : Complete Quadratic Combination
    lambdas : array-like, optional
        Modal frequencies. Required when method="cqc".
        Length must equal da.sizes[time_dim].
    damping : float or array-like
        Modal damping ratios.
    scale : array-like or None
        Modal scaling factors (optional).
    time_dim : str
        Name of the dimension representing modes.

    Returns
    -------
    xarray.DataArray
        Combined response. The `time_dim` is removed.
    """
    # -----------------------------------------------------------
    # SRSS combination: sqrt(sum(mu_i^2))
    # -----------------------------------------------------------
    if method.lower() == "srss":
        return np.sqrt((da**2).sum(dim=time_dim))

    # -----------------------------------------------------------
    # CQC combination
    # -----------------------------------------------------------
    if method.lower() != "cqc":
        raise ValueError(f"Unsupported method: {method}")  # noqa: TRY003

    if lambdas is None:
        raise ValueError("CQC requires modal frequencies `lambdas`.")  # noqa: TRY003

    lambdas = np.asarray(lambdas, dtype=float)
    n_modes = da.sizes[time_dim]

    if lambdas.size != n_modes:
        raise ValueError(f"Mismatch: len(lambdas)={lambdas.size}, time dimension={n_modes}")  # noqa: TRY003

    # Compute correlation matrix rho_ij
    rho = _cqc_correlation_matrix(lambdas, damping, scale)

    # Move mode dimension to front for easier math
    da_mode_first = da.transpose(time_dim, ...)
    other_dims = da_mode_first.dims[1:]
    other_shape = tuple(da_mode_first.sizes[d] for d in other_dims)

    # Flatten remaining dimensions → shape = (n_modes, n_points)
    mu = da_mode_first.data.reshape(n_modes, -1)

    # CQC formula: u^2 = mu^T * rho * mu
    u2 = np.einsum("ix,ij,jx->x", mu, rho, mu, optimize=True)
    u = np.sqrt(np.maximum(u2, 0.0))  # avoid negative due to numerical error

    # Reshape back to original non-mode dimensions
    u = u.reshape(other_shape)

    return xr.DataArray(
        u,
        dims=other_dims,
        coords={dim: da_mode_first.coords[dim] for dim in other_dims},
        attrs=da.attrs,
    )


def _prepare_modal_params(arr, n_total, n_modes):
    """
    Prepare modal parameters (lambdas, damping, scale) for modes 1..n_total-1.

    - If arr is None -> return None.
    - If arr is scalar -> return scalar (same for all modes).
    - If len(arr) == n_modes -> use as-is (already modal-only).
    - If len(arr) == n_total -> drop index 0 and use arr[1:].
    - Otherwise -> raise error.
    """
    if arr is None:
        return None

    arr = np.asarray(arr)
    if arr.ndim == 0:
        # Scalar parameter, same for all modes
        return arr

    if arr.size == n_modes:
        return arr
    if arr.size == n_total:
        return arr[1:]

    raise ValueError(  # noqa: TRY003
        f"Parameter length mismatch: got {arr.size}, expected {n_modes} (modal-only) or {n_total} (including index 0)."
    )


def combine_response_spectrum(
    data,
    method="srss",
    lambdas=None,
    damping=0.05,
    scale=None,
    time_dim="time",
    var_names=None,
    exclude_vars=None,
):
    """
    Combine modal responses stored in an xarray Dataset/DataArray.

    Convention
    ----------
    - Index 0 along `time_dim` is NOT modal data (e.g. original step).
    - Modal responses start from index 1..N-1.
    - len(lambdas) = N-1, i.e. number of modal entries.

    Parameters
    ----------
    data : xarray.Dataset
        Responses including one non-modal entry at index 0 and
        modal responses from index 1..N-1.
    method : {"srss", "cqc"}
        Combination method.
    lambdas : array-like, optional
        Modal frequencies. Required for CQC.
        Length should be N-1 (modal entries) or N (including index 0).
    damping : float or array-like
        Modal damping ratios. Same length rules as `lambdas`.
    scale : array-like or None
        Modal scaling factors. Same length rules as `lambdas`.
    time_dim : str
        The dimension corresponding to time / modal index.
    var_names : list of str, optional
        For Dataset input: which variables to process.
        If None, all data variables that depend on `time_dim` are processed.
    exclude_vars : list of str, optional
        For Dataset input: which variables to exclude from processing.
        Applied after `var_names`.

    Returns
    -------
    xarray.Dataset
        Same type as input, with:
        - `time_dim` coords reset to 0..N-1
        - index 0 updated as the combined response of indices 1..N-1
    """
    if not isinstance(data, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")  # noqa: TRY003

    ds = data.copy()

    if time_dim not in ds.dims:
        raise ValueError(f"{time_dim!r} is not a dimension of the Dataset.")  # noqa: TRY003

    n_total = ds.dims[time_dim]
    n_modes = n_total - 1

    # 1) Reset coordinate of time_dim to 0..N-1
    ds = ds.assign_coords({time_dim: np.arange(n_total, dtype=int)})

    # If there is no modal data, just return (only index 0 exists)
    if n_modes <= 0:
        return ds

    # Prepare list of variables to process
    if var_names is None:
        var_names = list(ds.data_vars)
    if exclude_vars is not None:
        var_names = [name for name in var_names if name not in exclude_vars]

    # Prepare modal parameters (shared across variables)
    lambdas_modal = _prepare_modal_params(lambdas, n_total, n_modes)
    damping_modal = _prepare_modal_params(damping, n_total, n_modes)
    scale_modal = _prepare_modal_params(scale, n_total, n_modes)

    for name in var_names:
        da = ds[name]

        # Only process variables that depend on time_dim
        if time_dim not in da.dims:
            continue

        # 2) Extract modal part for this variable
        da_modes = da.isel({time_dim: slice(1, None)})

        # 3) Combine modal responses
        combined = _combine_da(
            da_modes,
            method=method,
            lambdas=lambdas_modal,
            damping=damping_modal,
            scale=scale_modal,
            time_dim=time_dim,
        )

        # 4) Assign combined response to index 0 (overwrite)
        ds[name].loc[{time_dim: 0}] = combined

    return ds
