import numpy as np
import xarray as xr

from ..post import get_nodal_responses
from ._plot_resp_base import PlotResponseBase


class PlotNodalResponseBase(PlotResponseBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)
        self.resps_norm = None

    def set_comp_resp_type(self, resp_type, component):
        if resp_type.lower() in ["disp", "dispacement"]:
            self.resp_type = "disp"
        elif resp_type.lower() in ["vel", "velocity"]:
            self.resp_type = "vel"
        elif resp_type.lower() in ["accel", "acceleration"]:
            self.resp_type = "accel"
        elif resp_type.lower() in ["reaction", "reactionforce"]:
            self.resp_type = "reaction"
        elif resp_type.lower() in ["reactionincinertia", "reactionincinertiaforce"]:
            self.resp_type = "reactionIncInertia"
        elif resp_type.lower() in ["rayleighforces", "rayleigh"]:
            self.resp_type = "rayleighForces"
        elif resp_type.lower() in ["pressure"]:
            self.resp_type = "pressure"
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid response type: {resp_type}. "
                "Valid options are: disp, vel, accel, reaction, reactionIncInertia, rayleighForces, pressure."
            )
        if isinstance(component, str):
            self.component = component.upper()
        else:
            self.component = list(component)

        resp_data = get_nodal_responses(
            self.odb_tag, resp_type=self.resp_type, lazy_load=self.lazy_load, print_info=False
        )
        self.set_resp_step_data(resp_data)

    def _get_resp_clim_peak(self, idx=None):
        # If idx is None, get clim for all steps, else for specified step
        resps = [self._get_resp_da(i, self.component) for i in range(self.num_steps)]

        if self.ModelUpdate:
            resps_norm = [_resp_mag(r) for r in resps]  # list per step
        else:
            resps_da = xr.concat(resps, dim="time", join="exact")  # fast path
            resps_norm = resps_da if resps_da.ndim == 2 else _resp_mag(resps_da)
            resps_norm = resps_norm

        # 3) pick step (compute only tiny score array/scalars)
        step = _pick_step(idx, resps, self.ModelUpdate)

        # 4) clim (compute only two scalars)
        cmin, cmax = _clim_from_norm(resps_norm, step, idx)

        self.resps_norm = resps_norm
        self.clim = (cmin, cmax)
        return cmin, cmax, step

    def _get_step_norm(self, step):
        if self.resps_norm is None:
            raise RuntimeError("resps_norm is not computed yet. Call _get_resp_clim_peak first.")  # noqa: TRY003
        if self.ModelUpdate:
            return np.asanyarray(self.resps_norm[step])
        else:
            return np.asanyarray(self.resps_norm.isel(time=step))

    def _get_mesh_data(self, step, alpha):
        node_defo_coords = np.array(self._get_defo_coord_da(step, alpha))
        if self.resps_norm is not None:
            scalars = self._get_step_norm(step)
        else:
            node_resp = np.array(self._get_resp_da(step, self.component))
            scalars = node_resp if node_resp.ndim == 1 else np.linalg.norm(node_resp, axis=-1)
        return node_defo_coords, scalars

    def _make_title(self, *args, **kwargs):
        pass

    def _create_mesh(self, *args, **kwargs):
        pass

    def _update_mesh(self, *args, **kwargs):
        pass

    def plot_slide(self, *args, **kwargs):
        pass

    def plot_peak_step(self, *args, **kwargs):
        pass

    def plot_anim(self, *args, **kwargs):
        pass


def _resp_mag(da: xr.DataArray) -> xr.DataArray:
    """Magnitude for coloring: keep 1D as-is; otherwise sqrt(sum(x^2)) along the last dim."""
    if da.ndim <= 1:
        return da
    comp_dim = da.dims[-1]
    return (da * da).sum(dim=comp_dim, skipna=True) ** 0.5


def _pick_step(idx, resps: list[xr.DataArray], model_update: bool) -> int:
    """Pick step index from idx spec; compute only tiny arrays/scalars."""
    if isinstance(idx, (int, float)):
        return int(idx)

    mode = idx.lower() if isinstance(idx, str) else "absmax"
    if mode not in ("absmax", "max", "absmin", "min"):
        raise ValueError("Invalid idx, one of [absMax, absMin, Max, Min]")  # noqa: TRY003

    want_max = mode in ("absmax", "max")
    want_abs = mode in ("absmax", "absmin")

    def _to_numpy(a):
        data = a.data if isinstance(a, xr.DataArray) else a
        return data.compute() if hasattr(data, "compute") else np.asarray(data)

    if not model_update:
        # concat once; reduce over all non-time dims -> (time,)
        da = xr.concat(resps, dim="time", join="exact")
        if want_abs:
            da = np.abs(da)

        non_time_dims = [d for d in da.dims if d != "time"]
        scores = da.max(dim=non_time_dims, skipna=True) if want_max else da.min(dim=non_time_dims, skipna=True)

        s = _to_numpy(scores)
        return int(np.nanargmax(s) if want_max else np.nanargmin(s))

    # ragged tags: scalar reduction per step
    vals = []
    for r in resps:
        base = np.abs(r) if want_abs else r
        sc = base.max(skipna=True) if want_max else base.min(skipna=True)
        vals.append(float(_to_numpy(sc).item()))

    return int(np.nanargmax(vals) if want_max else np.nanargmin(vals))


def _clim_from_norm(
    resps_norm: xr.DataArray | list[xr.DataArray],
    step: int,
    idx,
) -> tuple[float, float]:
    """Compute (cmin,cmax). If idx is None -> global across all steps; else only selected step."""
    if isinstance(resps_norm, list):
        if idx is None:
            mins = [r.min(skipna=True) for r in resps_norm]
            maxs = [r.max(skipna=True) for r in resps_norm]
            cmin_da = xr.concat(mins, dim="time").min(skipna=True)
            cmax_da = xr.concat(maxs, dim="time").max(skipna=True)
        else:
            r = resps_norm[step]
            cmin_da, cmax_da = r.min(skipna=True), r.max(skipna=True)
    else:
        if idx is None:
            cmin_da, cmax_da = resps_norm.min(skipna=True), resps_norm.max(skipna=True)
        else:
            r = resps_norm.isel(time=step)
            cmin_da, cmax_da = r.min(skipna=True), r.max(skipna=True)

    cmin = float(cmin_da.data.compute()) if hasattr(cmin_da.data, "compute") else float(cmin_da.values)
    cmax = float(cmax_da.data.compute()) if hasattr(cmax_da.data, "compute") else float(cmax_da.values)
    return cmin, cmax
