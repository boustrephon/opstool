from typing import Optional

import numpy as np
import xarray as xr

from ..utils import CONFIGS, get_bounds


class PlotResponseBase:
    def __init__(
        self,
        model_info_steps: dict[str, xr.DataArray],
        resp_step: xr.Dataset,
        model_update: bool,
        nodal_resp_steps: Optional[xr.Dataset] = None,
    ):
        self.ModelInfoSteps = model_info_steps
        self.RespSteps = resp_step
        self.ModelUpdate = model_update
        self.nodal_resp_steps = nodal_resp_steps
        self.time = self.RespSteps.coords["time"].values
        self.num_steps = len(self.time)

        self.points_origin = self._get_node_da(0).to_numpy()
        self.points = self.points_origin.copy()
        self.bounds, self.min_bound_size, self.max_bound_size = get_bounds(self.points)
        model_dims = self._get_node_da(0).attrs["ndims"]
        # # show z-axis in 3d view
        self.show_zaxis = not np.max(model_dims) <= 2
        # ------------------------------------------------------------
        self.pargs = None
        self.resp_step = None  # response data
        self.resp_type = None  # response type
        self.component = None  # component to be visualized
        self.fiber_point = None  # fiber point for shell fiber response
        self.unit_symbol = ""  # unit symbol
        self.unit_factor = 1.0
        self.clim = (0, 1)  # color limits

        self.defo_scale_factor = None  # deformation scale factor

        self.PKG_NAME = self.pkg_name = CONFIGS.get_pkg_name()

    def set_unit(self, symbol: Optional[str] = None, factor: Optional[float] = None):
        # unit
        if symbol is not None:
            self.unit_symbol = symbol
        if factor is not None:
            self.unit_factor = factor

    def _get_model_da(self, key, idx):
        da = self.ModelInfoSteps.get(key)
        if da is None:
            return xr.DataArray([], name=key)

        t = idx if self.ModelUpdate else 0
        da = da.isel(time=t)

        # drop nodes/eles that do not exist in this step (2nd dim is tag dim)
        tag_dim = da.dims[1] if len(da.dims) > 1 else None
        if tag_dim:
            da = da.dropna(dim=tag_dim, how="any")

        return da

    def _get_node_da(self, idx):
        nodal_data = self._get_model_da("NodalData", idx)
        if len(nodal_data) > 0:
            unused_node_tags = nodal_data.attrs["unusedNodeTags"]
            if len(unused_node_tags) > 0:
                nodal_data = nodal_data.where(~nodal_data.coords["nodeTags"].isin(unused_node_tags), drop=True)
        return nodal_data.sel(coords=["x", "y", "z"])

    def _get_line_da(self, idx):
        return self._get_model_da("AllLineElesData", idx)

    def _get_unstru_da(self, idx):
        return self._get_model_da("UnstructuralData", idx)

    def _get_bc_da(self, idx):
        return self._get_model_da("FixedNodalData", idx)

    def _get_mp_constraint_da(self, idx):
        return self._get_model_da("MPConstraintData", idx)

    def _get_resp_da(self, time_idx, resp_type, component=None):
        da = self.RespSteps[resp_type].isel(time=time_idx)

        # drop nodes/eles that do not exist in this step
        if self.ModelUpdate:
            tag_dim = next((d for d in da.dims if d != "time"), None)
            if tag_dim is not None:
                da = da.dropna(dim=tag_dim, how="all")

        # no component selection
        if component is None or da.ndim == 1:
            return da * self.unit_factor

        # component dimension: assume last dim
        comp_dim = da.dims[-1]

        # choose sel vs isel deterministically
        da = da.isel({comp_dim: component}) if isinstance(component, (int, slice)) else da.sel({comp_dim: component})

        return da * self.unit_factor

    def _get_disp_da(self, idx):
        if self.nodal_resp_steps is None:
            data = self._get_resp_da(idx, "disp", ["UX", "UY", "UZ"])
            data = data / self.unit_factor  # come back to original unit
        else:
            data = self.nodal_resp_steps["disp"].isel(time=idx)
            if self.ModelUpdate:
                data = data.dropna(dim="nodeTags", how="all")
            data = data.sel(DOFs=["UX", "UY", "UZ"])
        return data

    def _set_defo_scale_factor(self, alpha=1.0):
        if self.defo_scale_factor is not None:
            return

        data = self.RespSteps["disp"] if self.nodal_resp_steps is None else self.nodal_resp_steps["disp"]
        comp_dim = data.dims[-1]
        defos_da = data.sel({comp_dim: ["UX", "UY", "UZ"]})

        # ---- compute alpha_ ----
        if isinstance(alpha, str) or alpha is True:
            comp_dim = defos_da.dims[-1]

            # magnitude = sqrt(sum(x^2)) over comp_dim
            mag = (defos_da * defos_da).sum(dim=comp_dim, skipna=True) ** 0.5

            # scalar max
            maxv = float(mag.max(skipna=True).item())
            alpha_ = 0.0 if maxv == 0.0 else (self.max_bound_size * self.pargs.scale_factor / maxv)

        elif alpha is False or alpha is None:
            alpha_ = 1.0
        else:
            alpha_ = float(alpha)

        self.defo_scale_factor = alpha_

    def _get_defo_coord_da(self, step, alpha):
        if not isinstance(alpha, bool) and alpha == 0.0:
            original_coords_da = self._get_node_da(step)
            return original_coords_da
        self._set_defo_scale_factor(alpha=alpha)
        defo = self._get_disp_da(step)
        pos_origin = self._get_node_da(step)
        coords = self.defo_scale_factor * np.array(defo) + np.array(pos_origin)
        node_deform_coords = xr.DataArray(coords, dims=pos_origin.dims, coords=pos_origin.coords)
        return node_deform_coords

    @staticmethod
    def _get_line_cells(line_data):
        if len(line_data) > 0:
            line_cells = line_data.to_numpy().astype(int)
            line_tags = line_data.coords["eleTags"]
        else:
            line_cells, line_tags = [], []
        return line_cells, line_tags

    @staticmethod
    def _get_unstru_cells(unstru_data):
        if len(unstru_data) > 0:
            unstru_tags = unstru_data.coords["eleTags"]
            unstru_cell_types = np.array(unstru_data[:, -1], dtype=int)
            unstru_cells = unstru_data.to_numpy()
            if not np.any(np.isnan(unstru_cells)):
                unstru_cells_new = unstru_cells[:, :-1].astype(int)
            else:
                unstru_cells_new = []
                for cell in unstru_cells:
                    num = int(cell[0])
                    data = [num] + [int(data) for data in cell[1 : 1 + num]]
                    unstru_cells_new.extend(data)
        else:
            unstru_tags, unstru_cell_types, unstru_cells_new = [], [], []
        return unstru_tags, unstru_cell_types, unstru_cells_new

    def _dropnan_by_time(self, da):
        dims = da.dims
        time_dim = dims[0]
        cleaned_dataarrays = []
        for t in range(da.sizes[time_dim]):
            da_2d = da.isel({time_dim: t})
            if da_2d.size == 0 or any(dim == 0 for dim in da_2d.shape):
                cleaned_dataarrays.append([])
            else:
                dim2 = dims[1]
                da_2d_cleaned = da_2d.dropna(dim=dim2, how="any") if self.ModelUpdate else da_2d
                cleaned_dataarrays.append(da_2d_cleaned)
        return cleaned_dataarrays

    def _plot_outline(self, *args, **kwargs):
        pass

    def _plot_bc(self, *args, **kwargs):
        pass

    def _plot_bc_update(self, *args, **kwargs):
        pass

    def _plot_mp_constraint(self, *args, **kwargs):
        pass

    def _plot_mp_constraint_update(self, *args, **kwargs):
        pass

    def _plot_all_mesh(self, *args, **kwargs):
        pass

    def _update_plotter(self, *args, **kwargs):
        pass
