from typing import Optional

import numpy as np
import xarray as xr

from ..post import get_model_data, get_nodal_responses
from ..utils import CONFIGS, get_bounds, get_random_color


class PlotResponseBase:
    def __init__(self, odb_tag, lazy_load=True):
        self.odb_tag = odb_tag
        self.lazy_load = lazy_load

        self.ModelInfoSteps = None
        self.ModelUpdate = False
        self.RespSteps = None
        self.nodal_disp_steps = None
        self.interp_beam_disp_on = False
        self.interp_beam_disp_steps = None
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

        self.set_model_info_step_data()
        self.check_interp_beam_disp()

        self._print_loading_info()

    def _print_loading_info(self):
        # Print loading info
        RESULTS_DIR = CONFIGS.get_output_dir()
        CONSOLE = CONFIGS.get_console()
        PKG_PREFIX = CONFIGS.get_pkg_prefix()
        RESP_FILE_NAME = CONFIGS.get_resp_filename()
        store_path = f"{RESULTS_DIR}\\" + f"{RESP_FILE_NAME}-{self.odb_tag}.odb"
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} Loading responses data from [bold {color}]{store_path}[/] ...")

    def set_model_info_step_data(self):
        self.ModelInfoSteps = get_model_data(
            self.odb_tag, data_type=None, from_responses=True, lazy_load=self.lazy_load, print_info=False
        )
        self.ModelUpdate = self.ModelInfoSteps.get("ModelUpdate", False)
        self.points_origin = self._get_node_da(0).to_numpy()
        self.points = self.points_origin.copy()
        self.bounds, self.min_bound_size, self.max_bound_size = get_bounds(self.points)
        model_dims = self._get_node_da(0).attrs["ndims"]
        # # show z-axis in 3d view
        self.show_zaxis = not np.max(model_dims) <= 2

    def set_resp_step_data(self, resp_steps):
        self.RespSteps = resp_steps
        self.time = self.RespSteps.coords["time"].values
        self.num_steps = len(self.time)

    def set_nodal_disp_step_data(self):
        if self.nodal_disp_steps is None:
            self.nodal_disp_steps = get_nodal_responses(
                self.odb_tag, resp_type="disp", lazy_load=self.lazy_load, print_info=False
            )

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
        if self.ModelUpdate:
            if "nodeTags" in da.dims:
                da = da.dropna(dim="nodeTags", how="all")
            if "eleTags" in da.dims:
                da = da.dropna(dim="eleTags", how="all")

        return da

    def _get_node_da(self, idx):
        nodal_data = self._get_model_da("NodalData", idx)
        return nodal_data.sel(coords=["x", "y", "z"])

    def _get_line_da(self, idx, enforce=False):
        if enforce or not self.interp_beam_disp_on:
            return self._get_model_da("AllLineElesData", idx)
        else:
            return xr.DataArray([], name="AllLineElesData")

    def _get_unstru_da(self, idx):
        return self._get_model_da("UnstructuralData", idx)

    def _get_bc_da(self, idx):
        return self._get_model_da("FixedNodalData", idx)

    def _get_mp_constraint_da(self, idx):
        return self._get_model_da("MPConstraintData", idx)

    def _get_resp_da(self, time_idx, component=None):
        da = self.RespSteps.isel(time=time_idx)

        # drop nodes/eles that do not exist in this step
        if self.ModelUpdate:
            if "nodeTags" in da.dims:
                da = da.dropna(dim="nodeTags", how="all")
            if "eleTags" in da.dims:
                da = da.dropna(dim="eleTags", how="all")

        # no component selection
        if component is None or da.ndim == 1:
            return da * self.unit_factor

        # component dimension: assume last dim
        comp_dim = da.dims[-1]

        # choose sel vs isel deterministically
        da = da.isel({comp_dim: component}) if isinstance(component, (int, slice)) else da.sel({comp_dim: component})

        return da * self.unit_factor

    def _get_disp_da(self, idx):
        self.set_nodal_disp_step_data()
        data = self.nodal_disp_steps.isel(time=idx).sel(DOFs=["UX", "UY", "UZ"])
        # if self.nodal_resp_steps is None:
        #     data = self._get_resp_da(idx, "disp", ["UX", "UY", "UZ"])
        #     data = data / self.unit_factor  # come back to original unit
        # else:
        #     data = self.nodal_disp_steps["disp"].isel(time=idx).sel(DOFs=["UX", "UY", "UZ"])
        if self.ModelUpdate:
            data = data.dropna(dim="nodeTags", how="all")
        return data

    def _set_defo_scale_factor(self, alpha=1.0):
        if self.defo_scale_factor is not None:
            return

        self.set_nodal_disp_step_data()

        defos_da = self.nodal_disp_steps.sel(DOFs=["UX", "UY", "UZ"])

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
        coords = self.defo_scale_factor * np.asanyarray(defo) + np.asanyarray(pos_origin)
        node_deform_coords = xr.DataArray(coords, dims=pos_origin.dims, coords=pos_origin.coords)
        return node_deform_coords

    def check_interp_beam_disp(self):
        if self.interp_beam_disp_on:
            interp_beam_disp_steps = get_nodal_responses(
                self.odb_tag, resp_type="interpolate_disp", lazy_load=self.lazy_load, print_info=False
            )
            if len(interp_beam_disp_steps) == 0:
                self.interp_beam_disp_on = False

    def set_interp_beam_on(self, on: bool):
        self.interp_beam_disp_on = on

    def set_interp_beam_disp_step_data(self):
        if self.interp_beam_disp_on and self.interp_beam_disp_steps is None:
            self.interp_beam_disp_steps = get_nodal_responses(
                self.odb_tag, resp_type="interpolate_disp", lazy_load=self.lazy_load, print_info=False
            )
            self.interp_beam_points = get_nodal_responses(
                self.odb_tag, resp_type="interpolate_points", lazy_load=self.lazy_load, print_info=False
            )
            self.interp_beam_cells = get_nodal_responses(
                self.odb_tag, resp_type="interpolate_cells", lazy_load=self.lazy_load, print_info=False
            )

    def get_interp_beam_data(self, idx, alpha):
        self.set_interp_beam_disp_step_data()
        points_data = self.interp_beam_points.isel(time=idx)
        cells_data = self.interp_beam_cells.isel(time=idx)
        disp_data = self.interp_beam_disp_steps.isel(time=idx)
        if self.ModelUpdate:
            points_data = points_data.dropna(dim="interpolate_pointID", how="all")
            disp_data = disp_data.dropna(dim="interpolate_pointID", how="all")
            cells_data = cells_data.dropna(dim="interpolate_lineID", how="all")
        points_origin = np.array(points_data)
        cells_data = np.array(cells_data)
        disp_data = np.array(disp_data)
        scalars = np.sqrt(np.sum(disp_data**2, axis=1))
        self._set_defo_scale_factor(alpha=alpha)
        points_defo = points_origin + self.defo_scale_factor * disp_data
        return points_origin, points_defo, cells_data, scalars

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
            unstru_cells = unstru_data.to_numpy()
            unstru_cell_types = np.array(unstru_cells[:, -1], dtype=int)
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
