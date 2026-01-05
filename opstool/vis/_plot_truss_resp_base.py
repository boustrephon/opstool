import numpy as np

from ..post import get_element_responses
from ._plot_resp_base import PlotResponseBase


class PlotTrussResponseBase(PlotResponseBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)

    def _get_truss_data(self, step):
        return self._get_model_da("TrussData", step)

    def _set_resp_type(self, resp_type: str):
        if resp_type.lower() in ["axialforce", "force"]:
            resp_type = "axialForce"
        elif resp_type.lower() in ["axialdefo", "axialdeformation", "deformation"]:
            resp_type = "axialDefo"
        elif resp_type.lower() in ["stress", "axialstress"]:
            resp_type = "Stress"
        elif resp_type.lower() in ["strain", "axialstrain"]:
            resp_type = "Strain"
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported response type {resp_type}!Valid options are: axialForce, axialDefo, Stress, Strain."
            )
        self.resp_type = resp_type

        # set response data
        resp_data = get_element_responses(
            self.odb_tag, ele_type="Truss", resp_type=self.resp_type, lazy_load=self.lazy_load, print_info=False
        )
        self.set_resp_step_data(resp_data)

    def _make_truss_info(self, ele_tags, step):
        pos = self._get_node_da(step).to_numpy()
        truss_data = self._get_truss_data(step)
        if ele_tags is None:
            truss_tags = truss_data.coords["eleTags"].values
            truss_cells = truss_data.to_numpy().astype(int)
        else:
            truss_tags = np.atleast_1d(ele_tags)
            truss_cells = truss_data.sel(eleTags=truss_tags).to_numpy().astype(int)
        truss_node_coords = []
        truss_cells_new = []
        for i, cell in enumerate(truss_cells):
            nodei, nodej = cell[1:]
            truss_node_coords.append(pos[nodei])
            truss_node_coords.append(pos[nodej])
            truss_cells_new.append([2, 2 * i, 2 * i + 1])
        truss_node_coords = np.array(truss_node_coords)
        return truss_tags, truss_node_coords, truss_cells_new

    def refactor_resp_step(self, resp_type: str, ele_tags):
        self._set_resp_type(resp_type)
        resps = []
        if self.ModelUpdate or ele_tags is not None:
            for i in range(self.num_steps):
                truss_tags, _, _ = self._make_truss_info(ele_tags, i)
                da = self._get_resp_da(i)
                da = da.sel(eleTags=truss_tags)
                resps.append(da)
        else:
            for i in range(self.num_steps):
                da = self._get_resp_da(i)
                resps.append(da)
        self.resp_step = resps  # update

    def _get_resp_peak(self, idx=None):
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.nanmax(np.abs(data)) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.max(data) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "absmin":
                resp = [np.nanmin(np.abs(data)) for data in self.resp_step]
                step = np.argmin(resp)
            elif idx.lower() == "min":
                resp = [np.nanmin(data) for data in self.resp_step]
                step = np.argmin(resp)
            else:
                raise ValueError("Invalid argument, one of [absMax, absMin, Max, Min]")  # noqa: TRY003
        elif isinstance(idx, (int, float)):
            step = int(idx)
        else:
            resp = [np.nanmax(np.abs(data)) for data in self.resp_step]  # absMax
            step = np.argmax(resp)
        resp = self.resp_step[step]
        maxv = np.amax(np.abs(resp))
        alpha_ = 0.0 if maxv == 0 else self.max_bound_size * self.pargs.scale_factor / maxv
        cmin, cmax = self._get_truss_resp_clim() if idx is None else (np.nanmin(resp), np.nanmax(resp))
        return step, (cmin, cmax), float(alpha_)

    def _get_truss_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        self.clim = (cmin, cmax)
        return cmin, cmax

    def _get_mesh_data(self, *args, **kwargs):
        pass

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
