import numpy as np

from ..post import get_element_responses
from ._plot_resp_base import PlotResponseBase


class PlotFrameResponseBase(PlotResponseBase):
    def __init__(self, odb_tag, lazy_load=True):
        super().__init__(odb_tag, lazy_load=lazy_load)
        self.plot_axis = None
        self.plot_axis_sign = 1.0
        self.sec_locs = None  # section locations
        self.secLocsData = None  # section locations data, dataset

        self.component_type = None  # component type to display text

    def _set_comp_resp_type(self, resp_type, component):
        if resp_type.lower() in ["localforces", "localforce"]:
            self.resp_type = "localForces"
        elif resp_type.lower() in ["basicforces", "basicforce"]:
            self.resp_type = "basicForces"
        elif resp_type.lower() in ["basicdeformations", "basicdeformation", "basicdefo"]:
            self.resp_type = "basicDeformations"
        elif resp_type.lower() in ["plasticdeformation", "plasticdeformations", "plasticdefo"]:
            self.resp_type = "plasticDeformation"
        elif resp_type.lower() in ["sectionforces", "sectionforce"]:
            self.resp_type = "sectionForces"
        elif resp_type.lower() in ["sectiondeformations", "sectiondeformation", "sectiondefo"]:
            self.resp_type = "sectionDeformations"
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid response type: {resp_type}. "
                "Valid options are: localForces, basicForces, basicDeformations, "
                "plasticDeformations, sectionForces, sectionDeformations."
            )
        # component type
        self.component_type = component.upper()
        if self.resp_type == "localForces":
            self._set_comp_type_local(component)
        elif self.resp_type in ["basicForces", "basicDeformations", "plasticDeformation"]:
            self._set_comp_type_basic(component)
        else:
            self._set_comp_type_section(component)

        # set response data
        resp_data = get_element_responses(
            self.odb_tag, ele_type="Frame", resp_type=self.resp_type, lazy_load=self.lazy_load, print_info=False
        )
        self.set_resp_step_data(resp_data)
        # section locations
        resp_data = get_element_responses(
            self.odb_tag, ele_type="Frame", resp_type="sectionLocs", lazy_load=self.lazy_load, print_info=False
        )
        self.secLocsData = resp_data.sel({resp_data.dims[-1]: "alpha"})

    def _set_comp_type_local(self, comp_type):
        if comp_type.upper() == "FX":
            self.component = ["FX1", "FX2"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        elif comp_type.upper() == "FY":
            self.component = ["FY1", "FY2"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        elif comp_type.upper() == "FZ":
            self.component = ["FZ1", "FZ2"]
            self.plot_axis = "z"
            self.plot_axis_sign = 1
        elif comp_type.upper() == "MX":
            self.component = ["MX1", "MX2"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        elif comp_type.upper() == "MY":
            self.component = ["MY1", "MY2"]
            self.plot_axis = "z"
            self.plot_axis_sign = -1
        elif comp_type.upper() == "MZ":
            self.component = ["MZ1", "MZ2"]
            self.plot_axis = "y"
            self.plot_axis_sign = -1
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for localForces: {comp_type}. Valid options are: FX, FY, FZ, MX, MY, MZ."
            )

    def _set_comp_type_basic(self, comp_type):
        if comp_type.upper() == "N":
            self.component = ["N", "N"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        elif comp_type.upper() == "MZ":
            self.component = ["MZ1", "MZ2"]
            self.plot_axis = "y"
            self.plot_axis_sign = -1
        elif comp_type.upper() == "MY":
            self.component = ["MY1", "MY2"]
            self.plot_axis = "z"
            self.plot_axis_sign = -1
        elif comp_type.upper() == "T":
            self.component = ["T", "T"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for {self.resp_type}: {comp_type}. Valid options are: N, MZ, MY, T."
            )

    def _set_comp_type_section(self, comp_type):
        if comp_type.upper() in ["N", "MZ", "VY", "T"]:
            self.component = comp_type.upper()
            self.plot_axis = "y"
            self.plot_axis_sign = 1
        elif comp_type.upper() in ["VZ", "MY"]:
            self.component = comp_type.upper()
            self.plot_axis = "z"
            self.plot_axis_sign = 1
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for {self.resp_type}: {comp_type}. Valid options are: N, MZ, VY, MY, VZ, T."
            )

    def _get_beam_data(self, step):
        return self._get_model_da("BeamData", step)

    def _make_frame_info(self, ele_tags, step):
        pos = self._get_node_da(step).to_numpy()
        beam_data = self._get_beam_data(step)
        beam_node_coords = []
        beam_cells = []
        if ele_tags is None:
            beam_tags = beam_data.coords["eleTags"].values
            beam_cells_orign = beam_data.loc[:, ["numNodes", "nodeI", "nodeJ"]].to_numpy().astype(int)
            yaxis = beam_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]]
            zaxis = beam_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]]
            for i, cell in enumerate(beam_cells_orign):
                nodei, nodej = cell[1:]
                beam_node_coords.append(pos[int(nodei)])
                beam_node_coords.append(pos[int(nodej)])
                beam_cells.append([2, 2 * i, 2 * i + 1])
        else:
            beam_tags = np.atleast_1d(ele_tags)
            beam_info = beam_data.sel(eleTags=beam_tags)
            yaxis, zaxis = [], []
            for i, etag in enumerate(beam_tags):
                nodei, nodej = beam_info.loc[etag, ["nodeI", "nodeJ"]]
                beam_node_coords.append(pos[int(nodei)])
                beam_node_coords.append(pos[int(nodej)])
                beam_cells.append([2, 2 * i, 2 * i + 1])
                yaxis.append(beam_info.loc[etag, ["yaxis-x", "yaxis-y", "yaxis-z"]])
                zaxis.append(beam_info.loc[etag, ["zaxis-x", "zaxis-y", "zaxis-z"]])
        beam_node_coords = np.array(beam_node_coords)
        yaxis, zaxis = np.array(yaxis), np.array(zaxis)
        return beam_tags, beam_node_coords, beam_cells, yaxis, zaxis

    def _get_sec_loc(self, step):
        sec_loc = self.secLocsData.isel(time=step)
        return sec_loc

    def refactor_resp_data(self, ele_tags, resp_type, component):
        self._set_comp_resp_type(resp_type, component)
        resps, sec_locs = [], []
        if self.ModelUpdate or ele_tags is not None:
            for i in range(self.num_steps):
                beam_tags, _, _, _, _ = self._make_frame_info(ele_tags, i)
                da = self._get_resp_da(i, self.component)
                da = da.sel(eleTags=beam_tags)
                resps.append(da)
                sec_da = self._get_sec_loc(i)
                sec_locs.append(sec_da.sel(eleTags=beam_tags))
        else:
            for i in range(self.num_steps):
                da = self._get_resp_da(i, self.component)
                resps.append(da)
                sec_da = self._get_sec_loc(i)
                sec_locs.append(sec_da)

        self.resp_step = resps
        self.sec_locs = sec_locs

    def _get_resp_scale_factor(self, idx=None):
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.nanmax(np.abs(data)) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.nanmax(data) for data in self.resp_step]
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
            resp = [np.nanmax(np.abs(data)) for data in self.resp_step]  # absmax
            step = np.argmax(resp)
        resp = self.resp_step[step]
        maxv = np.nanmax(np.abs(resp))
        alpha_ = 0.0 if maxv == 0 else self.max_bound_size * self.pargs.scale_factor / maxv
        cmin, cmax = self._get_resp_clim() if idx is None else (np.nanmin(resp), np.nanmax(resp))
        return float(alpha_), step, (cmin, cmax)

    def _get_resp_clim(self):
        maxv = [np.nanmax(data) for data in self.resp_step]
        minv = [np.nanmin(data) for data in self.resp_step]
        cmin, cmax = np.nanmin(minv), np.nanmax(maxv)
        self.clim = (cmin, cmax)
        return cmin, cmax

    def _get_resp_mesh(self, *args, **kwargs):
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
