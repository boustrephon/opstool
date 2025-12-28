from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ..model_data import GetFEMData
from ._response_base import ResponseBase


class ModelInfoStepData(ResponseBase):
    def __init__(self, **kargs):
        self.kargs = kargs
        super().__init__(**kargs)

        self.current_node_tags = None
        self.current_truss_tags = None
        self.current_frame_tags = None
        self.current_link_tags = None
        self.current_shell_tags = None
        self.current_plane_tags = None
        self.current_brick_tags = None
        self.current_contact_tags = None
        self.current_frame_load_data = None

        self.model_info_steps = {}

        # initial step
        model_info = self._get_model_info()
        for key, value in model_info.items():
            self.model_info_steps[key] = [value]

        self._set_current_tags(model_info)
        self.move_one_step(time_value=ops.getTime())

    def _get_model_info(self):
        model_info, _ = GetFEMData().get_model_info()
        nodal_data = model_info.get("NodalData")
        if nodal_data is not None and len(nodal_data) > 0:
            unused_node_tags = nodal_data.attrs["unusedNodeTags"]
            if len(unused_node_tags) > 0:
                nodal_data = nodal_data.where(~nodal_data.coords["nodeTags"].isin(unused_node_tags), drop=True)
                model_info["NodalData"] = nodal_data
        return model_info

    def add_resp_data_one_step(self):
        if self.model_update:
            model_info = self._get_model_info()
            for key, value in model_info.items():
                if key in self.model_info_steps:
                    self.model_info_steps[key].append(value)

            self._set_current_tags(model_info)
            self.move_one_step(time_value=ops.getTime())

    def _set_current_tags(self, model_info: dict):
        self._set_currnet_node_tags(model_info)
        self._set_current_truss_tags(model_info)
        self._set_current_frame_tags(model_info)
        self._set_current_link_tags(model_info)
        self._set_current_shell_tags(model_info)
        self._set_current_plane_tags(model_info)
        self._set_current_brick_tags(model_info)
        self._set_current_contact_tags(model_info)
        self._set_current_frame_load_data(model_info)

    def _set_currnet_node_tags(self, model_info: dict):
        da = model_info.get("NodalData")
        if da is not None and len(da) > 0:
            node_tags = list(da.coords["nodeTags"].data)
            self.current_node_tags = [int(tag) for tag in node_tags]

    def _set_current_truss_tags(self, model_info: dict):
        da = model_info.get("TrussData")
        if da is not None and len(da) > 0:
            self.current_truss_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_frame_tags(self, model_info: dict):
        da = model_info.get("BeamData")
        if da is not None and len(da) > 0:
            self.current_frame_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_link_tags(self, model_info: dict):
        da = model_info.get("LinkData")
        if da is not None and len(da) > 0:
            self.current_link_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_shell_tags(self, model_info: dict):
        da = model_info.get("ShellData")
        if da is not None and len(da) > 0:
            self.current_shell_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_plane_tags(self, model_info: dict):
        da = model_info.get("PlaneData")
        if da is not None and len(da) > 0:
            self.current_plane_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_brick_tags(self, model_info: dict):
        da = model_info.get("BrickData")
        if da is not None and len(da) > 0:
            self.current_brick_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_contact_tags(self, model_info: dict):
        da = model_info.get("ContactData")
        if da is not None and len(da) > 0:
            self.current_contact_tags = [int(tag) for tag in da.coords["eleTags"].values]

    def _set_current_frame_load_data(self, model_info: dict):
        da = model_info.get("FrameLoadData")
        if da is not None and len(da) > 0:
            self.current_frame_load_data = da

    def get_current_node_tags(self):
        return self.current_node_tags if self.current_node_tags is not None else []

    def get_current_truss_tags(self):
        return self.current_truss_tags if self.current_truss_tags is not None else []

    def get_current_frame_tags(self):
        return self.current_frame_tags if self.current_frame_tags is not None else []

    def get_current_link_tags(self):
        return self.current_link_tags if self.current_link_tags is not None else []

    def get_current_shell_tags(self):
        return self.current_shell_tags if self.current_shell_tags is not None else []

    def get_current_plane_tags(self):
        return self.current_plane_tags if self.current_plane_tags is not None else []

    def get_current_brick_tags(self):
        return self.current_brick_tags if self.current_brick_tags is not None else []

    def get_current_contact_tags(self):
        return self.current_contact_tags if self.current_contact_tags is not None else []

    def get_current_frame_load_data(self):
        return self.current_frame_load_data if self.current_frame_load_data is not None else []

    # -----------------------------------------------------------------------------------------------
    def reset_resp_step_data(self):
        self.times = []
        for key in self.model_info_steps:
            self.model_info_steps[key] = []

    def add_resp_data_to_dataset(self):
        model_info_steps = {}
        for key, data in self.model_info_steps.items():
            if len(data) == 0:
                continue
            elif len(data) == 1:
                new_data = data[0].expand_dims(dim="time")
                new_data.coords["time"] = [self.times[0]]
            else:
                new_data = xr.concat(data, dim="time", join="outer", fill_value=np.nan)
                new_data.coords["time"] = self.times
            model_info_steps[key] = new_data
            # if all data is void, this code will not work
            if "ModelUpdate" not in self.model_info_steps or len(self.model_info_steps["ModelUpdate"]) == 0:
                model_update = [1] if self.model_update else [0]
                model_info_steps["ModelUpdate"] = xr.DataArray(model_update, name="ModelUpdate")
        self.model_info_steps = model_info_steps

    def add_resp_data_to_datatree(self, dt):
        self.add_resp_data_to_dataset()
        for data in self.model_info_steps.values():
            if len(data) == 0:
                continue
            dt[f"ModelInfo/{data.name}"] = xr.Dataset({data.name: data})
        return dt

    @staticmethod
    def read_response(
        dt: xr.DataTree | list[xr.DataTree],
        data_type: str | None = None,
        unit_factors: dict | None = None,
        lazy: bool = True,
    ):
        dts = dt if isinstance(dt, (list, tuple)) else [dt]
        if not dts:
            return {}, False

        # ---- ModelUpdate: ONLY from the first part (if exists) ----
        mu = None
        if "ModelInfo" in dts[0]:
            mi0 = dts[0]["ModelInfo"]
            if "ModelUpdate" in mi0 and mi0["ModelUpdate"].ds is not None:
                ds_mu = mi0["ModelUpdate"].ds
                mu = ds_mu.get("ModelUpdate") or (
                    next(iter(ds_mu.data_vars.values()), None) if ds_mu.data_vars else None
                )
        model_update = bool(int(mu.item())) if mu is not None else False

        # ---- collect & aggregate ModelInfo children only ----
        buckets: dict[str, list[xr.Dataset]] = {}
        for t in dts:
            if "ModelInfo" not in t:
                continue
            for child in t["ModelInfo"].children.values():  # direct children only
                if child.ds is None:
                    continue
                if data_type is not None and child.name != data_type:
                    continue
                if lazy:
                    buckets.setdefault(child.name, []).append(child.ds)
                else:
                    buckets.setdefault(child.name, []).append(child.ds.load())

        if not buckets:
            model_info = {}
            return model_info, model_update

        # ---- concat/merge per child ----
        model_info = {}
        for name, dss in buckets.items():
            ds = dss[0] if len(dss) == 1 else xr.concat(dss, dim="time", join="outer", fill_value=np.nan)
            model_info[name] = ds[name] if name in ds.data_vars else ds

        if unit_factors:
            model_info = _unit_transform(model_info, unit_factors)

        model_info["ModelUpdate"] = model_update

        return model_info, model_update

    @staticmethod
    def read_data(dt: xr.DataTree | list[xr.DataTree], data_type: str | None = None, lazy: bool = True):
        """Read data from the data tree

        Parameters:
        -----------
        dt: xr.DataTree | list[xr.DataTree]
            The data tree or list of data trees.
        data_type: str
            The data type to read.
        """
        model_info, model_update = ModelInfoStepData.read_response(dt, data_type=data_type, lazy=lazy)
        if data_type is not None and data_type in model_info:
            data = model_info[data_type]
            if model_update:
                return data
            return data.isel(time=0)
        return model_info


def _unit_transform(model_info, unit_factors):
    disp = unit_factors["disp"]

    def scale_attr(da, key):
        if key in da.attrs and da.attrs[key] is not None:
            da.attrs[key] *= disp

    def scale_bounds(da):
        b = da.attrs.get("bounds")
        if b is not None:
            da.attrs["bounds"] = tuple(v * disp for v in b)

    # --------------------------------------------------
    # NodalData
    if "NodalData" in model_info:
        da = model_info["NodalData"]
        model_info["NodalData"] = da * disp  # lazy-safe

        scale_attr(model_info["NodalData"], "minBoundSize")
        scale_attr(model_info["NodalData"], "maxBoundSize")
        scale_bounds(model_info["NodalData"])

    # --------------------------------------------------
    # FixedNodalData
    if "FixedNodalData" in model_info:
        da = model_info["FixedNodalData"]
        if "info" in da.coords:
            model_info["FixedNodalData"] = da.where(
                ~da.coords["info"].isin(["x", "y", "z"]),
                da * disp,
            )

    # --------------------------------------------------
    # MPConstraintData
    if "MPConstraintData" in model_info:
        da = model_info["MPConstraintData"]
        if "info" in da.coords:
            model_info["MPConstraintData"] = da.where(
                ~da.coords["info"].isin(["xo", "yo", "zo"]),
                da * disp,
            )

    # --------------------------------------------------
    # Element centers
    if "eleCenters" in model_info:
        model_info["eleCenters"] = model_info["eleCenters"] * disp

    return model_info
