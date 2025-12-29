from __future__ import annotations

import numpy as np
import xarray as xr

from ...utils import get_opensees_module
from ._response_base import ResponseBase

ops = get_opensees_module()

RESP_NAME = "TrussResponses"


class TrussRespStepData(ResponseBase):
    def __init__(self, ele_tags, **kwargs):
        super().__init__(**kwargs)

        self.resp_name = RESP_NAME
        self.resp_types = ["axialForce", "axialDefo", "Stress", "Strain"]

        self.ele_tags = ele_tags
        self.add_resp_data_one_step(ele_tags=ele_tags)

    def add_resp_data_one_step(self, ele_tags):
        data = _get_truss_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                for name, data_ in zip(self.resp_types, data):
                    data_vars[name] = (["eleTags"], data_)
                ds = xr.Dataset(data_vars=data_vars, coords={"eleTags": ele_tags})
            else:
                for name in self.resp_types:
                    data_vars[name] = xr.DataArray([])
                ds = xr.Dataset(data_vars=data_vars)
            self.resp_step_data_list.append(ds)
        else:
            for name, data_ in zip(self.resp_types, data):
                self.resp_step_data_dict[name].append(data_)

        self.move_one_step(time_value=ops.getTime())

    def add_resp_data_to_dataset(self):

        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_step_data = xr.concat(self.resp_step_data_list, dim="time", join="outer")
            self.resp_step_data.coords["time"] = self.times
        else:
            data_vars = {}
            for name, data in self.resp_step_data_dict.items():
                data_vars[name] = (["time", "eleTags"], data)
            self.resp_step_data = xr.Dataset(
                data_vars=data_vars,
                coords={"time": self.times, "eleTags": self.ele_tags},
            )

    @staticmethod
    def read_response(
        dt: xr.DataTree | list[xr.DataTree],
        resp_type: str | None = None,
        ele_tags=None,
        unit_factors: dict | None = None,
        lazy: bool = True,
    ):
        dts = dt if isinstance(dt, (list, tuple)) else [dt]
        if not dts:
            return xr.Dataset()

        dss: list[xr.Dataset] = []
        for t in dts:
            if RESP_NAME not in t:
                continue
            ds = t[f"/{RESP_NAME}"].ds
            if ds is None:
                continue

            # 1) preselect variable(s)
            if resp_type is not None:
                if resp_type not in ds.data_vars:
                    continue
                ds = ds[[resp_type]]

            # 2) early nodeTags selection
            ds = TrussRespStepData._select_ele_tags(ds, ele_tags=ele_tags)

            # 3) if not lazy, load per-part to avoid lazy-concat instability
            if not lazy:
                ds = ds.load()

            dss.append(ds)

        if not dss:
            return xr.Dataset()

        resp_steps = dss[0] if len(dss) == 1 else xr.concat(dss, dim="time", join="outer", fill_value=np.nan)

        resp_steps = _unit_transform(resp_steps, unit_factors)

        if resp_type is not None and resp_type in resp_steps:
            return resp_steps[resp_type]
        return resp_steps


def _unit_transform(resp_steps, unit_factors):
    if not unit_factors:
        return resp_steps

    updates = {}

    if "axialForce" in resp_steps:
        updates["axialForce"] = resp_steps["axialForce"] * unit_factors["force"]

    if "axialDefo" in resp_steps:
        updates["axialDefo"] = resp_steps["axialDefo"] * unit_factors["disp"]

    if "Stress" in resp_steps:
        updates["Stress"] = resp_steps["Stress"] * unit_factors["stress"]

    return resp_steps.assign(**updates) if updates else resp_steps


def _get_truss_resp(truss_tags, dtype: dict):
    forces, defos, stressss, strains = [], [], [], []
    for etag in truss_tags:
        etag = int(etag)
        force = ops.eleResponse(etag, "axialForce")
        force = _reshape_resp(force)
        defo = ops.eleResponse(etag, "basicDeformation")
        defo = _reshape_resp(defo)
        stress = ops.eleResponse(etag, "material", "1", "stress")
        stress = _reshape_resp(stress)

        strain = ops.eleResponse(etag, "material", "1", "strain")
        if len(strain) == 0:
            strain = ops.eleResponse(etag, "section", "1", "deformation")
        strain = _reshape_resp(strain)

        forces.append(force)
        defos.append(defo)
        stressss.append(stress)
        strains.append(strain)

    forces = np.array(forces, dtype=dtype["float"])
    defos = np.array(defos, dtype=dtype["float"])
    stressss = np.array(stressss, dtype=dtype["float"])
    strains = np.array(strains, dtype=dtype["float"])
    return forces, defos, stressss, strains


def _reshape_resp(data):
    if len(data) == 0:
        return 0.0
    else:
        return data[0]
