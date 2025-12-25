from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase

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
    ):
        dts = dt if isinstance(dt, (list, tuple)) else [dt]
        if not dts:
            return []

        # collect datasets under /RESP_NAME (skip missing)
        dss = []
        for t in dts:
            if RESP_NAME not in t:
                continue
            node = t[f"/{RESP_NAME}"]
            if node.ds is not None:
                dss.append(node.ds)

        if not dss:
            return []

        ds = dss[0] if len(dss) == 1 else xr.concat(dss, dim="time", join="outer")
        ds = _unit_transform(ds, unit_factors=unit_factors)

        if resp_type is None:
            return ds if ele_tags is None else ds.sel(eleTags=ele_tags)

        if resp_type not in ds.data_vars:
            raise ValueError(f"resp_type {resp_type} not found in {list(ds.data_vars.keys())}")  # noqa: TRY003

        da = ds[resp_type]
        return da if ele_tags is None else da.sel(eleTags=ele_tags)


def _unit_transform(resp_steps, unit_factors):
    if not unit_factors:
        return resp_steps

    return resp_steps.assign(
        axialForce=resp_steps["axialForce"] * unit_factors["force"],
        axialDefo=resp_steps["axialDefo"] * unit_factors["disp"],
        Stress=resp_steps["Stress"] * unit_factors["stress"],
    )


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
