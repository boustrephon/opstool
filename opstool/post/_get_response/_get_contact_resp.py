from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import suppress_ops_print
from ._response_base import ResponseBase

RESP_NAME = "ContactResponses"


class ContactRespStepData(ResponseBase):
    def __init__(self, ele_tags, **kwargs):
        super().__init__(**kwargs)

        self.resp_name = RESP_NAME
        self.resp_types = ["globalForces", "localForces", "localDisp", "slips"]

        self.ele_tags = ele_tags

        self.attrs = {
            "Px": "Global force in the x-direction on the constrained node",
            "Py": "Global force in the y-direction on the constrained node",
            "Pz": "Global force in the z-direction on the constrained node",
            "N": "Normal force or deformation",
            "Tx": "Tangential force or deformation in the x-direction",
            "Ty": "Tangential force or deformation in the y-direction",
        }

        self.add_resp_data_one_step(ele_tags=ele_tags)

    def add_resp_data_one_step(self, ele_tags):
        with suppress_ops_print():
            global_forces, forces, defos, slips = _get_contact_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                data_vars["globalForces"] = (["eleTags", "globalDOFs"], global_forces)
                data_vars["localForces"] = (["eleTags", "localDOFs"], forces)
                data_vars["localDisp"] = (["eleTags", "localDOFs"], defos)
                data_vars["slips"] = (["eleTags", "slipDOFs"], slips)
                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "eleTags": ele_tags,
                        "globalDOFs": ["Px", "Py", "Pz"],
                        "localDOFs": ["N", "Tx", "Ty"],
                        "slipDOFs": ["Tx", "Ty"],
                    },
                    attrs=self.attrs,
                )
            else:
                data_vars["globalForces"] = xr.DataArray([])
                data_vars["localForces"] = xr.DataArray([])
                data_vars["localDisp"] = xr.DataArray([])
                data_vars["slips"] = xr.DataArray([])
                ds = xr.Dataset(data_vars=data_vars)
            self.resp_step_data_list.append(ds)
        else:
            datas = [global_forces, forces, defos, slips]
            for name, da in zip(self.resp_types, datas):
                self.resp_step_data_dict[name].append(da)

        self.move_one_step(time_value=ops.getTime())

    def add_resp_data_to_dataset(self):

        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_step_data = xr.concat(self.resp_step_data_list, dim="time", join="outer")
            self.resp_step_data.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["globalForces"] = (["time", "eleTags", "globalDOFs"], self.resp_step_data_dict["globalForces"])
            data_vars["localForces"] = (["time", "eleTags", "localDOFs"], self.resp_step_data_dict["localForces"])
            data_vars["localDisp"] = (["time", "eleTags", "localDOFs"], self.resp_step_data_dict["localDisp"])
            data_vars["slips"] = (["time", "eleTags", "slipDOFs"], self.resp_step_data_dict["slips"])
            self.resp_step_data = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "globalDOFs": ["Px", "Py", "Pz"],
                    "localDOFs": ["N", "Tx", "Ty"],
                    "slipDOFs": ["Tx", "Ty"],
                },
                attrs=self.attrs,
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

    ff = unit_factors["force"]
    df = unit_factors["disp"]

    return resp_steps.assign(
        globalForces=resp_steps["globalForces"] * ff,
        localForces=resp_steps["localForces"] * ff,
        localDisp=resp_steps["localDisp"] * df,
        slips=resp_steps["slips"] * df,
    )


def _get_contact_resp(link_tags, dtype):
    defos, forces, slips, global_forces = [], [], [], []
    for etag in link_tags:
        etag = int(etag)
        global_fo = _get_contact_resp_by_type(etag, ("force", "forces"), type_="global")
        defo = _get_contact_resp_by_type(etag, ("localDisplacement", "localDispJump"), type_="local")
        force = _get_contact_resp_by_type(
            etag, ("localForce", "localForces", "forcescalars", "forcescalar"), type_="local"
        )
        slip = _get_contact_resp_by_type(etag, ("slip",), type_="slip")
        global_forces.append(global_fo)
        defos.append(defo)
        forces.append(force)
        slips.append(slip)
    defos = np.array(defos, dtype=dtype["float"])
    forces = np.array(forces, dtype=dtype["float"])
    slips = np.array(slips, dtype=dtype["float"])
    global_forces = np.array(global_forces, dtype=dtype["float"])
    return global_forces, forces, defos, slips


def _get_contact_resp_by_type(etag, etypes, type_="local"):
    etag = int(etag)
    resp = _get_valid_ele_response(etag, etypes)

    if type_ == "local":
        return _format_local_response(resp)
    elif type_ == "global":
        return _format_global_response(resp)
    elif type_ == "slip":
        return _format_slip_response(resp)
    else:
        raise ValueError(f"Unsupported response type: {type_}")  # noqa: TRY003


def _get_valid_ele_response(etag, etypes):
    for name in etypes:
        resp = ops.eleResponse(etag, name)
        if resp:
            return resp
    return []


def _format_local_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0, 0.0]
    if len(resp) == 2:
        return [resp[0], resp[1], 0.0]
    return resp[:3]


def _format_global_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0, 0.0]
    if len(resp) == 2:
        return [resp[0], resp[1], 0.0]
    if len(resp) in (4, 6):
        return resp[-3:]
    return resp[-3:]


def _format_slip_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0]
    if len(resp) == 1:
        return [resp[0], resp[0]]
    return resp[:2]
