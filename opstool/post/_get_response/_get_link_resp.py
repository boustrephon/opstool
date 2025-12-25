from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase

RESP_NAME = "LinkResponses"


class LinkRespStepData(ResponseBase):
    def __init__(self, ele_tags, **kwargs):
        super().__init__(**kwargs)

        self.resp_name = RESP_NAME
        self.resp_types = ["basicDeformation", "basicForce"]
        self.ele_tags = ele_tags
        self.attrs = {
            "DOFs": "The DOFs are aligned with the local coordinate system. "
            "Note that these DOFs are not necessarily valid unless all degrees of freedom are "
            "assigned to the material (e.g., all six DOFs in 3D). "
            "For cases where the material is assigned to only partial DOFs, "
            "the actual DOFs are arranged sequentially, with the remaining ones padded with zeros."
        }
        self.DOFs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]

        self.add_resp_data_one_step(ele_tags=ele_tags)

    def add_resp_data_one_step(self, ele_tags):
        data = _get_link_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                for name, data_ in zip(self.resp_types, data):
                    data_vars[name] = (["eleTags", "DOFs"], data_)
                ds = xr.Dataset(data_vars=data_vars, coords={"eleTags": ele_tags, "DOFs": self.DOFs}, attrs=self.attrs)
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
            for name, data_ in self.resp_step_data_dict.items():
                data_vars[name] = (["time", "eleTags", "DOFs"], data_)
            self.resp_step_data = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "DOFs": self.DOFs,
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

    ff, mf, df = (unit_factors[k] for k in ("force", "moment", "disp"))
    trans, rot = ["UX", "UY", "UZ"], ["RX", "RY", "RZ"]

    def scale(da: xr.DataArray, keys: list[str], fac: float) -> xr.DataArray:
        c = da.coords.get("DOFs")
        if c is None:
            return da
        m = c.isin(keys)
        return da.where(~m, da * fac)

    if "basicForce" in resp_steps:
        bf = resp_steps["basicForce"]
        bf = scale(bf, trans, ff)
        bf = scale(bf, rot, mf)
    else:
        bf = None

    bd = scale(resp_steps["basicDeformation"], trans, df) if "basicDeformation" in resp_steps else None

    updates = {}
    if bf is not None:
        updates["basicForce"] = bf
    if bd is not None:
        updates["basicDeformation"] = bd
    return resp_steps.assign(**updates) if updates else resp_steps


def _get_link_resp(link_tags, dtype):
    defos, forces = [], []
    for etag in link_tags:
        etag = int(etag)
        defo = _get_link_resp_by_type(
            etag,
            (
                "basicDeformations",
                "basicDeformation",
                "deformations",
                "deformation",
                "basicDisplacements",
                "basicDisplacement",
            ),
        )
        force = _get_link_resp_by_type(etag, ("basicForces", "basicForce"))
        defos.append(defo)
        forces.append(force)
    defos = np.array(defos, dtype=dtype["float"])
    forces = np.array(forces, dtype=dtype["float"])
    return defos, forces


def _get_link_resp_by_type(etag, etypes):
    etag = int(etag)
    ntags = ops.eleNodes(etag)
    ndim = len(ops.nodeCoord(ntags[0]))
    resp = []
    for name in etypes:
        resp = ops.eleResponse(etag, name)
        if len(resp) > 0:
            break
    if len(resp) == 0:
        resp = [0.0] * 6
    elif ndim == 2 and len(resp) == 3:
        resp = [resp[0], resp[1], 0.0, 0.0, 0.0, resp[2]]
    elif len(resp) < 6:  # don't know dofs
        resp = resp + [0.0] * (6 - len(resp))
    elif len(resp) > 6:
        resp = resp[:6]
    return resp
