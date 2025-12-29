from __future__ import annotations

import numpy as np
import xarray as xr

from ...utils import get_opensees_module
from ._response_base import ResponseBase

ops = get_opensees_module()

RESP_NAME = "NodalResponses"


class NodalRespStepData(ResponseBase):
    def __init__(self, node_tags, **kwargs):
        super().__init__(**kwargs)

        self.resp_name = RESP_NAME
        self.resp_types = [
            "disp",
            "vel",
            "accel",
            "reaction",
            "reactionIncInertia",
            "rayleighForces",
            "pressure",
        ]

        self.node_tags = node_tags if node_tags is not None else ops.getNodeTags()

        self.attrs = {
            "UX": "Displacement in X direction",
            "UY": "Displacement in Y direction",
            "UZ": "Displacement in Z direction",
            "RX": "Rotation about X axis",
            "RY": "Rotation about Y axis",
            "RZ": "Rotation about Z axis",
        }

        self.add_resp_data_one_step(node_tags=node_tags)

    def add_resp_data_one_step(self, node_tags):
        # node_tags = ops.getNodeTags()
        disp, vel, accel, pressure = _get_nodal_resp(node_tags, dtype=self.dtype)
        reacts, reacts_inertia, rayleigh_forces = _get_nodal_react(node_tags, dtype=self.dtype)

        if self.model_update:
            datas = [disp, vel, accel, reacts, reacts_inertia, rayleigh_forces]
            data_vars = {}
            for name, data_ in zip(self.resp_types, datas):
                data_vars[name] = (["nodeTags", "DOFs"], data_)
            data_vars["pressure"] = (["nodeTags"], pressure)
            # can have different dimensions and coordinates
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "nodeTags": node_tags,
                    "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
                },
                attrs=self.attrs,
            )
            self.resp_step_data_list.append(ds)
        else:
            datas = [disp, vel, accel, reacts, reacts_inertia, rayleigh_forces, pressure]
            for name, data_ in zip(self.resp_types, datas):
                self.resp_step_data_dict[name].append(data_)

        self.move_one_step(time_value=ops.getTime())

    def add_resp_data_to_dataset(self):

        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_step_data = xr.concat(self.resp_step_data_list, dim="time", join="outer", fill_value=np.nan)
            self.resp_step_data.coords["time"] = self.times
        else:
            data_vars = {}
            for name in self.resp_types[:-1]:
                data_vars[name] = (["time", "nodeTags", "DOFs"], self.resp_step_data_dict[name])
            data_vars["pressure"] = (["time", "nodeTags"], self.resp_step_data_dict["pressure"])
            self.resp_step_data = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "nodeTags": self.node_tags,
                    "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
                },
                attrs=self.attrs,
            )

    @staticmethod
    def read_response(
        dt: xr.DataTree | list[xr.DataTree],
        resp_type: str | None = None,
        node_tags=None,
        unit_factors: dict | None = None,
        lazy: bool = True,
    ) -> xr.Dataset | xr.DataArray:
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
            ds = NodalRespStepData._select_node_tags(ds, node_tags=node_tags)

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


def _unit_transform(resp_steps: xr.Dataset, unit_factors: dict[str, float] | None) -> xr.Dataset:
    if not unit_factors:
        return resp_steps

    d = resp_steps
    dofs = d.get("DOFs") or d.coords.get("DOFs")
    if dofs is None:
        raise KeyError("DOFs coordinate not found")  # noqa: TRY003

    trans = ["UX", "UY", "UZ"]
    rot = ["RX", "RY", "RZ"]

    m_trans = dofs.isin(trans)
    m_rot = dofs.isin(rot)

    def _scale_trans(var: str, factor: float) -> xr.DataArray:
        da = d[var]
        return da.where(~m_trans, da * factor)

    def _scale_force_moment(var: str) -> xr.DataArray:
        da = d[var]
        return da.where(~m_trans, da * unit_factors["force"]).where(~m_rot, da * unit_factors["moment"])

    def _scale_vel_acc(var: str, ang_key: str) -> xr.DataArray:
        da = d[var]
        da = da.where(~m_trans, da * unit_factors[var])  # linear part
        return da.where(~m_rot, d[var] * unit_factors[ang_key])  # angular part

    updates = {}

    if "disp" in d:
        updates["disp"] = _scale_trans("disp", unit_factors["disp"])

    if "vel" in d:
        updates["vel"] = _scale_vel_acc("vel", "angular_vel")

    if "accel" in d:
        updates["accel"] = _scale_vel_acc("accel", "angular_accel")

    for v in ("reaction", "reactionIncInertia", "rayleighForces"):
        if v in d:
            updates[v] = _scale_force_moment(v)

    if "pressure" in d:
        updates["pressure"] = d["pressure"] * unit_factors["stress"]

    return d.assign(**updates) if updates else d


def handle_1d(disp, vel, accel):
    return (
        [*disp, 0.0, 0.0, 0.0, 0.0, 0.0],
        [*vel, 0.0, 0.0, 0.0, 0.0, 0.0],
        [*accel, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def handle_2d(disp, vel, accel):
    if len(disp) == 1:
        return handle_1d(disp, vel, accel)
    elif len(disp) == 2:
        return (
            [*disp, 0.0, 0.0, 0.0, 0.0],
            [*vel, 0.0, 0.0, 0.0, 0.0],
            [*accel, 0.0, 0.0, 0.0, 0.0],
        )
    elif len(disp) >= 3:
        # Assume (ux, uy, rz)
        return (
            [disp[0], disp[1], 0.0, 0.0, 0.0, disp[2]],
            [vel[0], vel[1], 0.0, 0.0, 0.0, vel[2]],
            [accel[0], accel[1], 0.0, 0.0, 0.0, accel[2]],
        )


def handle_3d(disp, vel, accel):
    if len(disp) == 3:
        return (
            [*disp, 0.0, 0.0, 0.0],
            [*vel, 0.0, 0.0, 0.0],
            [*accel, 0.0, 0.0, 0.0],
        )
    elif len(disp) == 4:
        return (
            [disp[0], disp[1], disp[2], 0.0, 0.0, disp[3]],
            [vel[0], vel[1], vel[2], 0.0, 0.0, vel[3]],
            [accel[0], accel[1], accel[2], 0.0, 0.0, accel[3]],
        )
    elif len(disp) < 6:
        pad_len = 6 - len(disp)
        return (
            disp + [0.0] * pad_len,
            vel + [0.0] * pad_len,
            accel + [0.0] * pad_len,
        )
    else:
        return (
            disp[:6],
            vel[:6],
            accel[:6],
        )


def _get_nodal_resp(node_tags, dtype: dict):
    node_disp, node_vel, node_accel, node_pressure = [], [], [], []
    all_node_tags = set(ops.getNodeTags())

    for tag in map(int, node_tags):
        if tag in all_node_tags:
            coord = ops.nodeCoord(tag)
            ndim = len(coord)
            disp = ops.nodeDisp(tag)
            vel = ops.nodeVel(tag)
            accel = ops.nodeAccel(tag)

            if ndim == 1:
                d, v, a = handle_1d(disp, vel, accel)
            elif ndim == 2:
                d, v, a = handle_2d(disp, vel, accel)
            else:
                d, v, a = handle_3d(disp, vel, accel)
        else:
            d = v = a = [np.nan] * 6

        node_disp.append(d)
        node_vel.append(v)
        node_accel.append(a)
        node_pressure.append(ops.nodePressure(tag))

    return (
        np.array(node_disp, dtype=dtype["float"]),
        np.array(node_vel, dtype=dtype["float"]),
        np.array(node_accel, dtype=dtype["float"]),
        np.array(node_pressure, dtype=dtype["float"]),
    )


def _get_react(tags):
    forces = []  # 6 data each row, Ux, Uy, Uz, Rx, Ry, Rz
    for tag in tags:
        tag = int(tag)
        if tag in ops.getNodeTags():
            coord = ops.nodeCoord(tag)
            fo = ops.nodeReaction(tag)
            ndim, ndf = len(coord), len(fo)
            if ndim == 1 or (ndim == 2 and ndf == 1):
                fo.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            elif ndim == 2 and ndf == 2:
                fo.extend([0.0, 0.0, 0.0, 0.0])
            elif ndim == 2 and ndf >= 3:
                fo = [fo[0], fo[1], 0.0, 0.0, 0.0, fo[2]]
            elif ndim == 3 and ndf == 3:
                fo.extend([0.0, 0.0, 0.0])
            elif ndim == 3 and ndf < 6:  # 3 ndim 6 dof
                fo.extend([0] * (6 - len(fo)))
            elif ndim == 3 and ndf > 6:
                fo = fo[:6]
        else:
            fo = [np.nan] * 6
        forces.append(fo)
    return forces


def _get_nodal_react(node_tags, dtype: dict):
    ops.reactions()
    reacts = np.array(_get_react(node_tags), dtype=dtype["float"])
    # rayleighForces
    ops.reactions("-rayleigh")
    rayleigh_forces = np.array(_get_react(node_tags), dtype=dtype["float"])
    # Include Inertia
    ops.reactions("-dynamic")
    reacts_inertia = np.array(_get_react(node_tags), dtype=dtype["float"])
    return reacts, reacts_inertia, rayleigh_forces
