from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase

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
            self.resp_step_data = xr.concat(self.resp_step_data_list, dim="time", join="outer")
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
    ) -> xr.Dataset | xr.DataArray:
        # ---- normalize dt to list ----
        dts = dt if isinstance(dt, (list, tuple)) else [dt]
        if not dts:
            return []

        # ---- collect response datasets under /RESP_NAME ----
        dss = []
        for t in dts:
            if RESP_NAME not in t:
                continue
            node = t[f"/{RESP_NAME}"]
            if node.ds is not None:
                dss.append(node.ds)

        if not dss:
            return []

        # ---- concat along time if multiple parts ----
        resp_steps = dss[0] if len(dss) == 1 else xr.concat(dss, dim="time", join="outer")

        # ---- unit transform ----
        resp_steps = _unit_transform(resp_steps, unit_factors)

        # ---- selection logic (unchanged semantics) ----
        if resp_type is None:
            return resp_steps if node_tags is None else resp_steps.sel(nodeTags=node_tags)

        if resp_type not in resp_steps.data_vars:
            raise ValueError(f"resp_type {resp_type} not found in {list(resp_steps.data_vars.keys())}")  # noqa: TRY003

        da = resp_steps[resp_type]
        return da if node_tags is None else da.sel(nodeTags=node_tags)


def _unit_transform(resp_steps: xr.Dataset, unit_factors: dict[str, float] | None) -> xr.Dataset:
    if not unit_factors:
        return resp_steps

    d = resp_steps
    dofs = d.get("DOFs", d.coords.get("DOFs", None))
    if dofs is None:
        raise KeyError("DOFs coordinate not found")  # noqa: TRY003

    trans = ["UX", "UY", "UZ"]
    rot = ["RX", "RY", "RZ"]

    def scale(var: str, factor: float, sel: list[str] | None = None) -> xr.DataArray:
        da = d[var]
        if sel is None:
            return da * factor
        m = dofs.isin(sel)
        return da.where(~m, da * factor)

    return d.assign(
        disp=scale("disp", unit_factors["disp"], trans),
        vel=scale("vel", unit_factors["vel"], trans).where(~dofs.isin(rot), d["vel"] * unit_factors["angular_vel"]),
        accel=scale("accel", unit_factors["accel"], trans).where(
            ~dofs.isin(rot), d["accel"] * unit_factors["angular_accel"]
        ),
        reaction=scale("reaction", unit_factors["force"], trans).where(
            ~dofs.isin(rot), d["reaction"] * unit_factors["moment"]
        ),
        reactionIncInertia=scale("reactionIncInertia", unit_factors["force"], trans).where(
            ~dofs.isin(rot), d["reactionIncInertia"] * unit_factors["moment"]
        ),
        rayleighForces=scale("rayleighForces", unit_factors["force"], trans).where(
            ~dofs.isin(rot), d["rayleighForces"] * unit_factors["moment"]
        ),
        pressure=d["pressure"] * unit_factors["stress"],
    )


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
