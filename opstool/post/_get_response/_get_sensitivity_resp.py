from __future__ import annotations

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase

RESP_NAME = "SensitivityResponses"


class SensitivityRespStepData(ResponseBase):
    def __init__(self, node_tags=None, ele_tags=None, sens_para_tags=None, **kwargs):
        super().__init__(**kwargs)
        self.resp_name = RESP_NAME
        self.resp_types = [
            "disp",
            "vel",
            "accel",
            "pressure",
            "lambdas",
            # "sensSectionForce",
        ]
        self.node_tags = node_tags if node_tags is not None else ops.getNodeTags()
        self.ele_tags = ele_tags if ele_tags is not None else []  # for future use
        self.sens_para_tags = sens_para_tags if sens_para_tags is not None else ops.getParamTags()

        self.attrs = {
            "UX": "Displacement in X direction",
            "UY": "Displacement in Y direction",
            "UZ": "Displacement in Z direction",
            "RX": "Rotation about X axis",
            "RY": "Rotation about Y axis",
            "RZ": "Rotation about Z axis",
        }
        self.DOFs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
        self.patternTags = None

        self.add_resp_data_one_step(node_tags=self.node_tags, sens_para_tags=self.sens_para_tags)

    def add_resp_data_one_step(self, node_tags, sens_para_tags):
        node_tags = node_tags if node_tags is not None else ops.getNodeTags()
        sens_para_tags = sens_para_tags if sens_para_tags is not None else ops.getParamTags()

        disp, vel, accel, pressure = _get_nodal_sens_resp(node_tags, sens_para_tags, dtype=self.dtype)
        lambdas_ = _get_sens_lambda(sens_para_tags, dtype=self.dtype)

        if self.patternTags is None:
            self.patternTags = [] if len(lambdas_) == 0 else np.arange(lambdas_.shape[1]) + 1

        if self.model_update:
            datas = [disp, vel, accel]
            data_vars = {}
            for name, data_ in zip(["disp", "vel", "accel"], datas):
                data_vars[name] = (["paraTags", "nodeTags", "DOFs"], data_)
            data_vars["pressure"] = (["paraTags", "nodeTags"], pressure)
            data_vars["lambdas"] = (["paraTags", "patternTags"], lambdas_)
            # can have different dimensions and coordinates
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "paraTags": sens_para_tags,
                    "nodeTags": node_tags,
                    "DOFs": self.DOFs,
                    "patternTags": self.patternTags,
                },
                attrs=self.attrs,
            )
            self.resp_step_data_list.append(ds)
        else:
            datas = [disp, vel, accel, pressure, lambdas_]
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
            for name in ["disp", "vel", "accel"]:
                data_vars[name] = (["time", "paraTags", "nodeTags", "DOFs"], self.resp_step_data_dict[name])
            data_vars["pressure"] = (["time", "paraTags", "nodeTags"], self.resp_step_data_dict["pressure"])
            data_vars["lambdas"] = (["time", "paraTags", "patternTags"], self.resp_step_data_dict["lambdas"])
            self.resp_step_data = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "paraTags": self.sens_para_tags,
                    "nodeTags": self.node_tags,
                    "DOFs": self.DOFs,
                    "patternTags": self.patternTags,
                },
                attrs=self.attrs,
            )

    @staticmethod
    def read_response(dt: xr.DataTree | list[xr.DataTree], resp_type: str | None = None):
        dts = dt if isinstance(dt, (list, tuple)) else [dt]
        if not dts:
            return []

        dss = []
        for t in dts:
            if RESP_NAME in t and t[f"/{RESP_NAME}"].ds is not None:
                dss.append(t[f"/{RESP_NAME}"].ds)

        if not dss:
            return []

        ds = dss[0] if len(dss) == 1 else xr.concat(dss, dim="time", join="outer")

        if resp_type is None:
            return ds

        if resp_type not in ds.data_vars:
            raise ValueError(f"resp_type {resp_type} not found in {list(ds.data_vars.keys())}")  # noqa: TRY003

        return ds[resp_type]


def _get_nodal_sens_resp(node_tags, sens_para_tags, dtype):
    all_node_tags = set(ops.getNodeTags())

    all_sens_disp = []
    all_sens_vel = []
    all_sens_accel = []
    all_sens_pressure = []

    for para_tag in map(int, sens_para_tags):
        node_sens_disp, node_sens_vel, node_sens_accel, node_sens_pressure = [], [], [], []

        for ntag in map(int, node_tags):
            if ntag in all_node_tags:
                disp, vel, accel, pressure = _get_node_sensitivities(ntag, para_tag)
            else:
                disp = [np.nan] * 6
                vel = [np.nan] * 6
                accel = [np.nan] * 6
                pressure = np.nan

            node_sens_disp.append(disp)
            node_sens_vel.append(vel)
            node_sens_accel.append(accel)
            node_sens_pressure.append(pressure)

        all_sens_disp.append(node_sens_disp)
        all_sens_vel.append(node_sens_vel)
        all_sens_accel.append(node_sens_accel)
        all_sens_pressure.append(node_sens_pressure)

    return (
        np.array(all_sens_disp, dtype=dtype["float"]),
        np.array(all_sens_vel, dtype=dtype["float"]),
        np.array(all_sens_accel, dtype=dtype["float"]),
        np.array(all_sens_pressure, dtype=dtype["float"]),
    )


def _get_node_sensitivities(ntag, para_tag):
    coord = ops.nodeCoord(ntag)
    disp_vec = ops.nodeDisp(ntag)
    ndim = len(coord)
    ndof = len(disp_vec)

    disp = [ops.sensNodeDisp(ntag, i + 1, para_tag) for i in range(ndof)]
    vel = [ops.sensNodeVel(ntag, i + 1, para_tag) for i in range(ndof)]
    accel = [ops.sensNodeAccel(ntag, i + 1, para_tag) for i in range(ndof)]
    pressure = ops.sensNodePressure(ntag, para_tag)

    disp = _standardize_response(disp, ndim, ndof)
    vel = _standardize_response(vel, ndim, ndof)
    accel = _standardize_response(accel, ndim, ndof)

    return disp, vel, accel, pressure


def _standardize_response(vec, ndim, ndof):
    """Convert DOF-based vector to 6-length standard response [ux, uy, uz, rx, ry, rz]"""
    if ndim == 1:
        return vec + [0.0] * (6 - len(vec))
    if ndim == 2:
        if ndof == 2:
            return vec + [0.0] * 4
        if ndof >= 3:
            return [vec[0], vec[1], 0.0, 0.0, 0.0, vec[2]]
    if ndim == 3:
        if ndof == 3:
            return vec + [0.0] * 3
        if ndof == 4:
            return [vec[0], vec[1], vec[2], 0.0, 0.0, vec[3]]
        if ndof < 6:
            return vec + [0.0] * (6 - len(vec))
    return vec[:6]


def _get_sens_lambda(sens_para_tags, dtype):
    pattern_tags = ops.getPatterns()
    all_sens_lambdas = []
    for para_tag in sens_para_tags:
        para_tag = int(para_tag)
        sens_lambads = []
        for ptag in pattern_tags:
            sens_lambads.append(ops.sensLambda(ptag, para_tag))
        all_sens_lambdas.append(sens_lambads)
    return np.array(all_sens_lambdas, dtype=dtype["float"])


def _get_sec_sens(ele_tags, sens_para_tags):
    pass
