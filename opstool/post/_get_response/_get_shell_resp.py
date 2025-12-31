from __future__ import annotations

from collections import defaultdict

import numpy as np
import xarray as xr

from ...utils import get_opensees_module, get_shell_gp2node_func, suppress_ops_print
from ._response_base import ResponseBase, expand_to_uniform_array

ops = get_opensees_module()

RESP_NAME = "ShellResponses"


class ShellRespStepData(ResponseBase):
    def __init__(self, ele_tags, compute_nodal_resp: str | None = None, **kargs):
        super().__init__(**kargs)
        self.resp_name = RESP_NAME
        self.resp_types = [
            "sectionForces",
            "sectionDeformations",
            "Stresses",
            "Strains",
            "sectionForcesAtNodes",
            "sectionDeformationsAtNodes",
            "StressesAtNodes",
            "StrainsAtNodes",
        ]
        self.ele_tags = ele_tags

        self.compute_nodal_resp = compute_nodal_resp
        self.nodal_resp_method = compute_nodal_resp

        self.attrs = {
            "FXX,FYY,FXY": "Membrane (in-plane) forces or deformations.",
            "MXX,MYY,MXY": "Bending moments or rotations (out-plane) of plate.",
            "VXZ,VYZ": "Shear forces or deformations.",
            "sigma11, sigma22": "Normal stress (strain) along local x, y",
            "sigma12, sigma23, sigma13": "Shear stress (strain).",
        }
        self.secDOFs = ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"]
        self.stressDOFs = ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"]
        self.GaussPoints = None
        self.fiberPoints = None

        self.add_resp_data_one_step(ele_tags)

    def add_resp_data_one_step(self, ele_tags):
        sec_forces, sec_defos, stresses, strains = _get_shell_resp_one_step(ele_tags, dtype=self.dtype)

        if self.compute_nodal_resp:
            method = self.nodal_resp_method
            node_sec_forces_avg, node_tags = _get_nodal_resp(ele_tags, sec_forces, method=method, dtype=self.dtype)
            node_sec_defo_avg, node_tags = _get_nodal_resp(ele_tags, sec_defos, method=method, dtype=self.dtype)
            node_stresses_avg, node_tags = _get_nodal_resp(ele_tags, stresses, method=method, dtype=self.dtype)
            node_strains_avg, node_tags = _get_nodal_resp(ele_tags, strains, method=method, dtype=self.dtype)
            self.node_tags = node_tags

        if self.GaussPoints is None:
            self.GaussPoints = np.arange(sec_forces.shape[1]) + 1
        if self.fiberPoints is None:
            self.fiberPoints = np.arange(stresses.shape[2]) + 1

        if self.model_update:
            data_vars = {}
            data_vars["sectionForces"] = (["eleTags", "GaussPoints", "secDOFs"], sec_forces)
            data_vars["sectionDeformations"] = (["eleTags", "GaussPoints", "secDOFs"], sec_defos)
            data_vars["Stresses"] = (["eleTags", "GaussPoints", "fiberPoints", "stressDOFs"], stresses)
            data_vars["Strains"] = (["eleTags", "GaussPoints", "fiberPoints", "stressDOFs"], strains)
            coords = {
                "eleTags": ele_tags,
                "GaussPoints": self.GaussPoints,
                "secDOFs": self.secDOFs,
                "fiberPoints": self.fiberPoints,
                "stressDOFs": self.stressDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["sectionForcesAtNodes"] = (["nodeTags", "secDOFs"], node_sec_forces_avg)
                data_vars["sectionDeformationsAtNodes"] = (["nodeTags", "secDOFs"], node_sec_defo_avg)
                if len(node_stresses_avg) > 0:
                    data_vars["StressesAtNodes"] = (["nodeTags", "fiberPoints", "stressDOFs"], node_stresses_avg)
                if len(node_strains_avg) > 0:
                    data_vars["StrainsAtNodes"] = (["nodeTags", "fiberPoints", "stressDOFs"], node_strains_avg)
                coords["nodeTags"] = node_tags
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)
            self.resp_step_data_list.append(ds)
        else:
            self.resp_step_data_dict["sectionForces"].append(sec_forces)
            self.resp_step_data_dict["sectionDeformations"].append(sec_defos)
            self.resp_step_data_dict["Stresses"].append(stresses)
            self.resp_step_data_dict["Strains"].append(strains)
            if self.compute_nodal_resp:
                self.resp_step_data_dict["sectionForcesAtNodes"].append(node_sec_forces_avg)
                self.resp_step_data_dict["sectionDeformationsAtNodes"].append(node_sec_defo_avg)
                if len(node_stresses_avg) > 0:
                    self.resp_step_data_dict["StressesAtNodes"].append(node_stresses_avg)
                if len(node_strains_avg) > 0:
                    self.resp_step_data_dict["StrainsAtNodes"].append(node_strains_avg)
        self.move_one_step(time_value=ops.getTime())

    def add_resp_data_to_dataset(self):

        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_step_data = xr.concat(self.resp_step_data_list, dim="time", join="outer")
            self.resp_step_data.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["sectionForces"] = (
                ["time", "eleTags", "GaussPoints", "secDOFs"],
                self.resp_step_data_dict["sectionForces"],
            )
            data_vars["sectionDeformations"] = (
                ["time", "eleTags", "GaussPoints", "secDOFs"],
                self.resp_step_data_dict["sectionDeformations"],
            )
            data_vars["Stresses"] = (
                ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
                self.resp_step_data_dict["Stresses"],
            )
            data_vars["Strains"] = (
                ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
                self.resp_step_data_dict["Strains"],
            )
            coords = {
                "time": self.times,
                "eleTags": self.ele_tags,
                "GaussPoints": self.GaussPoints,
                "secDOFs": self.secDOFs,
                "fiberPoints": self.fiberPoints,
                "stressDOFs": self.stressDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["sectionForcesAtNodes"] = (
                    ["time", "nodeTags", "secDOFs"],
                    self.resp_step_data_dict["sectionForcesAtNodes"],
                )
                data_vars["sectionDeformationsAtNodes"] = (
                    ["time", "nodeTags", "secDOFs"],
                    self.resp_step_data_dict["sectionDeformationsAtNodes"],
                )
                if len(self.resp_step_data_dict["StressesAtNodes"]) > 0:
                    data_vars["StressesAtNodes"] = (
                        ["time", "nodeTags", "fiberPoints", "stressDOFs"],
                        self.resp_step_data_dict["StressesAtNodes"],
                    )
                if len(self.resp_step_data_dict["StrainsAtNodes"]) > 0:
                    data_vars["StrainsAtNodes"] = (
                        ["time", "nodeTags", "fiberPoints", "stressDOFs"],
                        self.resp_step_data_dict["StrainsAtNodes"],
                    )
                coords["nodeTags"] = self.node_tags
            self.resp_step_data = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)

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
            ds = ShellRespStepData._select_ele_tags(ds, ele_tags=ele_tags)

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


def _unit_transform(resp_steps: xr.Dataset, unit_factors: dict) -> xr.Dataset:
    if not unit_factors:
        return resp_steps

    fpl = float(unit_factors.get("force_per_length", 1.0))
    mpl = float(unit_factors.get("moment_per_length", 1.0))
    sf = float(unit_factors.get("stress", 1.0))

    def scale_by_label(da: xr.DataArray, dim: str, labels: list[str], factor: float) -> xr.DataArray:
        c = da.coords.get(dim, None)
        if c is None:
            return da  # no coord -> skip (stay lazy-safe)
        m = c.isin(labels)
        return da.where(~m, da * factor)

    def maybe_update(ds: xr.Dataset, name: str, fn) -> xr.Dataset:
        if name not in ds.data_vars:
            return ds
        return ds.assign({name: fn(ds[name])})

    # -------------------------
    # sectionForces (+AtNodes)
    # -------------------------
    def _scale_section_forces(da: xr.DataArray) -> xr.DataArray:
        da = scale_by_label(da, "secDOFs", ["FXX", "FYY", "FXY", "VXZ", "VYZ"], fpl)
        da = scale_by_label(da, "secDOFs", ["MXX", "MYY", "MXY"], mpl)
        return da

    resp_steps = maybe_update(resp_steps, "sectionForces", _scale_section_forces)
    resp_steps = maybe_update(resp_steps, "sectionForcesAtNodes", _scale_section_forces)

    # -------------------------
    # stresses (+AtNodes)
    # -------------------------
    resp_steps = maybe_update(resp_steps, "Stresses", lambda da: da * sf)
    resp_steps = maybe_update(resp_steps, "StressesAtNodes", lambda da: da * sf)

    return resp_steps


def _get_shell_resp_one_step(ele_tags, dtype):
    sec_forces, sec_defos = [], []
    stresses, strains = [], []
    for _i, etag in enumerate(ele_tags):
        etag = int(etag)
        forces = ops.eleResponse(etag, "stresses")
        defos = ops.eleResponse(etag, "strains")
        sec_forces.append(_reorder_by_ele_type(etag, np.reshape(forces, (-1, 8))))
        sec_defos.append(_reorder_by_ele_type(etag, np.reshape(defos, (-1, 8))))
        # stress and strains
        num_sec = int(len(forces) / 8)
        sec_stress, sec_strain = [], []
        for j in range(num_sec):
            for k in range(100000000000000000):  # ugly but useful, loop for fiber layers
                stress = ops.eleResponse(etag, "Material", f"{j + 1}", "fiber", f"{k + 1}", "stresses")
                strain = ops.eleResponse(etag, "Material", f"{j + 1}", "fiber", f"{k + 1}", "strains")
                if len(stress) == 0 or len(strain) == 0:
                    break
                sec_stress.extend(stress)
                sec_strain.extend(strain)
        if len(sec_stress) == 0:  # elastic section response
            sec_stress, sec_strain = _get_elastic_section_stress(etag, sec_forces[-1])
        if len(sec_stress) == 0:  # still no stress/strain
            sec_stress = np.full((len(sec_forces[-1]), 1, 5), 0.0)
        if len(sec_strain) == 0:
            sec_strain = np.full((len(sec_forces[-1]), 1, 5), 0.0)
        sec_stress = np.reshape(sec_stress, (num_sec, -1, 5))
        sec_strain = np.reshape(sec_strain, (num_sec, -1, 5))
        stresses.append(_reorder_by_ele_type(etag, sec_stress))
        strains.append(_reorder_by_ele_type(etag, sec_strain))
    sec_forces = expand_to_uniform_array(sec_forces, dtype=dtype["float"])
    sec_defos = expand_to_uniform_array(sec_defos, dtype=dtype["float"])
    stresses = expand_to_uniform_array(stresses, dtype=dtype["float"])
    strains = expand_to_uniform_array(strains, dtype=dtype["float"])
    return sec_forces, sec_defos, stresses, strains


def _reorder_by_ele_type(etag, resp):
    ele_class_tag = ops.getEleClassTags(etag)[0]
    if ele_class_tag == 54 and len(resp) == 9:  # "ShellMITC9", 9 gps
        idx = [0, 2, 4, 6, 1, 3, 5, 7, 8]
    else:
        return resp
    return np.array([resp[i] for i in idx])


gp2node_type = {3: "tri", 6: "tri", 4: "quad", 8: "quad", 9: "quad"}


# Get nodal stresses and strains from the Gauss points of elements.
def _get_nodal_resp(ele_tags, ele_gp_resp, method, dtype):
    node_resp = defaultdict(list)
    for etag, gp_resp in zip(ele_tags, ele_gp_resp):
        etag = int(etag)
        ntags = ops.eleNodes(etag)
        gp_resp = drop_all_nan_rows(gp_resp)  # drop rows where all values are NaN
        if len(gp_resp) == 0:
            continue
        gp2node_func = get_shell_gp2node_func(ele_type=gp2node_type[len(ntags)], n=len(ntags), gp=len(gp_resp))
        if gp2node_func:
            resp = gp2node_func(method=method, gp_resp=gp_resp)
        else:
            resp = np.zeros((len(ntags), *gp_resp.shape[1:]), dtype=dtype["float"])
        for i, ntag in enumerate(ntags):
            node_resp[ntag].append(resp[i])
    # node_resp = dict(sorted(node_resp.items()))
    node_avg = {}

    for nid, vals in node_resp.items():
        arr = np.stack(vals, axis=0)  # shape: (k, m), k=num_samples, m=DOFs
        node_avg[nid] = np.nanmean(arr, axis=0)  # mean value
    node_avg = np.array(list(node_avg.values()), dtype=dtype["float"])
    node_tags = list(node_resp.keys())
    return node_avg, node_tags


def drop_all_nan_rows(arr: np.ndarray) -> np.ndarray:
    axis_to_check = tuple(range(1, arr.ndim))
    mask = ~np.isnan(arr).all(axis=axis_to_check)
    return arr[mask]


def _get_elastic_section_stress(eletag, sec_forces):
    with suppress_ops_print():
        E = _get_param_value(eletag, "E")
        nu = _get_param_value(eletag, "nu")
        h = _get_param_value(eletag, "h")
        # Ep_mod = _get_param_value(eletag, "Ep_mod")
        # rho = _get_param_value(eletag, "rho")
    sigmas, epses = [], []
    if E > 0 and nu >= 0 and h > 0:
        G = 0.5 * E / (1.0 + nu)
        xs = np.linspace(-h / 2, h / 2, 5)
        w = 12 / (h * h * h)
        for f11, f22, f12, m11, m22, m12, v13, v23 in sec_forces:
            sigma11 = f11 / h - w * m11 * xs
            sigma22 = f22 / h - w * m22 * xs
            sigma12 = f12 / h - w * m12 * xs
            sigma13 = v13 / h + np.zeros_like(xs)
            sigma23 = v23 / h + np.zeros_like(xs)
            eps11 = sigma11 / E
            eps22 = sigma22 / E
            eps12 = sigma12 / G
            eps13 = sigma13 / G
            eps23 = sigma23 / G
            sigma = np.array([sigma11, sigma22, sigma12, sigma23, sigma13]).T
            eps = np.array([eps11, eps22, eps12, eps23, eps13]).T
            sigmas.append(sigma)
            epses.append(eps)
        sigmas = np.array(sigmas)
        epses = np.array(epses)
    return sigmas, epses


def _get_param_value(eletag, param_name):
    paramTag = 1
    paramTags = ops.getParamTags()
    if len(paramTags) > 0:
        paramTag = max(paramTags) + 1
    ops.parameter(paramTag, "element", eletag, param_name)
    value = ops.getParamValue(paramTag)
    ops.remove("parameter", paramTag)
    return value
