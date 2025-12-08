from collections import defaultdict
from typing import Optional, Union

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import get_gp2node_func
from ._response_base import ResponseBase, _expand_to_uniform_array


class PlaneRespStepData(ResponseBase):
    def __init__(
        self,
        ele_tags=None,
        compute_measures: Union[bool, dict, str] = True,
        compute_nodal_resp: Optional[str] = None,
        model_update: bool = False,
        dtype: Optional[dict] = None,
    ):
        self.resp_names = [
            "Stresses",
            "Strains",
            "StressesAtNodes",
            "StressAtNodesErr",
            "StrainsAtNodes",
            "StrainsAtNodesErr",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []

        self.node_tags = None
        self.compute_measures = compute_measures
        self.compute_nodal_resp = compute_nodal_resp
        self.nodal_resp_method = compute_nodal_resp
        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        self.include_pore_pressure = bool(compute_nodal_resp)

        if compute_measures in [True, "All", "all", "ALL"]:
            self.measures = {"principal": [], "von_mises": [], "octahedral": [], "tau_max": []}
        elif isinstance(compute_measures, dict):
            self.measures = compute_measures
        else:
            self.measures = {}

        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.attrs = {
            "sigma11, sigma22, sigma12": "Normal stress and shear stress in the x-y plane.",
            "sigma33": "Out-of-plane normal stress.",
            "para#i": "The additional output of stress, which is useful for some elements, such as * eta_r * for some u-p elements. "
            "eta_r--Ratio between the shear (deviatoric) stress and peak shear strength at the current confinement.",
            "p1, p2, p3": "Principal stresses, p3=0 for 2D plane stress condition, p3!=0 for 3D plane strain condition.",
            "theta": "Angle (degrees) between x-axis and principal axis 1.",
            "sigma_vm": "Von Mises stress.",
            "tau_max": "Maximum shear stress, 0.5*(p1-p3).",
            "sigma_oct": "Octahedral normal stress, (p1+p2+p3)/3.",
            "tau_oct": "Octahedral shear stress, sqrt(2/3*J2).",
            "sigma_mohr_coulomb_sy_eq": "Mohr-Coulomb equivalent stress (using tensile and compressive strengths).",
            "sigma_mohr_coulomb_sy_intensity": "Mohr-Coulomb intensity (using tensile and compressive strengths).",
            "sigma_mohr_coulomb_c_phi_eq": "Mohr-Coulomb equivalent stress (using cohesion and friction angle).",
            "sigma_mohr_coulomb_c_phi_intensity": "Mohr-Coulomb intensity (using cohesion and friction angle).",
            "sigma_drucker_prager_sy_eq": "Drucker-Prager equivalent stress (using tensile and compressive strengths).",
            "sigma_drucker_prager_sy_intensity": "Drucker-Prager intensity (using tensile and compressive strengths).",
            "sigma_drucker_prager_c_phi_eq": "Drucker-Prager equivalent stress (using cohesion and friction angle).",
            "sigma_drucker_prager_c_phi_intensity": "Drucker-Prager intensity (using cohesion and friction angle).",
        }
        self.GaussPointsNo = None
        self.stressDOFs = ["sigma11", "sigma22", "sigma12", "sigma33"] + ["para#" + str(i + 1) for i in range(100)]
        self.strainDOFs = ["eps11", "eps22", "eps12"] + ["para#" + str(i + 1) for i in range(100)]
        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        if self.include_pore_pressure:
            self.resp_steps_dict["PorePressureAtNodes"] = []
        self.add_data_one_step(self.ele_tags)
        self.times = [0.0]
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self, ele_tags):
        stresses, strains = _get_gauss_resp(ele_tags, dtype=self.dtype)  # shape: (num_eles, num_gps, num_dofs)
        self.stressDOFs = self.stressDOFs[: stresses.shape[2]]
        self.strainDOFs = self.strainDOFs[: strains.shape[2]]

        if self.compute_nodal_resp:
            node_stress_avg, node_stress_rel_error, node_tags = _get_nodal_resp(
                ele_tags, stresses, method=self.nodal_resp_method, dtype=self.dtype
            )
            node_strain_avg, node_strain_rel_error, node_tags = _get_nodal_resp(
                ele_tags, strains, method=self.nodal_resp_method, dtype=self.dtype
            )
            self.node_tags = node_tags
            if len(node_tags) == 0:
                self.compute_nodal_resp = False
            if self.include_pore_pressure:
                pore_pressure = _get_nodal_pore_pressure(node_tags)
                # idx = [0, 1, 3] if stresses.shape[-1] >= 4 else [0, 1]
                # node_stress_avg[:, idx] -= pore_pressure[:, None]  # effective stress

        if self.GaussPointsNo is None:
            self.GaussPointsNo = np.arange(strains.shape[1]) + 1

        if self.model_update:
            data_vars = {}
            data_vars["Stresses"] = (["eleTags", "GaussPoints", "stressDOFs"], stresses)
            data_vars["Strains"] = (["eleTags", "GaussPoints", "strainDOFs"], strains)
            coords = {
                "eleTags": ele_tags,
                "GaussPoints": self.GaussPointsNo,
                "stressDOFs": self.stressDOFs,
                "strainDOFs": self.strainDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["StressesAtNodes"] = (["nodeTags", "stressDOFs"], node_stress_avg)
                data_vars["StrainsAtNodes"] = (["nodeTags", "strainDOFs"], node_strain_avg)
                data_vars["StressAtNodesErr"] = (["nodeTags", "stressDOFs"], node_stress_rel_error)
                data_vars["StrainsAtNodesErr"] = (["nodeTags", "strainDOFs"], node_strain_rel_error)
                coords["nodeTags"] = node_tags
                if self.include_pore_pressure:
                    data_vars["PorePressureAtNodes"] = (["nodeTags"], pore_pressure)

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)
            self.resp_steps_list.append(ds)
        else:
            self.resp_steps_dict["Stresses"].append(stresses)
            self.resp_steps_dict["Strains"].append(strains)
            if self.compute_nodal_resp:
                self.resp_steps_dict["StressesAtNodes"].append(node_stress_avg)
                self.resp_steps_dict["StrainsAtNodes"].append(node_strain_avg)
                self.resp_steps_dict["StressAtNodesErr"].append(node_stress_rel_error)
                self.resp_steps_dict["StrainsAtNodesErr"].append(node_strain_rel_error)
                if self.include_pore_pressure:
                    self.resp_steps_dict["PorePressureAtNodes"].append(pore_pressure)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["Stresses"] = (["time", "eleTags", "GaussPoints", "stressDOFs"], self.resp_steps_dict["Stresses"])
            data_vars["Strains"] = (["time", "eleTags", "GaussPoints", "strainDOFs"], self.resp_steps_dict["Strains"])
            coords = {
                "time": self.times,
                "eleTags": self.ele_tags,
                "GaussPoints": self.GaussPointsNo,
                "stressDOFs": self.stressDOFs,
                "strainDOFs": self.strainDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["StressesAtNodes"] = (
                    ["time", "nodeTags", "stressDOFs"],
                    self.resp_steps_dict["StressesAtNodes"],
                )
                data_vars["StrainsAtNodes"] = (
                    ["time", "nodeTags", "strainDOFs"],
                    self.resp_steps_dict["StrainsAtNodes"],
                )
                data_vars["StressAtNodesErr"] = (
                    ["time", "nodeTags", "stressDOFs"],
                    self.resp_steps_dict["StressAtNodesErr"],
                )
                data_vars["StrainsAtNodesErr"] = (
                    ["time", "nodeTags", "strainDOFs"],
                    self.resp_steps_dict["StrainsAtNodesErr"],
                )
                if self.include_pore_pressure:
                    data_vars["PorePressureAtNodes"] = (
                        ["time", "nodeTags"],
                        self.resp_steps_dict["PorePressureAtNodes"],
                    )
                coords["nodeTags"] = self.node_tags
            self.resp_steps = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)

        if (
            "PorePressureAtNodes" in self.resp_steps
            and np.abs(self.resp_steps["PorePressureAtNodes"].data).sum() < 1e-10
        ):
            self.resp_steps = self.resp_steps.drop_vars("PorePressureAtNodes")

        if self.compute_measures:
            self._compute_measures_()

    def _compute_measures_(self):
        stresses = self.resp_steps["Stresses"]

        if stresses.shape[-1] >= 3:
            stress_measures, measureDOFs = _calculate_stresses_measures(
                stresses.data, dtype=self.dtype, measures=self.measures
            )

            dims = ["time", "eleTags", "GaussPoints", "measures"]
            coords = {
                "time": stresses.coords["time"],
                "eleTags": stresses.coords["eleTags"],
                "GaussPoints": stresses.coords["GaussPoints"],
                "measures": measureDOFs,
            }

            self.resp_steps["StressMeasures"] = xr.DataArray(
                stress_measures,
                dims=dims,
                coords=coords,
                name="StressMeasures",
            )

            if self.compute_nodal_resp:
                # pore_pressure = self.resp_steps["PorePressureAtNodes"].data if self.include_pore_pressure else None
                node_stress_measures, measureDOFs = _calculate_stresses_measures(
                    self.resp_steps["StressesAtNodes"].data, dtype=self.dtype, measures=self.measures
                )
                dims = ["time", "nodeTags", "measures"]
                coords = {
                    "time": stresses.coords["time"],
                    "nodeTags": self.resp_steps["StressesAtNodes"].coords["nodeTags"],
                    "measures": measureDOFs,
                }
                self.resp_steps["StressMeasuresAtNodes"] = xr.DataArray(
                    node_stress_measures, dims=dims, coords=coords, name="StressMeasuresAtNodes"
                )

    def get_data(self):
        if self.resp_steps is None:
            self._to_xarray()
        return self.resp_steps

    def update_data(self, data):
        self.resp_steps = data

    def get_track(self):
        return self.step_track

    def add_to_datatree(self, dt: xr.DataTree):
        resp_steps = self.get_data()
        dt["/PlaneResponses"] = resp_steps
        return dt

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        stress_factor = unit_factors["stress"]

        resp_steps["Stresses"].loc[{"stressDOFs": ["sigma11", "sigma22", "sigma12"]}] *= stress_factor
        if "sigma33" in resp_steps["Stresses"].coords["stressDOFs"]:
            resp_steps["Stresses"].loc[{"stressDOFs": ["sigma33"]}] *= stress_factor
        if "StressMeasures" in resp_steps.data_vars:
            resp_steps["StressMeasures"] *= stress_factor
        if "StressMeasuresAtNodes" in resp_steps.data_vars:
            resp_steps["StressMeasuresAtNodes"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_datatree(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/PlaneResponses"].to_dataset()
        if unit_factors:
            resp_steps = PlaneRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = PlaneRespStepData.read_datatree(dt, unit_factors=unit_factors)
        if resp_type is None:
            if ele_tags is None:
                return ds
            else:
                return ds.sel(eleTags=ele_tags)
        else:
            if resp_type not in list(ds.keys()):
                raise ValueError(f"resp_type {resp_type} not found in {list(ds.keys())}")  # noqa: TRY003
            if ele_tags is not None:
                return ds[resp_type].sel(eleTags=ele_tags)
            else:
                return ds[resp_type]


gp2node_type = {3: "tri", 6: "tri", 4: "quad", 8: "quad", 9: "quad"}


# Get nodal stresses and strains from the Gauss points of elements.
def _get_nodal_resp(ele_tags, ele_gp_resp, method, dtype):
    # ele_gp_resp: shape (num_eles, num_gps, num_dofs)
    node_resp = defaultdict(list)
    for etag, gp_resp in zip(ele_tags, ele_gp_resp):
        etag = int(etag)
        ntags = ops.eleNodes(etag)
        gp_resp = gp_resp[~np.isnan(gp_resp).all(axis=1)]  #
        if len(gp_resp) == 0:
            continue
        gp2node_func = get_gp2node_func(ele_type=gp2node_type[len(ntags)], n=len(ntags), gp=len(gp_resp))
        if gp2node_func:
            resp = gp2node_func(method=method, gp_resp=gp_resp)
        else:
            resp = np.zeros((len(ntags), gp_resp.shape[1]), dtype=dtype["float"])
        for i, ntag in enumerate(ntags):
            node_resp[ntag].append(resp[i])
    # node_resp = dict(sorted(node_resp.items()))
    node_avg = {}
    # node_max = {}
    # node_min = {}
    node_ptp = {}  # Peak-to-peak: max - min
    # node_std = {}
    node_rel_error = {}

    for nid, vals in node_resp.items():
        arr = np.stack(vals, axis=0)  # shape: (k, m), k=num_samples, m=DOFs
        node_avg[nid] = np.nanmean(arr, axis=0)  # mean value
        # node_max[nid] = np.nanmax(arr, axis=0)  # maximum value
        # node_min[nid] = np.nanmin(arr, axis=0)  # minimum value
        node_ptp[nid] = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        # node_std[nid] = np.nanstd(arr, axis=0)  # standard deviation
        node_rel_error[nid] = node_ptp[nid] / (np.abs(node_avg[nid]) + 1e-8)  # avoid division by zero
        node_rel_error[nid][np.abs(node_avg[nid]) < 1e-8] = 0.0  # if avg is close to zero, set rel error to zero
    node_avg = np.array(list(node_avg.values()), dtype=dtype["float"])
    node_rel_error = np.array(list(node_rel_error.values()), dtype=dtype["float"])
    node_tags = list(node_resp.keys())
    return node_avg, node_rel_error, node_tags


#  Get Gauss point stresses and strains for all elements in the model.
def _get_gauss_resp(ele_tags, dtype):
    """Collect Gauss point stresses and strains for each element."""
    all_stresses, all_strains = [], []

    for etag in map(int, ele_tags):
        stresses = np.array(_collect_element_stresses(etag))
        strains = np.array(_collect_element_strains(etag))
        stresses, strains = _reorder_by_element_type(etag, stresses, strains)
        all_stresses.append(_reshape_stress(stresses))
        all_strains.append(strains)

    return (
        _expand_to_uniform_array(all_stresses, dtype=dtype["float"]),
        _expand_to_uniform_array(all_strains, dtype=dtype["float"]),
    )


def _collect_element_stresses(etag):
    stresses = []
    # stresses
    for i in range(100000000):
        s = _try_fetch(etag, i + 1, "stresses")
        if not s:
            break
        stresses.append(s)
    if len(stresses) == 0:
        s = ops.eleResponse(etag, "stresses")
        if s:
            stresses.append(s)
    if len(stresses) == 0:
        stresses.append([np.nan])
    return stresses


def _collect_element_strains(etag):
    strains = []
    for i in range(100000000):
        e = _try_fetch(etag, i + 1, "strains")
        if not e:
            break
        strains.append(e)
    if len(strains) == 0:
        e = ops.eleResponse(etag, "strains")
        if e:
            strains.append(e)
    if len(strains) == 0:
        strains.append([np.nan])
    return strains


def _try_fetch(etag, idx, key):
    """Try multiple ops.eleResponse paths to fetch value."""
    for prefix in ["material", "integrPoint"]:
        val = ops.eleResponse(etag, prefix, str(idx), key)
        if val:
            return val
    return []


def _reorder_by_element_type(etag, stress, strain):
    ele_class_tag = ops.getEleClassTags(etag)[0]
    if ele_class_tag == 209 and len(stress) == 3:  # SixNodeTri, 3 gps
        idx = [2, 0, 1]
    elif ele_class_tag == 61 and len(stress) == 9:  # NineNodeMixedQuad, 9 gps
        idx = [0, 6, 8, 2, 3, 7, 5, 1, 4]
    else:
        return stress, strain
    return np.array([stress[i] for i in idx]), np.array([strain[i] for i in idx])


def _reshape_stress(stress):
    stress = np.array(stress)
    if stress.ndim == 1:
        stress = np.reshape(stress, (-1, 1))
    num_stress = stress.shape[1]
    if num_stress >= 4:
        # sigma_xx, sigma_yy, sigma_zz, sigma_xy, ηr, where ηr is the ratio between the shear (deviatoric) stress and peak
        # shear strength at the current confinement (0<=ηr<=1.0).
        # # sigma_xx, sigma_yy, sigma_zz, sigma_xy if num_stress ==4
        stress[:, [2, 3]] = stress[:, [3, 2]]
    return stress


def _calculate_stresses_measures(stress_array, dtype, measures):
    # unpack stress components (plane stress default)
    sig11 = stress_array[..., 0]
    sig22 = stress_array[..., 1]
    sig12 = stress_array[..., 2]
    sig33 = stress_array[..., 3] if stress_array.shape[-1] >= 4 else np.zeros_like(sig11)

    # principal stresses
    p1, p2, p3, theta = _compute_principal(sig11, sig22, sig12, sig33)

    # output containers
    data = []
    dofs = []

    # definition of measure handlers
    handlers = {
        "principal": lambda: ([p1, p2, p3, theta], ["p1", "p2", "p3", "theta"]),
        "von_mises": lambda: ([_von_mises(sig11, sig22, sig33, sig12)], ["sigma_vm"]),
        "tau_max": lambda: ([_tau_max(p1, p3)], ["tau_max"]),
        "octahedral": lambda: (
            list(_octahedral_stress(p1, p2, p3)),
            ["sigma_oct", "tau_oct"],
        ),
        "mohr_coulomb_sy": lambda params: (
            list(_sig_mohr_coulomb_sy(p1, p2, p3, **params)),
            ["sigma_mohr_coulomb_sy_eq", "sigma_mohr_coulomb_sy_intensity"],
        ),
        "mohr_coulomb_c_phi": lambda params: (
            list(_sig_mohr_coulomb_c_phi(p1, p2, p3, **params)),
            ["sigma_mohr_coulomb_c_phi_eq", "sigma_mohr_coulomb_c_phi_intensity"],
        ),
        "drucker_prager_sy": lambda params: (
            list(_sig_drucker_prager_sy(p1, p2, p3, **params)),
            ["sigma_drucker_prager_sy_eq", "sigma_drucker_prager_sy_intensity"],
        ),
        "drucker_prager_c_phi": lambda params: (
            list(_sig_drucker_prager_c_phi(p1, p2, p3, **params)),
            ["sigma_drucker_prager_c_phi_eq", "sigma_drucker_prager_c_phi_intensity"],
        ),
    }

    # iterate user-requested measures
    for measure, params in measures.items():
        mkey = measure.lower()

        if mkey not in handlers:
            raise ValueError(f"Measure '{measure}' not recognized.")  # noqa: TRY003

        handler = handlers[mkey]

        # check if handler requires parameters
        if isinstance(params, dict) and len(params) > 0:
            vals, names = handler(params)
        else:
            vals, names = handler()

        data.extend(vals)
        dofs.extend(names)

    stress_measures = np.stack(data, axis=-1).astype(dtype["float"])
    return stress_measures, dofs


def _compute_principal(sig11, sig22, sig12, sig33):
    sig_avg = (sig11 + sig22) / 2.0
    radius = np.sqrt(((sig11 - sig22) / 2.0) ** 2 + sig12**2)
    p1_2d = sig_avg + radius
    p2_2d = sig_avg - radius
    # principal direction
    theta = np.zeros_like(sig11)
    mask = np.abs(sig11 - sig22) > 1e-10
    theta[mask] = 0.5 * np.arctan2(2.0 * sig12[mask], sig11[mask] - sig22[mask])
    # special case: sig11 ≈ sig22 but non-zero shear
    mask_equal = (~mask) & (np.abs(sig12) > 1e-10)
    theta[mask_equal] = 0.25 * np.pi * np.sign(sig12[mask_equal])
    theta_deg = np.degrees(theta)
    # 3D principal stresses
    p_array = np.stack([p1_2d, p2_2d, sig33], axis=-1)
    # sort along the last axis: [..., 0] = min, [..., 2] = max
    p_sorted = np.sort(p_array, axis=-1)
    p3_3d = p_sorted[..., 0]
    p2_3d = p_sorted[..., 1]
    p1_3d = p_sorted[..., 2]
    return p1_3d, p2_3d, p3_3d, theta_deg


def _tau_max(p1, p3):
    return 0.5 * (p1 - p3)


def _von_mises(sig11, sig22, sig33, sig12):
    sig_vm = np.sqrt(((sig11 - sig22) ** 2 + (sig22 - sig33) ** 2 + (sig33 - sig11) ** 2) / 2.0 + 3.0 * sig12**2)
    return sig_vm


def _octahedral_stress(p1, p2, p3):
    I1 = p1 + p2 + p3
    J2 = 1 / 6.0 * ((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2)
    sig_oct = I1 / 3.0
    tau_oct = np.sqrt(2.0 / 3.0 * J2)
    return sig_oct, tau_oct


def _sig_mohr_coulomb_sy(p1, p2, p3, syc, syt):
    m = syc / (syt + 1e-10)
    K = (m - 1.0) / (m + 1.0)
    t12 = np.abs(p1 - p2) + K * (p1 + p2)
    t13 = np.abs(p1 - p3) + K * (p1 + p3)
    t23 = np.abs(p2 - p3) + K * (p2 + p3)
    tmax = np.maximum(np.maximum(t12, t13), t23)
    mc_eq = 0.5 * (m + 1.0) * tmax
    return mc_eq, syc


def _sig_mohr_coulomb_c_phi(p1, p2, p3, c, phi):
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)

    def pair_eq(si, sj):
        tau_ij = 0.5 * np.abs(si - sj)
        sig_ij = 0.5 * (si + sj)
        return tau_ij / cos_phi - sig_ij * tan_phi

    eq12 = pair_eq(p1, p2)
    eq13 = pair_eq(p1, p3)
    eq23 = pair_eq(p2, p3)
    sigma_eq = np.maximum(np.maximum(eq12, eq13), eq23)
    intensity = c
    return sigma_eq, intensity


def _sig_drucker_prager_sy(p1, p2, p3, syc, syt):
    m = syc / (syt + 1e-10)
    I1 = p1 + p2 + p3
    diff_sq = (p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2
    q_part = np.sqrt(0.5 * diff_sq)

    sigma_eq = 0.5 * (m - 1.0) * I1 + 0.5 * (m + 1.0) * q_part

    return sigma_eq, syc


def _sig_drucker_prager_c_phi(p1, p2, p3, c, phi, kind):
    I1 = p1 + p2 + p3
    J2 = ((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2) / 6.0
    sqrtJ2 = np.sqrt(J2)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    if kind.lower() == "circumscribed":  # circumscribed
        A = 6.0 * c * cos_phi / (np.sqrt(3.0) * (3.0 - sin_phi))
        B = 2.0 * sin_phi / (np.sqrt(3.0) * (3.0 - sin_phi))
    elif kind.lower() == "middle":  # middle
        A = 6.0 * c * cos_phi / (np.sqrt(3.0) * (3.0 + sin_phi))
        B = 2.0 * sin_phi / (np.sqrt(3.0) * (3.0 + sin_phi))
    elif kind.lower() == "inscribed":  # inscribed
        A = 3.0 * c * cos_phi / np.sqrt(9.0 + 3.0 * sin_phi**2)
        B = sin_phi / np.sqrt(9.0 + 3.0 * sin_phi**2)
    else:
        raise ValueError("kind must be 'circumscribed', 'middle', or 'inscribed'.")  # noqa: TRY003
    sigma_eq = sqrtJ2 - B * I1  # equivalent stress
    sigma_y = A  # intensity
    return sigma_eq, sigma_y


def _get_nodal_pore_pressure(node_tags):
    pressure = []
    for ntag in node_tags:
        vel = ops.nodeVel(ntag)
        p = vel[2] if len(vel) == 3 else 0.0
        pressure.append(p)
    return np.array(pressure)
