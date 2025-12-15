from collections import defaultdict
from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import get_gp2node_func
from ._response_base import ResponseBase, _expand_to_uniform_array


class BrickRespStepData(ResponseBase):
    def __init__(
        self,
        ele_tags=None,
        compute_measures: bool = True,
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

        self.compute_measures = compute_measures
        self.compute_nodal_resp = compute_nodal_resp
        self.nodal_resp_method = compute_nodal_resp
        self.node_tags = None
        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)
        self.include_pore_pressure = bool(compute_nodal_resp)
        if compute_measures in [True, "All", "all", "ALL"]:
            self.measures = {"principal": [], "von_mises": [], "octahedral": [], "tau_max": []}
        elif isinstance(compute_measures, dict):
            self.measures = compute_measures
        else:
            self.measures = {}

        self.attrs = {
            "sigma11, sigma22, sigma33": "Normal stress (strain) along x, y, z.",
            "sigma12, sigma23, sigma13": "Shear stress (strain).",
            "para#i": "The additional output of stress, which is useful for some elements, such as * eta_r * for some u-p elements. "
            "eta_r--Ratio between the shear (deviatoric) stress and peak shear strength at the current confinement.",
            "p1, p2, p3": "Principal stresses, p3=0 for 2D plane stress condition, p3!=0 for 3D plane strain condition.",
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
        self.stressDOFs = ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"] + [
            f"para#{i + 1}" for i in range(100)
        ]
        self.strainDOFs = ["eps11", "eps22", "eps33", "eps12", "eps23", "eps13"] + [f"para#{i + 1}" for i in range(100)]

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

        if self.GaussPointsNo is None:
            self.GaussPointsNo = np.arange(stresses.shape[1]) + 1

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

        if stresses.shape[-1] >= 6:
            stress_measures, measureStressDOFs = _calculate_stresses_measures_4D(
                stresses.data, dtype=self.dtype, measures=self.measures
            )

            dims = ["time", "eleTags", "GaussPoints", "measures"]
            coords = {
                "time": stresses.coords["time"],
                "eleTags": stresses.coords["eleTags"],
                "GaussPoints": stresses.coords["GaussPoints"],
                "measures": measureStressDOFs,
            }

            self.resp_steps["StressMeasures"] = xr.DataArray(
                stress_measures,
                dims=dims,
                coords=coords,
                name="StressMeasures",
            )
            if self.compute_nodal_resp:
                node_stress_measures, measureStressDOFs = _calculate_stresses_measures_4D(
                    self.resp_steps["StressesAtNodes"].data, dtype=self.dtype, measures=self.measures
                )
                dims = ["time", "nodeTags", "measures"]
                coords = {
                    "time": stresses.coords["time"],
                    "nodeTags": self.resp_steps["StressesAtNodes"].coords["nodeTags"],
                    "measures": measureStressDOFs,
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
        dt["/SolidResponses"] = resp_steps
        return dt

    @staticmethod
    def read_datatree(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/SolidResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = BrickRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        stress_factor = unit_factors["stress"]

        resp_steps["Stresses"].loc[
            {"stressDOFs": ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"]}
        ] *= stress_factor

        if "StressMeasures" in resp_steps.data_vars:
            resp_steps["StressMeasures"] *= stress_factor
        if "StressMeasuresAtNodes" in resp_steps.data_vars:
            resp_steps["StressMeasuresAtNodes"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = BrickRespStepData.read_datatree(dt, unit_factors=unit_factors)
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


gp2node_type = {4: "tet", 10: "tet", 8: "brick", 20: "brick", 27: "brick"}


# Get nodal stresses and strains from the Gauss points of elements.
def _get_nodal_resp(ele_tags, ele_gp_resp, method, dtype):
    node_resp = defaultdict(list)
    for etag, gp_resp in zip(ele_tags, ele_gp_resp):
        etag = int(etag)
        ntags = ops.eleNodes(etag)
        gp_resp = gp_resp[~np.isnan(gp_resp).all(axis=1)]
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
def _get_gauss_resp(ele_tags, dtype: dict):
    all_stresses, all_strains = [], []
    for etag in ele_tags:
        etag = int(etag)
        integr_point_stress = []
        integr_point_strain = []
        for i in range(100000000):  # Ugly but useful
            # loop for integrPoint
            stress_, strain_ = _get_gp_resp_by_one(etag, i)
            if stress_ is None or strain_ is None:
                break
            integr_point_stress.append(stress_)
            integr_point_strain.append(strain_)
        # Call material response directly
        if len(integr_point_stress) == 0 or len(integr_point_strain) == 0:
            stress = ops.eleResponse(etag, "stresses")
            strain = ops.eleResponse(etag, "strains")
            if len(stress) > 0:
                integr_point_stress.append(stress)
            if len(strain) > 0:
                integr_point_strain.append(strain)
        # Finally, if void set to 0.0
        if len(integr_point_stress) == 0:
            integr_point_stress.append([np.nan])
        if len(integr_point_strain) == 0:
            integr_point_strain.append([np.nan])

        all_stresses.append(np.array(integr_point_stress))
        all_strains.append(np.array(integr_point_strain))
    stresses = _expand_to_uniform_array(all_stresses, dtype=dtype["float"])
    strains = _expand_to_uniform_array(all_strains, dtype=dtype["float"])
    return stresses, strains


def _get_gp_resp_by_one(etag, i):
    stress_ = ops.eleResponse(etag, "material", f"{i + 1}", "stresses")
    if len(stress_) == 0:
        stress_ = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stresses")
    strain_ = ops.eleResponse(etag, "material", f"{i + 1}", "strains")
    if len(strain_) == 0:
        strain_ = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strains")
    if len(stress_) == 0 or len(strain_) == 0:
        return None, None
    return stress_, strain_


def _calculate_stresses_measures_4D(stress_array, dtype, measures=None):
    p1, p2, p3 = _principal_stresses(stress_array)

    # output containers
    data = []
    dofs = []

    # definition of measure handlers
    handlers = {
        "principal": lambda: ([p1, p2, p3], ["p1", "p2", "p3"]),
        "von_mises": lambda: ([_von_mises_stress(stress_array)], ["sigma_vm"]),
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
            list(_sig_mohr_coulomb_c_phi(p1, p2, p3, *params)),
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


def _principal_stresses(stress_array):
    """
    Compute principal stresses and a representative in-plane angle.

    Parameters
    ----------
    stress_array : ndarray
        (..., 6) = [s11,s22,s33,s12,s23,s13].

    Returns
    -------
    p1, p2, p3, theta_deg : ndarrays
        p1 ≥ p2 ≥ p3, theta_deg = angle (deg) of major principal direction
        projected to x-y plane: atan2(vy, vx).
    """
    # Calculate principal stresses
    # Using the stress tensor to calculate eigenvalues
    stress_tensor = _assemble_stress_tensor_4D(stress_array)
    # Calculate principal stresses (eigenvalues)
    principal_stresses = np.linalg.eigvalsh(stress_tensor)  # Returns sorted eigenvalues
    p1 = principal_stresses[..., 2]  # Maximum principal stress
    p2 = principal_stresses[..., 1]  # Intermediate principal stress
    p3 = principal_stresses[..., 0]  # Minimum principal stress

    return p1, p2, p3


def _von_mises_stress(stress_array):
    sig11 = stress_array[..., 0]
    sig22 = stress_array[..., 1]
    sig33 = stress_array[..., 2]
    sig12 = stress_array[..., 3]
    sig23 = stress_array[..., 4]
    sig13 = stress_array[..., 5]

    # von Mises equivalent stress
    sigma_vm = np.sqrt(
        0.5 * ((sig11 - sig22) ** 2 + (sig22 - sig33) ** 2 + (sig33 - sig11) ** 2)
        + 3.0 * (sig12**2 + sig23**2 + sig13**2)
    )
    return sigma_vm


def _tau_max(p1, p3):
    return 0.5 * (p1 - p3)


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


def _assemble_stress_tensor_4D(stress_array):
    """
    Assemble a 4D stress array [time, eleTags, GaussPoints, 6]
    into a 5D stress tensor array [time, eleTags, GaussPoints, 3, 3].
    Handles NaNs safely (returns 0.0 where data is missing).

    Parameters:
        stress_array (np.ndarray): shape (time, eleTags, GaussPoints, 6)

    Returns:
        np.ndarray: shape (time, eleTags, GaussPoints, 3, 3)
    """
    if stress_array.ndim == 4:
        num_time, num_elements, num_gauss_points, _ = stress_array.shape
        stress_tensor = np.full((num_time, num_elements, num_gauss_points, 3, 3), np.nan)

        for t in range(num_time):
            for i in range(num_elements):
                for j in range(num_gauss_points):
                    sig11 = stress_array[t, i, j, 0]
                    sig22 = stress_array[t, i, j, 1]
                    sig33 = stress_array[t, i, j, 2]
                    tau12 = stress_array[t, i, j, 3]
                    tau23 = stress_array[t, i, j, 4]
                    tau13 = stress_array[t, i, j, 5]

                    if np.any(np.isnan([sig11, sig22, sig33, tau12, tau23, tau13])):
                        continue  # skip invalid tensor

                    stress_tensor[t, i, j, ...] = np.array([
                        [sig11, tau12, tau13],
                        [tau12, sig22, tau23],
                        [tau13, tau23, sig33],
                    ])
    elif stress_array.ndim == 3:
        num_time, num_nodes, _ = stress_array.shape
        stress_tensor = np.full((num_time, num_nodes, 3, 3), np.nan)
        for t in range(num_time):
            for i in range(num_nodes):
                sig11 = stress_array[t, i, 0]
                sig22 = stress_array[t, i, 1]
                sig33 = stress_array[t, i, 2]
                tau12 = stress_array[t, i, 3]
                tau23 = stress_array[t, i, 4]
                tau13 = stress_array[t, i, 5]

                if np.any(np.isnan([sig11, sig22, sig33, tau12, tau23, tau13])):
                    continue
                stress_tensor[t, i, ...] = np.array([
                    [sig11, tau12, tau13],
                    [tau12, sig22, tau23],
                    [tau13, tau23, sig33],
                ])
    else:
        raise ValueError(f"Invalid stress_array shape: {stress_array.shape}. Expected 3D or 4D array.")  # noqa: TRY003

    return np.nan_to_num(stress_tensor, nan=0.0)


def _get_nodal_pore_pressure(node_tags):
    pressure = []
    for ntag in node_tags:
        vel = ops.nodeVel(ntag)
        p = vel[3] if len(vel) == 4 else 0.0
        pressure.append(p)
    return np.array(pressure)
