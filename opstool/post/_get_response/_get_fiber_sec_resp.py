import warnings
from typing import ClassVar, Optional, Union

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from opstool.utils import OPS_ELE_TAGS

from ._response_base import ResponseBase, _expand_to_uniform_array

ELE_CLASS_TAGS = OPS_ELE_TAGS.Truss + OPS_ELE_TAGS.Beam + OPS_ELE_TAGS.Link


class FiberSecData:
    ELE_SEC_KEYS: ClassVar[dict] = {}  # key: ele_tag, value: sec_num
    SAVE_ALL: ClassVar[bool] = False
    FUTURE_ELE_TAGS: ClassVar[list] = []
    CURRENT_ELE_TAGS: ClassVar[list] = []

    @classmethod
    def add_data(cls, ele_tags=None):
        if ele_tags is None:
            return None
        elif isinstance(ele_tags, str) and ele_tags.lower() == "all":
            ele_tags = ops.getEleTags()
            cls.CURRENT_ELE_TAGS = ele_tags
            cls.SAVE_ALL = True
        else:
            ele_tags = np.atleast_1d(ele_tags)
            all_ele_tags = ops.getEleTags()
            if not set(ele_tags).issubset(set(all_ele_tags)):
                diffs = list(set(ele_tags) - set(all_ele_tags))
                cls.FUTURE_ELE_TAGS.extend(diffs)
                cls.CURRENT_ELE_TAGS = list(set(ele_tags) - set(diffs))
            else:
                cls.CURRENT_ELE_TAGS = ele_tags

        for ele_tag in cls.CURRENT_ELE_TAGS:
            ele_tag = int(ele_tag)
            class_tag = ops.getEleClassTags(ele_tag)
            if isinstance(class_tag, (list, tuple)):
                class_tag = class_tag[0]
            if class_tag in ELE_CLASS_TAGS:
                sec_locs = _get_section_locs(ele_tag)
                if len(sec_locs) > 0:
                    cls.ELE_SEC_KEYS[ele_tag] = len(sec_locs)

    @classmethod
    def update_data(cls):
        current_ele_tags = ops.getEleTags()
        set_original = set(cls.ELE_SEC_KEYS.keys())
        set_updated = set(current_ele_tags)
        added = set_updated - set_original
        removed = set_original - set_updated
        for ele_tag in removed:
            cls.ELE_SEC_KEYS.pop(ele_tag)
        added_ele_secs = {}
        for ele_tag in added:
            ele_tag = int(ele_tag)
            if ele_tag in cls.FUTURE_ELE_TAGS or cls.SAVE_ALL:
                class_tag = ops.getEleClassTags(ele_tag)
                if isinstance(class_tag, (list, tuple)):
                    class_tag = class_tag[0]
                if class_tag in ELE_CLASS_TAGS:
                    sec_locs = _get_section_locs(ele_tag)
                    if len(sec_locs) > 0:
                        cls.ELE_SEC_KEYS[ele_tag] = len(sec_locs)
                        added_ele_secs[ele_tag] = len(sec_locs)
        return added_ele_secs

    @classmethod
    def get_eletags(cls):
        return list(cls.ELE_SEC_KEYS.keys())

    @classmethod
    def get_ele_sec_keys(cls):
        return cls.ELE_SEC_KEYS

    @classmethod
    def get_ele_sec_history(cls):
        return cls.ELE_SEC_HISTORY


def _set_fiber_sec_data(fiber_ele_tags: Optional[Union[str, list, tuple]] = None):
    FiberSecData.add_data(fiber_ele_tags)


class FiberSecRespStepData(ResponseBase):
    def __init__(
        self,
        fiber_ele_tags: Optional[Union[str, list, tuple]] = None,
        dtype: Optional[dict] = None,
        model_update: bool = False,
    ):
        _set_fiber_sec_data(fiber_ele_tags)
        self.ELE_SEC_KEYS = FiberSecData.get_ele_sec_keys().copy()
        self.resp_names = ["Stresses", "Strains", "secForce", "secDefo"]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.fiber_geo_data_list = []
        self.fiber_geo_data = None
        self.times = []
        self.step_track = 0
        self.model_update = model_update

        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.secPoints = None
        self.fiberPoints = None
        self.DOFs = ["P", "Mz", "My", "T"]

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.add_data_one_step()
        self.step_track = 0
        self.times = [0.0]

        if self.ELE_SEC_KEYS != {}:
            self.fiber_geo_data_list.append(self._get_fiber_geo_data(self.ELE_SEC_KEYS))

    def reset(self):
        self.initialize()

    def add_data_one_step(self):
        if self.model_update:
            added_ele_sec = FiberSecData.update_data()
            ELE_SEC_KEYS = FiberSecData.get_ele_sec_keys()
            stress, strain, defo, force = _get_fiber_sec_resp(ELE_SEC_KEYS, dtype=self.dtype)
            data_vars = {
                "Stresses": (["eleTags", "secPoints", "fiberPoints"], stress),
                "Strains": (["eleTags", "secPoints", "fiberPoints"], strain),
                "secForce": (["eleTags", "secPoints", "DOFs"], force),
                "secDefo": (["eleTags", "secPoints", "DOFs"], defo),
            }

            # can have different dimensions and coordinates
            secPoints = np.arange(stress.shape[1]) + 1
            fiberPoints = np.arange(stress.shape[2]) + 1
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "eleTags": list(ELE_SEC_KEYS.keys()),
                    "secPoints": secPoints,
                    "fiberPoints": fiberPoints,
                    "DOFs": self.DOFs,
                },
            )
            self.resp_steps_list.append(ds)
            if added_ele_sec != {}:
                self.fiber_geo_data_list.append(self._get_fiber_geo_data(added_ele_sec))
        else:
            stress, strain, defo, force = _get_fiber_sec_resp(self.ELE_SEC_KEYS, dtype=self.dtype)
            self.resp_steps_dict["Stresses"].append(stress)
            self.resp_steps_dict["Strains"].append(strain)
            self.resp_steps_dict["secForce"].append(force)
            self.resp_steps_dict["secDefo"].append(defo)

            if self.secPoints is None:
                self.secPoints = np.arange(stress.shape[1]) + 1
                self.fiberPoints = np.arange(stress.shape[2]) + 1

        self.step_track += 1
        self.times.append(ops.getTime())

    def _get_fiber_geo_data(self, ELE_SEC_KEYS):
        all_ys, all_zs, all_mats, all_areas = [], [], [], []
        for ele_tag, sec_num in ELE_SEC_KEYS.items():
            ele_tag = int(ele_tag)
            sec_num = int(sec_num)
            ys, zs, areas, mats = [], [], [], []
            for i in range(sec_num):
                fiber_data = _get_fiber_sec_data(ele_tag, i + 1)
                ys.append(fiber_data[:, 0])
                zs.append(fiber_data[:, 1])
                areas.append(fiber_data[:, 2])
                if fiber_data.shape[1] == 6:
                    mats.append(fiber_data[:, 3])
                else:
                    mats.append(np.full(fiber_data.shape[0], np.nan))
            all_ys.append(_expand_to_uniform_array(ys))
            all_zs.append(_expand_to_uniform_array(zs))
            all_areas.append(_expand_to_uniform_array(areas))
            all_mats.append(_expand_to_uniform_array(mats))
        all_ys = _expand_to_uniform_array(all_ys)
        all_zs = _expand_to_uniform_array(all_zs)
        all_areas = _expand_to_uniform_array(all_areas)
        all_mats = _expand_to_uniform_array(all_mats)
        sec_points = np.arange(all_ys.shape[1]) + 1
        fiber_points = np.arange(all_ys.shape[2]) + 1
        data_vars_geo = {
            "ys": (["eleTags", "secPoints", "fiberPoints"], all_ys),
            "zs": (["eleTags", "secPoints", "fiberPoints"], all_zs),
            "areas": (["eleTags", "secPoints", "fiberPoints"], all_areas),
            "matTags": (["eleTags", "secPoints", "fiberPoints"], all_mats),
        }
        ds_geo = xr.Dataset(
            data_vars=data_vars_geo,
            coords={
                "eleTags": list(ELE_SEC_KEYS.keys()),
                "secPoints": sec_points,
                "fiberPoints": fiber_points,
            },
        )
        return ds_geo

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
            self.fiber_geo_data = xr.concat(self.fiber_geo_data_list, dim="eleTags", join="outer")
        else:
            data_vars = {
                "Stresses": (("time", "eleTags", "secPoints", "fiberPoints"), self.resp_steps_dict["Stresses"]),
                "Strains": (("time", "eleTags", "secPoints", "fiberPoints"), self.resp_steps_dict["Strains"]),
                "secDefo": (("time", "eleTags", "secPoints", "DOFs"), self.resp_steps_dict["secDefo"]),
                "secForce": (("time", "eleTags", "secPoints", "DOFs"), self.resp_steps_dict["secForce"]),
            }
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": list(self.ELE_SEC_KEYS.keys()),
                    "secPoints": self.secPoints,
                    "fiberPoints": self.fiberPoints,
                    "DOFs": self.DOFs,
                },
            )
            self.fiber_geo_data = self.fiber_geo_data_list[0]

        for key, values in self.fiber_geo_data.data_vars.items():
            self.resp_steps[key] = values

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
        dt["/FiberSectionResponses"] = resp_steps
        return dt

    @staticmethod
    def read_datatree(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/FiberSectionResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = FiberSecRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        moment_factor = unit_factors["moment"]
        curvature_factor = unit_factors["curvature"]
        disp_factor = unit_factors["disp"]
        stress_factor = unit_factors["stress"]

        # ---------------------------------------------------------
        resp_steps["Stresses"] *= stress_factor
        # ---------------------------------------------------------
        resp_steps["secForce"].loc[{"DOFs": ["P"]}] *= force_factor
        resp_steps["secForce"].loc[{"DOFs": ["Mz", "My", "T"]}] *= moment_factor
        resp_steps["secDefo"].loc[{"DOFs": ["Mz", "My", "T"]}] *= curvature_factor
        # --------------------------------------------------------
        resp_steps["ys"] *= disp_factor
        resp_steps["zs"] *= disp_factor
        resp_steps["areas"] *= disp_factor**2

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = FiberSecRespStepData.read_datatree(dt, unit_factors=unit_factors)
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


def _get_fiber_sec_resp(ele_secs: dict, dtype: dict):
    """Get the fiber section responses one step.

    Parameters
    -----------
    ele_secs: Union[list, tuple]
        [(ele_tag1, sec_tag1), (ele_tag1, sec_tag2), …, (ele_tag2, sec_tag1), …]
    """
    all_stress = []
    all_strains = []
    for ele_tag, sec_num in ele_secs.items():
        ele_tag = int(ele_tag)
        sec_num = int(sec_num)
        stress, strain, defo, force = [], [], [], []
        for i in range(sec_num):
            fiber_data = _get_fiber_sec_data(ele_tag, i + 1, dtype=dtype)
            stress.append(fiber_data[:, -2])
            strain.append(fiber_data[:, -1])
        all_stress.append(_expand_to_uniform_array(stress))
        all_strains.append(_expand_to_uniform_array(strain))
    all_stress = _expand_to_uniform_array(all_stress, dtype=dtype["float"])
    all_strains = _expand_to_uniform_array(all_strains, dtype=dtype["float"])

    if len(all_stress) == 0:
        all_stress = np.array([[[np.nan]]], dtype=dtype["float"])
    if len(all_strains) == 0:
        all_strains = np.array([[[np.nan]]], dtype=dtype["float"])
    # -----------------------------------------------------------------------
    all_defo = []
    all_force = []
    for ele_tag, sec_num in ele_secs.items():
        ele_tag = int(ele_tag)
        sec_num = int(sec_num)
        defo, force = [], []
        for i in range(sec_num):
            defo_forces = ops.eleResponse(ele_tag, "section", f"{i + 1}", "forceAndDeformation")
            if len(defo_forces) == 4:
                defo_forces = [
                    defo_forces[0],  # epsilon
                    defo_forces[1],  # kappaz
                    0.0,  # kappay
                    0.0,  # theta
                    defo_forces[2],  # P
                    defo_forces[3],  # Mz
                    0.0,  # My
                    0.0,  # T
                ]
            elif len(defo_forces) == 0:
                defo_forces = [0.0] * 8
            defo.append(defo_forces[:4])
            force.append(defo_forces[4:])
        all_defo.append(np.array(defo))
        all_force.append(np.array(force))
    all_defo = _expand_to_uniform_array(all_defo, dtype=dtype["float"])
    all_force = _expand_to_uniform_array(all_force, dtype=dtype["float"])
    if len(all_defo) == 0:
        all_defo = np.array([[[np.nan] * 4]], dtype=dtype["float"])
    if len(all_force) == 0:
        all_force = np.array([[[np.nan] * 4]], dtype=dtype["float"])
    return all_stress, all_strains, all_defo, all_force


def _get_fiber_sec_data(ele_tag: int, sec_num: int = 1, dtype: Optional[dict] = None):
    """Get the fiber sec data for a beam element.

    Parameters
    ----------
    ele_tag: int
        The element tag to which the sec is to be displayed.
    sec_num: int
        Which integration point sec is displayed, tag from 1 from segment i to j.

    Returns
    -------
    fiber_data: ArrayLike
    """
    if dtype is None:
        dtype = {"float": np.float32, "int": np.int32}
    # Extract fiber data using eleResponse() command
    sec_loc = _get_section_locs(ele_tag)
    if len(sec_loc) == 0:
        warnings.warn(f"eleTag {ele_tag} have no fiber sec!", stacklevel=2)
        return np.array([[np.nan] * 6])
    if sec_num > len(sec_loc):
        warnings.warn(
            f"Section number {sec_num} larger than max number {len(sec_loc)} of elemeng with tag {ele_tag}!"
            f"Section number {sec_num} set to {len(sec_loc)}!",
            stacklevel=2,
        )
        sec_num = len(sec_loc)
    ele_tag = int(ele_tag)
    # ------------------------------------------------------------------
    fiber_data = ops.eleResponse(ele_tag, "section", f"{sec_num}", "fiberData2")
    if len(fiber_data) == 0:
        fiber_data = ops.eleResponse(ele_tag, "section", "fiberData2")
    if len(fiber_data) > 0:
        # From column 1 to 6: "yCoord", "zCoord", "area", 'mat', "stress", "strain"
        fiber_data = np.reshape(fiber_data, (-1, 6))  # to six columns
        return fiber_data.astype(dtype["float"])
    # ------------------------------------------------------------------
    fiber_data = ops.eleResponse(ele_tag, "section", f"{sec_num}", "fiberData")
    if len(fiber_data) == 0:
        fiber_data = ops.eleResponse(ele_tag, "section", "fiberData")
    if len(fiber_data) > 0:
        # From column 1 to 5: "yCoord", "zCoord", "area", "stress", "strain"
        fiber_data = np.reshape(fiber_data, (-1, 5))  # to five columns
        return fiber_data.astype(dtype["float"])
    # ------------------------------------------------------------------
    return np.array([[np.nan] * 6])


def _get_section_locs(eletag):
    sec_locs = ops.sectionLocation(eletag)
    if not sec_locs:
        num_secs = 0
        for i in range(100000):
            output = ops.eleResponse(eletag, "section", f"{i + 1}", "forces")
            if (not output) and i == 0:
                output = ops.eleResponse(eletag, "section", "forces")
                num_secs = 1 if output else 0
                break
            elif not output:
                break
            num_secs += 1
        if num_secs == 0:
            sec_locs = np.array([])
        elif num_secs == 1:
            sec_locs = np.array([0.5])
        else:
            sec_locs = np.linspace(0, 1, num_secs)
    return sec_locs
