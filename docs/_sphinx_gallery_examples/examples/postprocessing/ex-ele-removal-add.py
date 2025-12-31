r"""
Remove and Add Elements
======================================
Author: `@WJinger <https://github.com/WJinger>`_
"""

# %%
# Import some packages
# ------------------------
from collections import namedtuple
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.plotly as opsplt

# %%
# Set base variables
# --------------------

# Set unit
UNIT = opst.pre.UnitSystem(length="m", force="kn", time="sec")

# Set constant
g = 9.80665 * (UNIT.m / UNIT.sec**2)
Ubig = 1.0e10
Usmall = 1.0e-10

# Set data path
# data_path = "./OutData"
# os.makedirs(data_path, exist_ok=True)
# opst.post.set_odb_path(data_path)  # set opstool ODB path


# %%
# Section creator function
# -----------------------------
def secCreate(sec_name: str, secTag: int, matTag: int, w: float, h: float, info: bool = True) -> dict[str, Any]:
    """
    # Just for a simple rectangle section

    :param sec_name: section name
    :param secTag: section tag
    :param matTag: material tag
    :param w: width
    :param h: height
    :param info: if True, print section information

    :return: a dictionary with the section properties

    """

    # section outline
    sec_outline = [[0, 0], [w, 0], [w, h], [0, h]]
    inner_geo = opst.pre.section.create_polygon_patch(sec_outline)
    SEC = opst.pre.section.FiberSecMesh(sec_name=sec_name)
    SEC.add_patch_group({"inner": inner_geo})
    SEC.set_ops_mat_tag({"inner": matTag})
    SEC.set_mesh_size({"inner": (w + h) / 20.0})
    SEC.set_mesh_color({"inner": "lightblue"})
    SEC.mesh()

    # section properties
    SEC.centring()
    sec_props = SEC.get_frame_props(display_results=info)  # print section properties

    if info:
        SEC.view(fill=True, show_legend=True, aspect_ratio="equal")
        plt.show()

    # define section
    SEC.to_opspy_cmds(secTag=secTag, G=100 * UNIT.gpa)

    return sec_props


# %%
# Model creator function
# -----------------------------

# %%
# Values for return
RETURN_VAL = namedtuple(
    "RETURN_VAL",
    ["ctrl_node", "remove_ele", "add_ele", "top_link_ele", "base_link_ele"],
)
"""
Remove element:
    ops.remove('ele', RETURN_VAL.remove_ele)
Add element: 
    ops.element(*RETURN_VAL.add_ele)
"""


# %%
# Model creater function
def triangleStruc(top_free: bool = False, base_free: bool = False, info: bool = True) -> RETURN_VAL:
    """
    # Create a simple triangle structure

    :param top_free: if True, add zeroLength element to the top
    :param base_free: if True, add zeroLengthSection element to the base
    :param info: if True, print section element information

    :return a namedtuple with the following values:
    - :ctrl_node: the control node for static analysis
    - :remove_ele: the element to be removed
    - :add_ele: the element to be added
    - :top_link_ele: the top zeroLength element
    - :base_link_ele: the base zeroLengthSection element
    """

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Initialize
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # For vertical element
    transf_ver = 1
    ops.geomTransf("PDelta", transf_ver, *(-1, 0, 0))
    # For other element
    transf_other = 2
    ops.geomTransf("PDelta", transf_other, *(0, 0, 1))

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Steel material
    steelTag = 10
    ops.uniaxialMaterial("Steel02", steelTag, 400 * UNIT.mpa, 206 * UNIT.gpa, 0.01)

    # Elastic no tension material
    ENTTag = 20
    ops.uniaxialMaterial("ENT", ENTTag, 200 * UNIT.gpa)

    # Elastic material
    elasticTag = 30
    ops.uniaxialMaterial("Elastic", elasticTag, 1.0e6)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Define vertical element section
    secTag_1 = 1
    W_1 = 10.0 * UNIT.cm
    H_1 = 10.0 * UNIT.cm
    sec_props_1 = secCreate(sec_name="vertical", secTag=secTag_1, matTag=steelTag, w=W_1, h=H_1, info=info)

    # Define vertical base link element section
    secTag_2 = 2
    sec_props_2 = secCreate(
        sec_name="vertical base",
        secTag=secTag_2,
        matTag=ENTTag,
        w=W_1,
        h=H_1,
        info=info,
    )

    # Define other element section
    secTag_3 = 3
    W_2 = 2.0 * UNIT.cm
    H_2 = 2.0 * UNIT.cm
    sec_props_3 = secCreate(sec_name="oblique", secTag=secTag_3, matTag=steelTag, w=W_2, h=H_2, info=info)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Node Tags
    nodeTag_base = 1
    nodeTag_top = 2
    nodeTag_base_link = 3  # Base link node
    nodeTag_top_link = 4  # Top link node
    nodeTag_left_base = 5
    nodeTag_right_base = 6

    # Element Tags
    eleTag_main = 1
    eleTag_left = 2
    eleTag_right = 3
    eleTag_base_link = 4  # Base link element
    eleTag_top_link = 5  # Top link element

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Create nodes
    ops.node(nodeTag_base, 0.0, 0.0, 0.0)
    ops.node(nodeTag_top, 0.0, 0.0, 1.0 * UNIT.m)
    ops.node(nodeTag_left_base, 0.0, -1.0 * UNIT.m, 0.0)
    ops.node(nodeTag_right_base, 0.0, 1.0 * UNIT.m, 0.0)

    # Nodes fix
    ops.fix(nodeTag_base, 1, 1, 1, 1, 1, 1)
    ops.fix(nodeTag_left_base, 1, 1, 1, 1, 1, 1)
    ops.fix(nodeTag_right_base, 1, 1, 1, 1, 1, 1)

    # Integration
    npTag_1 = 1
    npTag_2 = 2
    ops.beamIntegration("Legendre", npTag_1, secTag_1, 5)
    ops.beamIntegration("Legendre", npTag_2, secTag_3, 5)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    """
    Note: 
        In order to check the local axes of zeroLength and zeroLengthSection, the length of zeroLength and zeroLengthSection elements is not zero.
        It causes opensees warning but does not affect the results.
    """

    if top_free and not base_free:
        print("CASE: zeroLength")

        ops.node(nodeTag_top_link, 0.0, 0.0, 1.2 * UNIT.m)  # For top zeroLength
        ops.element(
            "dispBeamColumn",
            eleTag_main,
            *(nodeTag_top, nodeTag_base),
            transf_ver,
            npTag_1,
        )  # Single vertical element
        ops.element(
            "dispBeamColumn",
            eleTag_left,
            *(nodeTag_top, nodeTag_left_base),
            transf_ver,
            npTag_2,
        )  # Left oblique element

        mats = (elasticTag, elasticTag, elasticTag, elasticTag, elasticTag, elasticTag)
        ops.element(
            "zeroLength",
            eleTag_top_link,
            *(nodeTag_top_link, nodeTag_top),
            "-mat",
            *mats,
            "-dir",
            *(1, 2, 3, 4, 5, 6),
            "-orient",
            *(0, 0, -1),
            *(0, -1, 0),
        )

        return_val = RETURN_VAL(
            ctrl_node=nodeTag_top_link,
            remove_ele=eleTag_left,
            add_ele=(
                "dispBeamColumn",
                eleTag_right,
                *(nodeTag_top, nodeTag_right_base),
                transf_ver,
                npTag_2,
            ),
            top_link_ele=eleTag_top_link,
            base_link_ele=None,
        )

    elif not top_free and base_free:
        print("CASE: zeroLengthSection")

        ops.node(nodeTag_base_link, 0.0, 0.0, 0.2 * UNIT.m)  # For base zeroLengthSection
        ops.element(
            "dispBeamColumn",
            eleTag_main,
            *(nodeTag_top, nodeTag_base_link),
            transf_ver,
            npTag_1,
        )  # Single vertical element
        ops.element(
            "dispBeamColumn",
            eleTag_left,
            *(nodeTag_top, nodeTag_left_base),
            transf_ver,
            npTag_2,
        )  # Left oblique element

        ops.equalDOF(*(nodeTag_base, nodeTag_base_link), *(1, 2, 3))
        ops.element(
            "zeroLengthSection",
            eleTag_base_link,
            *(nodeTag_base_link, nodeTag_base),
            secTag_2,
            "-orient",
            *(0, 0, -1),
            *(0, -1, 0),
        )

        return_val = RETURN_VAL(
            ctrl_node=nodeTag_top,
            remove_ele=eleTag_left,
            add_ele=(
                "dispBeamColumn",
                eleTag_right,
                *(nodeTag_top, nodeTag_right_base),
                transf_ver,
                npTag_2,
            ),
            top_link_ele=None,
            base_link_ele=eleTag_base_link,
        )

    elif top_free and base_free:
        print("CASE: zeroLength & zeroLengthSection")

        ops.node(nodeTag_top_link, 0.0, 0.0, 1.2 * UNIT.m)  # For top zeroLength
        ops.node(nodeTag_base_link, 0.0, 0.0, 0.2 * UNIT.m)  # For base zeroLengthSection
        ops.element(
            "dispBeamColumn",
            eleTag_main,
            *(nodeTag_top, nodeTag_base_link),
            transf_ver,
            npTag_1,
        )  # Single vertical element
        ops.element(
            "dispBeamColumn",
            eleTag_left,
            *(nodeTag_top, nodeTag_left_base),
            transf_ver,
            npTag_2,
        )  # Left oblique element

        mats = (elasticTag, elasticTag, elasticTag, elasticTag, elasticTag, elasticTag)
        ops.element(
            "zeroLength",
            eleTag_top_link,
            *(nodeTag_top_link, nodeTag_top),
            "-mat",
            *mats,
            "-dir",
            *(1, 2, 3, 4, 5, 6),
            "-orient",
            *(0, 0, -1),
            *(0, -1, 0),
        )

        ops.equalDOF(*(nodeTag_base, nodeTag_base_link), *(1, 2, 3))
        ops.element(
            "zeroLengthSection",
            eleTag_base_link,
            *(nodeTag_base_link, nodeTag_base),
            secTag_2,
            "-orient",
            *(0, 0, -1),
            *(0, -1, 0),
        )

        return_val = RETURN_VAL(
            ctrl_node=nodeTag_top_link,
            remove_ele=eleTag_left,
            add_ele=(
                "dispBeamColumn",
                eleTag_right,
                *(nodeTag_top, nodeTag_right_base),
                transf_ver,
                npTag_2,
            ),
            top_link_ele=eleTag_top_link,
            base_link_ele=eleTag_base_link,
        )

    else:
        print("CASE: None")

        ops.element(
            "dispBeamColumn",
            eleTag_main,
            *(nodeTag_top, nodeTag_base),
            transf_ver,
            npTag_1,
        )  # Single vertical element
        ops.element(
            "dispBeamColumn",
            eleTag_left,
            *(nodeTag_top, nodeTag_left_base),
            transf_ver,
            npTag_2,
        )  # Left oblique element

        return_val = RETURN_VAL(
            ctrl_node=nodeTag_top,
            remove_ele=eleTag_left,
            add_ele=(
                "dispBeamColumn",
                eleTag_right,
                *(nodeTag_top, nodeTag_right_base),
                transf_ver,
                npTag_2,
            ),
            top_link_ele=None,
            base_link_ele=None,
        )

    return return_val


# %%
# Analysis function
# -----------------------------


def analysisLib(
    targets: Union[list, tuple, np.ndarray],
    patternTag: int,
    ctrl_node: int,
    ODB: opst.post.CreateODB,
) -> tuple[np.ndarray, np.ndarray]:
    """
    # Static analysis function

    :param targets: Displacement path
    :param patternTag: Pattern tag
    :param ctrl_node: Control node tag
    :param ODB: CreateODB object

    :return: Displacement and force
    """

    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")

    analysis = opst.anlys.SmartAnalyze("Static")
    segs = analysis.static_split(targets=targets, maxStep=0.01 * UNIT.m)

    force_lambda: Union[list, float] = [0.0]
    node_disp: Union[list, float] = [0.0]
    for seg in segs:
        ok = analysis.StaticAnalyze(node=ctrl_node, dof=2, seg=seg)  # node tag 1, dof 2
        if ok < 0:
            raise RuntimeError("Analysis failed")  # noqa: TRY003

        # Fetch response
        ODB.fetch_response_step()
        force_lambda.append(ops.getLoadFactor(patternTag))
        node_disp.append(ops.nodeDisp(ctrl_node, 2))

    return np.array(node_disp), np.array(force_lambda)


# %%
# Check single column hysteretic curve
# -------------------------------------

# %%
# For Case 1, I just want to look at the hysteresis curve with only vertical element.

# Case
CASE_1 = 1
# Create model
model_info = triangleStruc(top_free=True, base_free=True, info=True)

# %%
# Remove oblique element
ops.remove("ele", model_info.remove_ele)

# Create data base
ODB = opst.post.CreateODB(odb_tag=CASE_1, save_fiber_sec_resp=True, fiber_ele_tags=[1, 2, 3, 4], model_update=True)

# Linear timeSeries
ts = 1
ops.timeSeries("Linear", ts)

# %%
# Pattern for static analysis
pattern = 100
ops.pattern("Plain", pattern, ts)
ops.load(model_info.ctrl_node, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

# %%
# Displacement path
disp_path = (
    np.array([
        0.0,
        0.05,
        -0.05,
        0.10,
        -0.10,
        0.15,
        -0.15,
        0.20,
        -0.20,
        0.25,
        -0.25,
        0.30,
        -0.30,
        0.0,
    ])
    * UNIT.m
)

# %%
# Analysis
disp, force = analysisLib(targets=disp_path, patternTag=pattern, ctrl_node=model_info.ctrl_node, ODB=ODB)
# Save data base
ODB.save_response()

# %%
# View results
plt.close("all")
plt.title("Single Column Hysteretic Curve")
plt.plot(disp, force, linewidth=1.0, label="Single Column", zorder=2)
plt.xlim(-0.4, 0.4)
plt.ylim(-500 * UNIT.kn, 500 * UNIT.kn)
plt.xlabel("Displacement (m)")
plt.ylabel("Force (kN)")
plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
plt.grid(linestyle="--", linewidth=0.5, zorder=1)
plt.show()

# %%
# Visualize model response
fig = opsplt.plot_nodal_responses_animation(odb_tag=CASE_1, resp_type="disp", show_undeformed=True)
fig
# fig.show()

# %%
# Remove and add element during analysis
# ---------------------------------------

# Case
CASE_2 = 2

"""Ok, For case 2 I want to remove and add element during analysis."""

# Create model
model_info = triangleStruc(top_free=True, base_free=True, info=False)

# Create data base
ODB = opst.post.CreateODB(odb_tag=CASE_2, fiber_ele_tags="ALL", model_update=True)

# Linear timeSeries
ts = 1
ops.timeSeries("Linear", ts)

# Pattern for static analysis
pattern_1 = 100
ops.pattern("Plain", pattern_1, ts)
ops.load(model_info.ctrl_node, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

# Displacement path
disp_path_1 = (
    np.array([
        0.0,
        0.05,
        -0.05,
        0.10,
        -0.10,
        0.15,
        -0.15,
        # 0.20, -0.20,
        # 0.25, -0.25,
        # 0.30, -0.30,
        0.0,
    ])
    * UNIT.m
)

# Analysis
disp_1, force_1 = analysisLib(targets=disp_path_1, patternTag=pattern_1, ctrl_node=model_info.ctrl_node, ODB=ODB)

# %%
# Remove element
ops.loadConst("-time", 0.0)  # Important !!!
ops.remove("ele", model_info.remove_ele)  # Remove element

# Pattern for static analysis (After remove element)
pattern_2 = 200
ops.pattern("Plain", pattern_2, ts)
ops.load(model_info.ctrl_node, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

# Displacement path (After remove element)
disp_path_2 = (
    np.array([
        0.0,
        # 0.05, -0.05,
        # 0.10, -0.10,
        # 0.15, -0.15,
        0.20,
        -0.20,
        0.25,
        -0.25,
        0.30,
        -0.30,
        0.0,
    ])
    * UNIT.m
)

# Analysis (After remove element)
disp_2, force_2 = analysisLib(targets=disp_path_2, patternTag=pattern_2, ctrl_node=model_info.ctrl_node, ODB=ODB)

# %%
# Add element
ops.loadConst("-time", 0.0)  # Important !!!
ops.element(*model_info.add_ele)  # Add element

# Pattern for static analysis (After add element)
pattern_3 = 300
ops.pattern("Plain", pattern_3, ts)
ops.load(model_info.ctrl_node, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

# Displacement path (After add element)
disp_path_3 = (
    np.array([
        0.0,
        0.05,
        -0.05,
        0.10,
        -0.10,
        0.15,
        -0.15,
        0.20,
        -0.20,
        0.25,
        -0.25,
        0.30,
        -0.30,
        0.0,
    ])
    * UNIT.m
)

# Analysis (After add element)
disp_3, force_3 = analysisLib(targets=disp_path_3, patternTag=pattern_3, ctrl_node=model_info.ctrl_node, ODB=ODB)

# %%
# Save data base
ODB.save_response()


# %%
# View results (After remove element)
# sphinx_gallery_thumbnail_number = 6
plt.close("all")
plt.title("Triangle Remove Element Hysteretic Curve")

plt.plot(
    disp,
    force,
    label="Single Column",
    color="gray",
    linewidth=1.5,
    linestyle="--",
    zorder=2,
)
plt.plot(disp_1, force_1, linewidth=1.0, label="Before Remove", zorder=3)
plt.plot(disp_2, force_2 + force_1[-1], linewidth=1.0, label="After Remove", zorder=3)

plt.xlim(-0.4, 0.4)
plt.ylim(-500 * UNIT.kn, 500 * UNIT.kn)
plt.xlabel("Displacement (m)")
plt.ylabel("Force (kN)")
plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
plt.grid(linestyle="--", linewidth=0.5, zorder=1)
plt.show()

# %%
# View results (After add element)
plt.close("all")
plt.title("Demaged Single Column Add Element Hysteretic Curve")

plt.plot(
    disp,
    force,
    label="Single Column",
    color="gray",
    linewidth=1.5,
    linestyle="--",
    zorder=2,
)
plt.plot(
    disp_3,
    force_3 + force_1[-1] + force_2[-1],
    linewidth=1.0,
    label="After Add",
    zorder=3,
)

plt.xlim(-0.4, 0.4)
plt.ylim(-500 * UNIT.kn, 500 * UNIT.kn)
plt.xlabel("Displacement (m)")
plt.ylabel("Force (kN)")
plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
plt.grid(linestyle="--", linewidth=0.5, zorder=1)
plt.show()

# %%
# Visualize model responses
fig = opsplt.plot_nodal_responses_animation(odb_tag=CASE_2, resp_type="disp", show_undeformed=True)
fig
# fig.show()

# %%
# Visualize final model (Case 2)
opsplt.set_plot_props(point_size=5, line_width=3)
fig = opsplt.plot_model(
    show_ele_numbering=True,
    show_local_axes=True,
)
fig
# fig.show()

# %%
# Check zeroLength element
ODB_ele_Link = opst.post.get_element_responses(odb_tag=CASE_2, ele_type="Link", print_info=False)
print(f"ODB_ele_link.eleTags: {ODB_ele_Link.eleTags.values}")

# Check zeroLengthSection element
ODB_ele_sec = opst.post.get_element_responses(odb_tag=CASE_2, ele_type="FiberSection", print_info=False)
print(f"ODB_ele_sec.eleTags: {ODB_ele_sec.eleTags.values}")

# %%
print(ODB_ele_sec.data_vars)

# %%
print(ODB_ele_sec.data_vars["ys"])
