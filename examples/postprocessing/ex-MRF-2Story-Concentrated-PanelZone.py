r"""
Pushover and Dynamic Analyses of 2-Story Moment Frame with Panel Zones and RBS
==================================================================================

Original example by Laura Eads, Stanford University, see `here <https://opensees.berkeley.edu/wiki/index.php?title=Pushover_and_Dynamic_Analyses_of_2-Story_Moment_Frame_with_Panel_Zones_and_RBS>`_

This example is an extension of the Pushover Analysis of 2-Story Moment Frame and Dynamic Analysis of
2-Story Moment Frame examples which illustrates the explicit modeling of shear distortions in panel zones
and uses reduced beam sections (RBS) which are offset from the panel zones.
Both pushover and dynamic analyses are performed in this example.
The structure is the same 2-story, 1-bay steel moment resisting frame used in the other examples where the nonlinear
behavior is represented using the concentrated plasticity concept with rotational springs.
The rotational behavior of the plastic regions follows a bilinear hysteretic response based on
the Modified Ibarra Krawinkler Deterioration Model (Ibarra et al. 2005, Lignos and Krawinkler 2009, 2010).
For this example, all modes of cyclic deterioration are neglected.
A leaning column carrying gravity loads is linked to the frame to simulate P-Delta effects.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

import opstool as opst

# %%
# Some helper functions
# -----------------------


# %%
# Creates a bilinear rotational spring that follows the Modified Ibarra Krawinkler Deterioration Model (used in the concentrated model)


def rot_spring_2d_mod_ik_model(
    ele_id: int,
    node_r: int,
    node_c: int,
    K: float,
    as_pos: float,
    as_neg: float,
    My_pos: float,
    My_neg: float,
    LS: float,
    LK: float,
    LA: float,
    LD: float,
    cS: float,
    cK: float,
    cA: float,
    cD: float,
    th_pP: float,
    th_pN: float,
    th_pcP: float,
    th_pcN: float,
    ResP: float,
    ResN: float,
    th_uP: float,
    th_uN: float,
    DP: float,
    DN: float,
):
    """
    Create a 2D rotational spring using the Modified Ibarra–Krawinkler
    deterioration model (Bilin material) in OpenSeesPy.

    This function reproduces the Tcl procedure `rotSpring2DModIKModel`
    by D. G. Lignos, creating:
      1) A uniaxial Bilin material with strength and stiffness deterioration
      2) A zeroLength element acting in rotational DOF (dir = 6)
      3) An equalDOF constraint on translational DOFs (1, 2)

    Parameters
    ----------
    ele_id : int
        Element (and material) identification tag.
    node_r : int
        Retained (master) node.
    node_c : int
        Constrained (slave) node.
    K : float
        Initial stiffness after n-modification (Ibarra & Krawinkler, 2005).
    as_pos, as_neg : float
        Post-yield strain hardening ratios (positive / negative).
    My_pos, My_neg : float
        Yield moments (positive / negative).
    LS : float
        Basic strength deterioration parameter.
    LK : float
        Unloading stiffness deterioration parameter.
    LA : float
        Accelerated reloading stiffness deterioration parameter.
    LD : float
        Post-capping strength deterioration parameter.
    cS, cK, cA, cD : float
        Exponents for deterioration laws.
    th_pP, th_pN : float
        Plastic rotation capacities (positive / negative).
    th_pcP, th_pcN : float
        Post-capping rotation capacities (positive / negative).
    ResP, ResN : float
        Residual strength ratios (positive / negative).
    th_uP, th_uN : float
        Ultimate rotation capacities (positive / negative).
    DP, DN : float
        Cyclic deterioration rates (positive / negative).

    References
    ----------
    Ibarra & Krawinkler (2005)
    Ibarra et al. (2005)
    Lignos & Krawinkler (2009, 2010)
    """

    # --- Uniaxial material: Bilin (Modified Ibarra–Krawinkler model)
    ops.uniaxialMaterial(
        "Bilin",
        ele_id,
        K,
        as_pos,
        as_neg,
        My_pos,
        My_neg,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    # --- Zero-length rotational spring (about Z, dir=6 in 2D frame)
    ops.element(
        "zeroLength",
        ele_id,
        node_r,
        node_c,
        "-mat",
        ele_id,
        "-dir",
        6,
    )

    # --- Constrain translational DOFs (UX, UY)
    ops.equalDOF(node_r, node_c, 1, 2)


# %%
# Creates a low-stiffness rotational spring used in a leaning column


def rot_leaning_col(ele_id: int, node_r: int, node_c: int, K: float = 1e-9) -> None:
    """
    Create a zero-length rotational spring for a leaning column in OpenSeesPy.

    This function reproduces the Tcl procedure `rotLeaningCol`:
      1) Defines an Elastic uniaxial material with very small stiffness (default K=1e-9)
      2) Creates a zeroLength element acting in rotational DOF (dir = 6)
      3) Constrains translational DOFs (1, 2) via equalDOF (multi-point constraint)

    Parameters
    ----------
    ele_id : int
        Unique element (and material) tag for this zero-length rotational spring.
    node_r : int
        Retained (master) node for the multi-point constraint.
    node_c : int
        Constrained (slave) node for the multi-point constraint.
    K : float, optional
        Spring stiffness for the Elastic uniaxial material.
        Default is 1e-9 (consistent with the original Tcl script).
        Units follow the user's model unit system.

    Notes
    -----
    - The spring is assigned to rotational DOF `6`, consistent with typical 2D frame
      conventions in OpenSees where dir=6 corresponds to RZ (rotation about Z).
    - Translational DOFs (1, 2) are tied between node_r and node_c to avoid spurious
      relative translation across the spring.
    """

    # Uniaxial material: Elastic (very small stiffness)
    ops.uniaxialMaterial("Elastic", ele_id, K)

    # Zero-length rotational spring (about Z): dir = 6
    ops.element("zeroLength", ele_id, node_r, node_c, "-mat", ele_id, "-dir", 6)

    # Tie translations (UX, UY)
    ops.equalDOF(node_r, node_c, 1, 2)


# %%
# Creates eight elastic elements which form a rectangular panel zone


def elem_panel_zone_2d(
    ele_id: int,
    node_r: int,
    E: float,
    A_pz: float,
    I_pz: float,
    transf_tag: int,
) -> dict:
    """
    Create 2D panel-zone rigid-link elements using elasticBeamColumn.

    This function reproduces the Tcl procedure `elemPanelZone2D`:
      - Derives panel-zone node tags from the base node tag `node_r`
      - Creates 8 elasticBeamColumn elements (rigid links) around the joint panel zone

    Parameters
    ----------
    ele_id : int
        Base element tag for the panel zone (8 elements will be created: ele_id ... ele_id+7).
    node_r : int
        Node tag for the first point (top-left) of the panel zone. This node tag defines
        all other panel-zone node tags by a fixed numbering scheme (as in the Tcl script).
    E : float
        Young's modulus.
    A_pz : float
        Cross-sectional area of the rigid links forming the panel zone.
    I_pz : float
        Moment of inertia of the rigid links forming the panel zone.
    transf_tag : int
        Geometric transformation tag.

    Returns
    -------
    dict
        A dictionary containing:
          - "nodes": dict of derived node tags
          - "elements": list of created element tags (length 8)

    Notes
    -----
    - This function only creates elements. You must ensure all involved nodes exist.
    - DOF convention assumes 2D frame elements (ndm=2, ndf=3) and elasticBeamColumn usage.
    - Node numbering scheme is exactly copied from the original Tcl code.
    """

    # --- panel zone nodes (copied numbering scheme) ---
    node_xy01 = node_r  # top left of joint
    node_xy02 = node_xy01 + 1  # top left of joint (paired)
    node_xy03 = node_xy01 + 2  # top right of joint
    node_xy04 = node_xy01 + 3  # top right of joint (paired)
    node_xy05 = node_xy01 + 4  # middle right of joint
    node_xy06 = node_xy01 + 5  # bottom right of joint
    node_xy07 = node_xy01 + 6  # bottom right of joint (paired)
    node_xy08 = node_xy01 + 7  # bottom left of joint
    node_xy09 = node_xy01 + 8  # bottom left of joint (paired)
    node_xy10 = node_xy01 + 9  # middle left of joint

    # Tcl: set node_xy6  [expr ($node_xy01-1)/10 + 6]
    # Tcl: set node_xy7  [expr ($node_xy01-1)/10 + 7]
    # In Tcl, "/" is integer division when both operands are integers.
    base = (node_xy01 - 1) // 10
    node_xy6 = base + 6  # bottom center of joint
    node_xy7 = base + 7  # top center of joint

    # --- element tags (8 per panel zone) ---
    x1 = ele_id + 0
    x2 = ele_id + 1
    x3 = ele_id + 2
    x4 = ele_id + 3
    x5 = ele_id + 4
    x6 = ele_id + 5
    x7 = ele_id + 6
    x8 = ele_id + 7
    ele_tags = [x1, x2, x3, x4, x5, x6, x7, x8]

    # --- create panel zone elements (elasticBeamColumn) ---
    # tag, ndI, ndJ, A, E, I, transfTag
    ops.element("elasticBeamColumn", x1, node_xy02, node_xy7, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x2, node_xy7, node_xy03, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x3, node_xy05, node_xy04, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x4, node_xy06, node_xy05, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x5, node_xy6, node_xy07, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x6, node_xy08, node_xy6, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x7, node_xy09, node_xy10, A_pz, E, I_pz, transf_tag)
    ops.element("elasticBeamColumn", x8, node_xy10, node_xy01, A_pz, E, I_pz, transf_tag)

    return {
        "nodes": {
            "node_xy01": node_xy01,
            "node_xy02": node_xy02,
            "node_xy03": node_xy03,
            "node_xy04": node_xy04,
            "node_xy05": node_xy05,
            "node_xy06": node_xy06,
            "node_xy07": node_xy07,
            "node_xy08": node_xy08,
            "node_xy09": node_xy09,
            "node_xy10": node_xy10,
            "node_xy6": node_xy6,
            "node_xy7": node_xy7,
        },
        "elements": ele_tags,
    }


# %%
# Creates a rotational spring to capture panel zone shear distortions


def rot_panel_zone_2d(
    ele_id: int,
    node_r: int,
    node_c: int,
    E: float,
    Fy: float,
    dc: float,
    bf_c: float,
    tf_c: float,
    tp: float,
    db: float,
    Ry: float,
    a_s: float,
    nu: float = 0.30,
) -> dict:
    """
    Create a 2D panel-zone rotational spring using a trilinear equivalent
    Hysteretic uniaxial material and a zeroLength element (dir=6), plus
    equalDOF constraints as in the Tcl procedure `rotPanelZone2D`.

    Parameters
    ----------
    ele_id : int
        Unique element (and material) tag for this zero-length rotational spring.
    node_r : int
        Retained (master) node for the zeroLength spring.
    node_c : int
        Constrained (slave) node for the zeroLength spring.
    E : float
        Young's modulus.
    Fy : float
        Yield strength.
    dc : float
        Column depth.
    bf_c : float
        Column flange width.
    tf_c : float
        Column flange thickness.
    tp : float
        Panel zone thickness.
    db : float
        Beam depth.
    Ry : float
        Expected value factor for yield strength (typical 1.2).
        NOTE: Kept for interface compatibility with the Tcl script; not used in
        the provided Tcl equations.
    a_s : float
        Assumed strain hardening ratio (as in the Tcl script).
    nu : float, optional
        Poisson's ratio used to compute shear modulus G = E / (2*(1+nu)).
        Default 0.30 (same as Tcl).

    Returns
    -------
    dict
        Dictionary with computed stiffness/points and derived node tags, useful
        for debugging and reporting.

    Notes
    -----
    - The spring acts on rotational DOF `6` (RZ) using `zeroLength -dir 6`.
    - Translational DOFs (1,2) are tied between several node pairs to enforce
      rigid panel-zone kinematics (copied from Tcl numbering).
    - The Tcl code defines a symmetric Hysteretic material without pinching/damage
      (pinchX=1, pinchY=1, damage1=0, damage2=0, beta=0).
    """

    # --------------------------
    # Trilinear spring backbone
    # --------------------------
    Vy = 0.55 * Fy * dc * tp  # yield shear
    G = E / (2.0 * (1.0 + nu))  # shear modulus
    Ke = 0.95 * G * tp * dc  # elastic stiffness
    Kp = 0.95 * G * bf_c * (tf_c * tf_c) / db  # plastic stiffness

    gamma1_y = Vy / Ke
    M1y = gamma1_y * (Ke * db)

    gamma2_y = 4.0 * gamma1_y
    M2y = M1y + (Kp * db) * (gamma2_y - gamma1_y)

    gamma3_y = 100.0 * gamma1_y
    M3y = M2y + (a_s * Ke * db) * (gamma3_y - gamma2_y)

    # --------------------------
    # Uniaxial material (Hysteretic)
    # --------------------------
    # Tcl:
    # uniaxialMaterial Hysteretic eleID
    #   M1y gamma1_y  M2y gamma2_y M3y gamma3_y
    #  -M1y -gamma1_y -M2y -gamma2_y -M3y -gamma3_y
    #   1 1 0.0 0.0 0.0
    ops.uniaxialMaterial(
        "Hysteretic",
        ele_id,
        M1y,
        gamma1_y,
        M2y,
        gamma2_y,
        M3y,
        gamma3_y,
        -M1y,
        -gamma1_y,
        -M2y,
        -gamma2_y,
        -M3y,
        -gamma3_y,
        1.0,
        1.0,  # pinchX, pinchY (no pinching)
        0.0,
        0.0,  # damage1, damage2 (no damage)
        0.0,  # beta
    )

    # --------------------------
    # ZeroLength rotational spring
    # --------------------------
    ops.element("zeroLength", ele_id, node_r, node_c, "-mat", ele_id, "-dir", 6)

    # Tie translations between spring nodes
    ops.equalDOF(node_r, node_c, 1, 2)

    # --------------------------
    # Additional MPC constraints (copied numbering scheme)
    # --------------------------
    # Left Top Corner of PZ
    nodeR_1 = node_r - 2
    nodeR_2 = nodeR_1 + 1

    # Right Bottom Corner of PZ
    nodeR_6 = node_r + 3
    nodeR_7 = nodeR_6 + 1

    # Left Bottom Corner of PZ
    nodeL_8 = node_r + 5
    nodeL_9 = nodeL_8 + 1

    ops.equalDOF(nodeR_1, nodeR_2, 1, 2)
    ops.equalDOF(nodeR_6, nodeR_7, 1, 2)
    ops.equalDOF(nodeL_8, nodeL_9, 1, 2)

    return {
        "Vy": Vy,
        "G": G,
        "Ke": Ke,
        "Kp": Kp,
        "gamma1_y": gamma1_y,
        "gamma2_y": gamma2_y,
        "gamma3_y": gamma3_y,
        "M1y": M1y,
        "M2y": M2y,
        "M3y": M3y,
        "nodes": {
            "node_r": node_r,
            "node_c": node_c,
            "nodeR_1": nodeR_1,
            "nodeR_2": nodeR_2,
            "nodeR_6": nodeR_6,
            "nodeR_7": nodeR_7,
            "nodeL_8": nodeL_8,
            "nodeL_9": nodeL_9,
        },
    }


# %%
# Geometry definitions
# ------------------------------------------------------------

NStories = 2
NBays = 1
WBay = 30.0 * 12.0
HStory1 = 15.0 * 12.0
HStoryTyp = 12.0 * 12.0
HBuilding = HStory1 + (NStories - 1) * HStoryTyp

Pier1 = 0.0
Pier2 = Pier1 + WBay
Pier3 = Pier2 + WBay  # P-delta column line
Floor1 = 0.0
Floor2 = Floor1 + HStory1
Floor3 = Floor2 + HStoryTyp

# panel zone dimensions
pzlat23 = 24.5 / 2.0  # half column depth
pzvert23 = 27.1 / 2.0  # half beam depth

# plastic hinge offsets from beam-column centerlines
phlat23 = pzlat23 + 7.5 + 22.5 / 2.0
phvert23 = pzvert23 + 0.0

# ------------------------------------------------------------
# Masses
# ------------------------------------------------------------
g = 386.2
Floor2Weight = 535.0
Floor3Weight = 525.0
WBuilding = Floor2Weight + Floor3Weight

NodalMass2 = (Floor2Weight / g) / 2.0
NodalMass3 = (Floor3Weight / g) / 2.0
Negligible = 1e-9


# %%
# Build the model
# ------------------------------------------------------------
# %%
def build_model() -> None:
    """
    Build the concentrated-plasticity + panel-zone model (2D, 2-story, 1-bay + P-delta column)
    converted from the provided Tcl script.

    Parameters
    ----------
    analysis_type : {"pushover", "dynamic"}
        Choose analysis type.
    """

    # ------------------------------------------------------------
    # Set up & output directory
    # ------------------------------------------------------------
    ops.wipe()
    ops.model("BasicBuilder", "-ndm", 2, "-ndf", 3)

    # ------------------------------------------------------------
    # Nodes (direct translation)
    # ------------------------------------------------------------
    # Base / P-delta
    ops.node(11, Pier1, Floor1)
    ops.node(21, Pier2, Floor1)
    ops.node(31, Pier3, Floor1)
    ops.node(32, Pier3, Floor2)
    ops.node(33, Pier3, Floor3)

    # Column hinge nodes
    ops.node(117, Pier1, Floor1)
    ops.node(217, Pier2, Floor1)

    ops.node(125, Pier1, Floor2 - phvert23)
    ops.node(126, Pier1, Floor2 - phvert23)
    ops.node(225, Pier2, Floor2 - phvert23)
    ops.node(226, Pier2, Floor2 - phvert23)
    ops.node(326, Pier3, Floor2)

    ops.node(127, Pier1, Floor2 + phvert23)
    ops.node(128, Pier1, Floor2 + phvert23)
    ops.node(227, Pier2, Floor2 + phvert23)
    ops.node(228, Pier2, Floor2 + phvert23)
    ops.node(327, Pier3, Floor2)

    ops.node(135, Pier1, Floor3 - phvert23)
    ops.node(136, Pier1, Floor3 - phvert23)
    ops.node(235, Pier2, Floor3 - phvert23)
    ops.node(236, Pier2, Floor3 - phvert23)
    ops.node(336, Pier3, Floor3)

    # Beam hinge nodes
    ops.node(121, Pier1 + phlat23, Floor2)
    ops.node(122, Pier1 + phlat23, Floor2)
    ops.node(223, Pier2 - phlat23, Floor2)
    ops.node(224, Pier2 - phlat23, Floor2)

    ops.node(131, Pier1 + phlat23, Floor3)
    ops.node(132, Pier1 + phlat23, Floor3)
    ops.node(233, Pier2 - phlat23, Floor3)
    ops.node(234, Pier2 - phlat23, Floor3)

    # Panel zone nodes (Pier 1, Floor 2)
    ops.node(1201, Pier1 - pzlat23, Floor2 + phvert23)
    ops.node(1202, Pier1 - pzlat23, Floor2 + phvert23)
    ops.node(1203, Pier1 + pzlat23, Floor2 + phvert23)
    ops.node(1204, Pier1 + pzlat23, Floor2 + phvert23)
    ops.node(1205, Pier1 + pzlat23, Floor2)
    ops.node(1206, Pier1 + pzlat23, Floor2 - phvert23)
    ops.node(1207, Pier1 + pzlat23, Floor2 - phvert23)
    ops.node(1208, Pier1 - pzlat23, Floor2 - phvert23)
    ops.node(1209, Pier1 - pzlat23, Floor2 - phvert23)
    ops.node(1210, Pier1 - pzlat23, Floor2)

    # Panel zone nodes (Pier 2, Floor 2)
    ops.node(2201, Pier2 - pzlat23, Floor2 + phvert23)
    ops.node(2202, Pier2 - pzlat23, Floor2 + phvert23)
    ops.node(2203, Pier2 + pzlat23, Floor2 + phvert23)
    ops.node(2204, Pier2 + pzlat23, Floor2 + phvert23)
    ops.node(2205, Pier2 + pzlat23, Floor2)
    ops.node(2206, Pier2 + pzlat23, Floor2 - phvert23)
    ops.node(2207, Pier2 + pzlat23, Floor2 - phvert23)
    ops.node(2208, Pier2 - pzlat23, Floor2 - phvert23)
    ops.node(2209, Pier2 - pzlat23, Floor2 - phvert23)
    ops.node(2210, Pier2 - pzlat23, Floor2)

    # Panel zone nodes (Pier 1, Floor 3)
    ops.node(1301, Pier1 - pzlat23, Floor3 + phvert23)
    ops.node(1302, Pier1 - pzlat23, Floor3 + phvert23)
    ops.node(1303, Pier1 + pzlat23, Floor3 + phvert23)
    ops.node(1304, Pier1 + pzlat23, Floor3 + phvert23)
    ops.node(1305, Pier1 + pzlat23, Floor3)
    ops.node(1306, Pier1 + pzlat23, Floor3 - phvert23)
    ops.node(1307, Pier1 + pzlat23, Floor3 - phvert23)
    ops.node(1308, Pier1 - pzlat23, Floor3 - phvert23)
    ops.node(1309, Pier1 - pzlat23, Floor3 - phvert23)
    ops.node(1310, Pier1 - pzlat23, Floor3)
    ops.node(137, Pier1, Floor3 + phvert23)  # extra top node (no column above)

    # Panel zone nodes (Pier 2, Floor 3)
    ops.node(2301, Pier2 - pzlat23, Floor3 + phvert23)
    ops.node(2302, Pier2 - pzlat23, Floor3 + phvert23)
    ops.node(2303, Pier2 + pzlat23, Floor3 + phvert23)
    ops.node(2304, Pier2 + pzlat23, Floor3 + phvert23)
    ops.node(2305, Pier2 + pzlat23, Floor3)
    ops.node(2306, Pier2 + pzlat23, Floor3 - phvert23)
    ops.node(2307, Pier2 + pzlat23, Floor3 - phvert23)
    ops.node(2308, Pier2 - pzlat23, Floor3 - phvert23)
    ops.node(2309, Pier2 - pzlat23, Floor3 - phvert23)
    ops.node(2310, Pier2 - pzlat23, Floor3)
    ops.node(237, Pier2, Floor3 + phvert23)

    # ------------------------------------------------------------
    # Mass assignment (only at joint nodes)
    # ------------------------------------------------------------
    ops.mass(1205, NodalMass2, Negligible, Negligible)
    ops.mass(2205, NodalMass2, Negligible, Negligible)
    ops.mass(1305, NodalMass3, Negligible, Negligible)
    ops.mass(2305, NodalMass3, Negligible, Negligible)

    # equalDOF constraints for rigid diaphragm in x
    dof1 = 1
    ops.equalDOF(1205, 2205, dof1)
    ops.equalDOF(1205, 32, dof1)
    ops.equalDOF(1305, 2305, dof1)
    ops.equalDOF(1305, 33, dof1)

    # Base fixities
    ops.fix(11, 1, 1, 1)
    ops.fix(21, 1, 1, 1)
    ops.fix(31, 1, 1, 0)  # P-delta column pinned

    # ------------------------------------------------------------
    # Section properties / materials
    # ------------------------------------------------------------
    Es = 29000.0
    Fy = 50.0

    # Columns W24x131
    Acol_12 = 38.5
    Icol_12 = 4020.0
    Mycol_12 = 20350.0
    dcol_12 = 24.5
    bfcol_12 = 12.9
    tfcol_12 = 0.96
    twcol_12 = 0.605

    # Beams W27x102
    Abeam_23 = 30.0
    Ibeam_23 = 3620.0
    Mybeam_23 = 10938.0
    dbeam_23 = 27.1

    # n-modification
    n = 10.0
    Icol_12mod = Icol_12 * (n + 1.0) / n
    Ibeam_23mod = Ibeam_23 * (n + 1.0) / n

    Ks_col_1 = n * 6.0 * Es * Icol_12mod / (HStory1 - phvert23)
    Ks_col_2 = n * 6.0 * Es * Icol_12mod / (HStoryTyp - 2.0 * phvert23)
    Ks_beam_23 = n * 6.0 * Es * Ibeam_23mod / (WBay - 2.0 * phlat23)

    # ------------------------------------------------------------
    # Geometric transformation
    # ------------------------------------------------------------
    PDeltaTransf = 1
    ops.geomTransf("PDelta", PDeltaTransf)

    # ------------------------------------------------------------
    # Elastic frame elements
    # ------------------------------------------------------------
    # Columns
    ops.element("elasticBeamColumn", 111, 117, 125, Acol_12, Es, Icol_12mod, PDeltaTransf)
    ops.element("elasticBeamColumn", 121, 217, 225, Acol_12, Es, Icol_12mod, PDeltaTransf)

    ops.element("elasticBeamColumn", 112, 128, 135, Acol_12, Es, Icol_12mod, PDeltaTransf)
    ops.element("elasticBeamColumn", 122, 228, 235, Acol_12, Es, Icol_12mod, PDeltaTransf)

    # Beams (note: some use unmodified I, some use modified I)
    ops.element("elasticBeamColumn", 2121, 1205, 121, Abeam_23, Es, Ibeam_23, PDeltaTransf)
    ops.element("elasticBeamColumn", 212, 122, 223, Abeam_23, Es, Ibeam_23mod, PDeltaTransf)
    ops.element("elasticBeamColumn", 2122, 224, 2210, Abeam_23, Es, Ibeam_23, PDeltaTransf)

    ops.element("elasticBeamColumn", 2131, 1305, 131, Abeam_23, Es, Ibeam_23, PDeltaTransf)
    ops.element("elasticBeamColumn", 213, 132, 233, Abeam_23, Es, Ibeam_23mod, PDeltaTransf)
    ops.element("elasticBeamColumn", 2132, 234, 2310, Abeam_23, Es, Ibeam_23, PDeltaTransf)

    # ------------------------------------------------------------
    # P-delta column + rigid trusses
    # ------------------------------------------------------------
    TrussMatID = 600
    Arigid = 1000.0
    Irigid = 100000.0
    ops.uniaxialMaterial("Elastic", TrussMatID, Es)

    ops.element("truss", 622, 2205, 32, Arigid, TrussMatID)
    ops.element("truss", 623, 2305, 33, Arigid, TrussMatID)

    ops.element("elasticBeamColumn", 731, 31, 326, Arigid, Es, Irigid, PDeltaTransf)
    ops.element("elasticBeamColumn", 732, 327, 336, Arigid, Es, Irigid, PDeltaTransf)

    # ------------------------------------------------------------
    # Panel zone rigid elements (call your converted function)
    # ------------------------------------------------------------
    Apz = 1000.0
    Ipz = 1.0e5

    # elem_panel_zone_2d(ele_id, node_r, E, A_pz, I_pz, transf_tag)
    elem_panel_zone_2d(500121, 1201, Es, Apz, Ipz, PDeltaTransf)
    elem_panel_zone_2d(500221, 2201, Es, Apz, Ipz, PDeltaTransf)
    elem_panel_zone_2d(500131, 1301, Es, Apz, Ipz, PDeltaTransf)
    elem_panel_zone_2d(500231, 2301, Es, Apz, Ipz, PDeltaTransf)

    # ------------------------------------------------------------
    # Rotational springs (Modified IK)
    # ------------------------------------------------------------
    McMy = 1.05
    LS = LK = LA = LD = 1000.0
    cS = cK = cA = cD = 1.0
    th_pP = th_pN = 0.025
    th_pcP = th_pcN = 0.3
    ResP = ResN = 0.4
    th_uP = th_uN = 0.4
    DP = DN = 1.0

    a_mem = (n + 1.0) * (Mycol_12 * (McMy - 1.0)) / (Ks_col_1 * th_pP)
    b = a_mem / (1.0 + n * (1.0 - a_mem))

    # Story 1 column springs
    rot_spring_2d_mod_ik_model(
        3111,
        11,
        117,
        Ks_col_1,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        3211,
        21,
        217,
        Ks_col_1,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    rot_spring_2d_mod_ik_model(
        3112,
        126,
        125,
        Ks_col_1,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        3212,
        226,
        225,
        Ks_col_1,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    # Story 2 b recompute
    a_mem = (n + 1.0) * (Mycol_12 * (McMy - 1.0)) / (Ks_col_2 * th_pP)
    b = a_mem / (1.0 + n * (1.0 - a_mem))

    rot_spring_2d_mod_ik_model(
        3121,
        127,
        128,
        Ks_col_2,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        3221,
        227,
        228,
        Ks_col_2,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    rot_spring_2d_mod_ik_model(
        3122,
        136,
        135,
        Ks_col_2,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        3222,
        236,
        235,
        Ks_col_2,
        b,
        b,
        Mycol_12,
        -Mycol_12,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    ops.region(1, "-ele", 3111, 3211, 3112, 3212, 3121, 3221, 3122, 3222)

    # Beam springs
    th_pP = th_pN = 0.02
    th_pcP = th_pcN = 0.16
    a_mem = (n + 1.0) * (Mybeam_23 * (McMy - 1.0)) / (Ks_beam_23 * th_pP)
    b = a_mem / (1.0 + n * (1.0 - a_mem))

    rot_spring_2d_mod_ik_model(
        4121,
        121,
        122,
        Ks_beam_23,
        b,
        b,
        Mybeam_23,
        -Mybeam_23,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        4122,
        223,
        224,
        Ks_beam_23,
        b,
        b,
        Mybeam_23,
        -Mybeam_23,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        4131,
        131,
        132,
        Ks_beam_23,
        b,
        b,
        Mybeam_23,
        -Mybeam_23,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )
    rot_spring_2d_mod_ik_model(
        4132,
        233,
        234,
        Ks_beam_23,
        b,
        b,
        Mybeam_23,
        -Mybeam_23,
        LS,
        LK,
        LA,
        LD,
        cS,
        cK,
        cA,
        cD,
        th_pP,
        th_pN,
        th_pcP,
        th_pcN,
        ResP,
        ResN,
        th_uP,
        th_uN,
        DP,
        DN,
    )

    ops.region(2, "-ele", 4121, 4122, 4131, 4132)

    # Panel zone springs
    Ry = 1.2
    as_PZ = 0.03
    rot_panel_zone_2d(
        41200,
        1203,
        1204,
        Es,
        Fy,
        dcol_12,
        bfcol_12,
        tfcol_12,
        twcol_12,
        dbeam_23,
        Ry,
        as_PZ,
    )
    rot_panel_zone_2d(
        42200,
        2203,
        2204,
        Es,
        Fy,
        dcol_12,
        bfcol_12,
        tfcol_12,
        twcol_12,
        dbeam_23,
        Ry,
        as_PZ,
    )
    rot_panel_zone_2d(
        41300,
        1303,
        1304,
        Es,
        Fy,
        dcol_12,
        bfcol_12,
        tfcol_12,
        twcol_12,
        dbeam_23,
        Ry,
        as_PZ,
    )
    rot_panel_zone_2d(
        42300,
        2303,
        2304,
        Es,
        Fy,
        dcol_12,
        bfcol_12,
        tfcol_12,
        twcol_12,
        dbeam_23,
        Ry,
        as_PZ,
    )

    # P-delta leaning column rotational springs (zero stiffness)
    rot_leaning_col(5312, 32, 326)
    rot_leaning_col(5321, 32, 327)
    rot_leaning_col(5322, 33, 336)
    ops.region(3, "-ele", 5312, 5321, 5322)
    print("Model Built")


# %%
# Run gravity analyses
# ------------------------------------------------------------


def run_gravity_analysis() -> None:
    ops.wipeAnalysis()
    ops.timeSeries("Constant", 101)
    ops.pattern("Plain", 101, 101)

    P_PD2 = -398.02
    P_PD3 = -391.31
    ops.load(32, 0.0, P_PD2, 0.0)
    ops.load(33, 0.0, P_PD3, 0.0)

    P_F2 = 0.5 * (-Floor2Weight - P_PD2)
    P_F3 = 0.5 * (-Floor3Weight - P_PD3)

    ops.load(127, 0.0, P_F2, 0.0)
    ops.load(227, 0.0, P_F2, 0.0)
    ops.load(137, 0.0, P_F3, 0.0)
    ops.load(237, 0.0, P_F3, 0.0)

    Tol = 1.0e-6
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", Tol, 6)
    ops.algorithm("Newton")
    NstepGravity = 10
    DGravity = 1.0 / NstepGravity
    ops.integrator("LoadControl", DGravity)
    ops.analysis("Static")
    ops.analyze(NstepGravity)

    ops.loadConst("-time", 0.0)

    print("Gravity analysis complete")


# %%
# Visualize the model
# ------------------------------------------------------------
build_model()
run_gravity_analysis()

opst.vis.pyvista.plot_model(show_outline=False).show()

# %%
# Visualize first and fourth mode shapes
# ------------------------------------------------------------
opst.vis.pyvista.plot_eigen(mode_tags=[1, 4], subplots=False).show()


# %%
# Run pushover analysis
# ------------------------------------------------------------

build_model()
run_gravity_analysis()

# %%
# Pushover analysis setup
lat2 = 16.255
lat3 = 31.636

ops.timeSeries("Linear", 200)
ops.pattern("Plain", 200, 200)
ops.load(1205, lat2, 0.0, 0.0)
ops.load(2205, lat2, 0.0, 0.0)
ops.load(1305, lat3, 0.0, 0.0)
ops.load(2305, lat3, 0.0, 0.0)

IDctrlNode = 1305
IDctrlDOF = 1
Dmax = 0.1 * HBuilding
Dincr = 0.05

ops.wipeAnalysis()
ops.constraints("Plain")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormUnbalance", 1.0e-5, 400)
ops.algorithm("Newton")
ops.integrator("DisplacementControl", IDctrlNode, IDctrlDOF, Dincr)
ops.analysis("Static")

# %%
# Pushover analysis loop and ODB recording
smart_analysis = opst.anlys.SmartAnalyze(
    analysis_type="Static",
    testTol=1.0e-6,
    testType="NormDispIncr",
    testIterTimes=100,
    tryAddTestTimes=True,  # add test times to the analysis
    testIterTimesMore=[250, 500, 1000],
    tryAlterAlgoTypes=False,  # fix algorithm
    algoTypes=[40],  # algorithm is KrylovNewton
    relaxation=0.5,
    minStep=1e-6,  # minimum step size for substepping
    debugMode=True,
    printPer=100,
)
segs = smart_analysis.static_split(targets=[0, Dmax], maxStep=Dincr)

ODB = opst.post.CreateODB(odb_tag="pushover", interpolate_beam_disp=True)
for seg in segs:
    smart_analysis.StaticAnalyze(node=IDctrlNode, dof=IDctrlDOF, seg=seg)
    ODB.fetch_response_step()
ODB.save_response()
print("Pushover complete")

# %%
# Postprocessing pushover anslysis results
# ------------------------------------------------------------

# %%
# Pushover plots
# ****************

# %%
# plot disp
opst.vis.pyvista.plot_nodal_responses(
    odb_tag="pushover",
    resp_type="disp",
    slides=True,
    interpolate_beam_disp=True,
    defo_scale=5.0,
).show()

# %%
# plot section forces

# sphinx_gallery_thumbnail_number = 4
opst.vis.pyvista.plot_frame_responses(
    odb_tag="pushover",
    resp_type="sectionForces",
    slides=False,
).show()

# %%
# plot displacement animation

# opst.vis.pyvista.plot_nodal_responses_animation(
#     odb_tag="pushover",
#     resp_type="disp",
#     interpolate_beam_disp=True,
#     defo_scale=2.0,
#     framerate=30,
#     savefig="pushover_disp_animation.mp4",  # mp4 more efficient but gif more widely supported
# ).close()

# %%
# Pushover data
# ****************

disp_pushover = opst.post.get_nodal_responses(odb_tag="pushover", resp_type="disp")
react_pushover = opst.post.get_nodal_responses(odb_tag="pushover", resp_type="reaction")
print(disp_pushover)
print(react_pushover)

# %%
tota_react_x = react_pushover.sum(dim="nodeTags", skipna=True).sel(DOFs="UX")
disp_x_ctrl = disp_pushover.sel(nodeTags=IDctrlNode, DOFs="UX")

plt.plot(disp_x_ctrl, -tota_react_x, lw=2)
plt.xlabel("Control Node Displacement UX (in)")
plt.ylabel("Base Shear (kips)")
plt.title("Pushover Curve")
plt.grid(True)
plt.show()


# %%
# Run dynamic analysis
# ------------------------------------------------------------
build_model()
run_gravity_analysis()

# %%
# Eigenvalue analysis and Rayleigh damping
pi = 2.0 * np.asin(1.0)
nEigenI = 1
nEigenJ = 2

lam = ops.eigen(nEigenJ)  # returns list-like
lamI = lam[nEigenI - 1]
lamJ = lam[nEigenJ - 1]
w1 = np.sqrt(lamI)
w2 = np.sqrt(lamJ)
T1 = 2.0 * pi / w1
T2 = 2.0 * pi / w2
print(f"T1 = {T1} s")
print(f"T2 = {T2} s")
# Rayleigh damping
zeta = 0.02
n = 10.0  # n-modification factor
a0 = zeta * 2.0 * w1 * w2 / (w1 + w2)
a1 = zeta * 2.0 / (w1 + w2)
a1_mod = a1 * (1.0 + n) / n

# Regions for damping (same as Tcl)
ops.region(4, "-eleRange", 111, 213, "-rayleigh", 0.0, 0.0, a1_mod, 0.0)
ops.region(5, "-eleRange", 2121, 2132, "-rayleigh", 0.0, 0.0, a1, 0.0)
ops.region(6, "-eleRange", 500000, 599999, "-rayleigh", 0.0, 0.0, a1, 0.0)
ops.region(7, "-node", 1205, 1305, 2205, 2305, "-rayleigh", a0, 0.0, 0.0, 0.0)

# %%
# Ground motion parameters (you MUST set GMfile correctly)
patternID = 1
GMdirection = 1
GMfile = "utils/NR94cnp.txt"  # <-- change to your accel file (typically .txt/.AT2)
dt = 0.01
Scalefact = 1

gm_data = np.loadtxt(GMfile).ravel() * g

plt.plot(gm_data)
plt.show()

# %%
# TimeSeries (OpenSeesPy style: define a timeSeries, then reference by tag)
tsTag = 11
ops.timeSeries("Path", tsTag, "-dt", dt, "-values", *gm_data, "-factor", Scalefact)

# Uniform excitation
ops.pattern("UniformExcitation", patternID, GMdirection, "-accel", tsTag)

# %%
# Transient analysis setup
dt_analysis = dt
ops.wipeAnalysis()
ops.constraints("Plain")
ops.numberer("RCM")
ops.system("UmfPack")
ops.test("NormDispIncr", 1.0e-8, 10)
ops.algorithm("Newton")
ops.integrator("Newmark", 0.5, 0.25)
ops.analysis("Transient")

# %%
# Dynamic analysis loop and ODB recording
NumSteps = len(gm_data)

smart_analysis = opst.anlys.SmartAnalyze(
    analysis_type="Transient",
    testTol=1.0e-6,
    testType="NormDispIncr",
    testIterTimes=100,
    tryAddTestTimes=True,  # add test times to the analysis
    testIterTimesMore=[250, 500, 1000],
    tryAlterAlgoTypes=False,  # fix algorithm
    algoTypes=[40],  # algorithm is KrylovNewton
    relaxation=0.5,
    minStep=1e-6,  # minimum step size for substepping
    debugMode=True,
    printPer=100,
)
smart_analysis.transient_split(NumSteps)
ODB = opst.post.CreateODB(odb_tag="dynamic", interpolate_beam_disp=True)
for _ in range(NumSteps):
    smart_analysis.TransientAnalyze(dt_analysis)
    ODB.fetch_response_step()
ODB.save_response()

print("Dynamic analysis complete")


# %%
# Seismic analysis postprocessing
# ------------------------------------------------------------

# %%
# plot disp
opst.vis.pyvista.plot_nodal_responses(
    odb_tag="dynamic",
    resp_type="disp",
    slides=False,
    interpolate_beam_disp=False,
    defo_scale=2.0,
).show()

# %%

# opst.vis.pyvista.plot_nodal_responses_animation(
#     odb_tag="dynamic",
#     resp_type="disp",
#     interpolate_beam_disp=False,
#     defo_scale=3.0,
#     framerate=300,
#     savefig="dynamic_disp_animation.mp4",  # mp4 more efficient but gif more widely supported
# ).close()

# %%
disp_dynamcic = opst.post.get_nodal_responses(odb_tag="dynamic", resp_type="disp")
react_dynamcic = opst.post.get_nodal_responses(odb_tag="dynamic", resp_type="reaction")


# %%
disp_ctr = disp_dynamcic.sel(nodeTags=IDctrlNode, DOFs="UX")
plt.plot(disp_ctr.time, disp_ctr)
plt.xlabel("Time (s)")
plt.ylabel("Control Node Displacement UX (in)")
plt.title("Dynamic Analysis - Control Node Displacement Time History")
plt.grid(True)
plt.show()

# %%
link_resp = opst.post.get_element_responses(odb_tag="dynamic", ele_type="Link")
defo = link_resp["basicDeformation"].sel(eleTags=3111, DOFs="UX")
forces = link_resp["basicForce"].sel(eleTags=3111, DOFs="UX")

plt.plot(defo, forces)
plt.xlabel("Deformation UX (in)")
plt.ylabel("Force UX (kips)")
plt.title("element 3111 Link Force-Deformation Time History")
plt.grid(True)
plt.show()
