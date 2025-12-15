import numpy as np
import openseespy.opensees as ops


def get_brinch(gamma, b, z, phi, c):
    """
    Computes the ultimate lateral resistance using the Brinch Hansen method.

    Parameters
    ----------
    gamma : float or array
        Soil unit weight
    b : float
        Width (e.g., pile diameter)
    z : float or array
        Depth
    phi : float
        Friction angle (radians)
    c : float
        Cohesion

    Returns
    -------
    pu : float or array
        Ultimate lateral resistance
    """

    # pressure at ground surface
    Kqo = np.exp((np.pi / 2 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(np.pi / 4 + phi / 2) - np.exp(
        -(np.pi / 2 - phi) * np.tan(phi)
    ) * np.cos(phi) * np.tan(np.pi / 4 - phi / 2)

    Kco = (
        np.cos(phi)
        / np.sin(phi)
        * (np.exp((np.pi / 2 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(np.pi / 4 + phi / 2) - 1)
    )

    # pressure at great depth
    dcinf = 1.58 + 4.09 * (np.tan(phi)) ** 4
    Nc = np.cos(phi) / np.sin(phi) * (np.exp(np.pi * np.tan(phi)) * (np.tan(np.pi / 4 + phi / 2)) ** 2 - 1)
    Ko = 1 - np.sin(phi)
    Kcinf = Nc * dcinf
    Kqinf = Kcinf * Ko * np.tan(phi)

    # pressure at arbitrary depth
    zb = z / b

    aq = (Kqo / (Kqinf - Kqo)) * (Ko * np.sin(phi) / np.sin(np.pi / 4 + phi / 2))
    KqD = (Kqo + Kqinf * aq * zb) / (1 + aq * zb)

    ac = (Kco / (Kcinf - Kco)) * 2 * np.sin(np.pi / 4 + phi / 2)
    KcD = (Kco + Kcinf * ac * zb) / (1 + ac * zb)

    pu = (gamma * z * KqD + c * KcD) * b

    return pu


def get_api_stiffness(phi, k_switch=0):
    """
    Determine the API-recommended subgrade modulus for a given friction angle.

    Parameters
    ----------
    phi : float or array
        Friction angle in radians.
    k_switch : int
        1 = above groundwater table
        0 = below groundwater table (default)

    Returns
    -------
    kAPI : float or array
        API subgrade modulus
    """

    # convert to degrees
    phi_deg = np.asarray(phi) * 180.0 / np.pi

    # list of friction angles (API chart)
    frict = np.array([28.8, 29.5, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0])

    # subgrade modulus values
    if k_switch == 1:
        # above groundwater table
        k = 271.45 * np.array([10, 23, 45, 61, 80, 100, 120, 140, 160, 182, 215, 250, 275])
    else:
        # below groundwater table (default)
        k = 271.45 * np.array([10, 20, 33, 42, 50, 60, 70, 85, 95, 107, 122, 141, 155])

    # linear interpolation (MATLAB interp1 equivalent)
    kAPI = np.interp(phi_deg, frict, k)

    return kAPI


def m_data_sort(D, z, T, H, phi, gamma, n):
    """
    Python version of mDataSort.m

    Parameters
    ----------
    D : float
        Pile diameter (or width)
    z : float
        Depth of liquefied / soft layer center (or reference depth)
    T : float
        Thickness of liquefied layer
    H : float
        Total pile length (analysis depth)
    phi : float
        Friction angle (radians)
    gamma : float
        Soil unit weight
    n : int
        Number of nodes / points along depth

    Returns
    -------
    pu : (n,) ndarray
        Ultimate lateral resistance (integrated over element length and flipped)
    y50 : (n,) ndarray
        Displacement at 50% of ultimate resistance (flipped)
    d : (n,) ndarray
        Nodal depths (from H down to 0)
    mPro : (m,) ndarray
        Displacement profile modifier for lateral spreading, defined
        for depths dL where d < z
    """

    # vector of nodal depths: 0 -> H, then flipped to H -> 0
    d = np.linspace(0.0, H, n)
    d = np.flip(d)  # equivalent to flipdim(d,1) for column

    # linear fit parameters for p_u (xp) and k_T (xk)
    xp = np.array([
        [0.421, 0.315],
        [0.428, 0.343],
        [0.474, 0.432],
        [3.577, 0.224],
    ])

    xk = np.array([
        [0.346, 0.220],
        [1.000, 0.252],
        [0.400, 0.341],
        [0.299, 0.239],
    ])

    # --------- ultimate lateral resistance reduction parameters ---------
    B = (z * np.tan(phi)) ** 3 * T / D**4

    aA = D / (z * np.tan(phi)) * xp[0, 0] * B ** xp[0, 1]
    aB = D / (z * np.tan(phi)) * xp[1, 0] * B ** xp[1, 1]
    scA = D**2 / (z * np.tan(phi)) * xp[2, 0] * B ** xp[2, 1]
    scB = D**2 / (z * np.tan(phi)) * xp[3, 0] * B ** xp[3, 1]

    # masks for different zones
    idA = d < (z - T)
    idB = d > z

    dA = d[idA]
    dB = d[idB]

    sA = (z - T) - dA
    sB = dB - z

    purA = 1.0 - aA * np.exp(-sA / scA)
    purB = 1.0 - aB * np.exp(-sB / scB)

    pur = np.zeros_like(d)
    nr = np.ones_like(d)

    pur[idA] = purA
    pur[idB] = purB

    zero_mask = pur == 0.0
    pur[zero_mask] = 0.01
    nr[zero_mask] = 0.1  # kept for completeness, though not used later

    # --------- initial stiffness reduction parameters ---------
    B = (z * np.tan(phi)) ** 3 * T / D**4

    aA = D / (z * np.tan(phi)) * np.exp(xk[0, 0]) * B ** xk[0, 1]
    aB = D / (z * np.tan(phi)) * np.exp(xk[1, 0]) * B ** xk[1, 1]
    scA = D**2 / (z * np.tan(phi)) * np.exp(xk[2, 0]) * B ** xk[2, 1]
    scB = D**2 / (z * np.tan(phi)) * np.exp(xk[3, 0]) * B ** xk[3, 1]

    ktrA = 1.0 - aA * np.exp(-sA / scA)
    ktrB = 1.0 - aB * np.exp(-sB / scB)

    ktr = np.zeros_like(d)
    ktr[idA] = ktrA
    ktr[idB] = ktrB

    ktr[ktr == 0.0] = 0.1

    # --------- ultimate lateral resistance distribution ---------
    # puRaw = getBrinch(gamma, D, d, phi, 0);
    pu_raw = get_brinch(gamma, D, d, phi, 0.0)
    pu = pu_raw * pur

    # --------- initial stiffness distribution (API stiffness) ---------
    # NOTE in MATLAB: getAPIstiffness(phi,2) -> below groundwater
    kAPI = get_api_stiffness(phi, k_switch=2)

    sigV = d * gamma
    sigV = np.where(sigV == 0.0, 0.01 * gamma, sigV)

    cSigma = np.sqrt(50.0 / sigV)
    kStar = cSigma * kAPI

    kt = (kStar * d) * ktr
    kt[kt == 0.0] = 0.01

    # --------- y50 distribution ---------
    y50 = np.arctanh(0.5) * (pu / kt)
    y50 = np.flip(y50)

    # --------- flip pu and scale by element length for pySimple model ---------
    elem_length = d[0] - d[1]  # d is descending, so this is positive
    pu = pu * elem_length
    pu = np.flip(pu)

    # --------- displacement profile for lateral spreading analysis ---------
    idL = d < z
    dL = d[idL]

    R = D / 2.0
    mPro = R * np.ones_like(dL)
    m = T / R

    idT = dL < T
    mPro[idT] = dL[idT] / m

    return pu, y50, d, mPro


def run_lateral_spreading_model(
    D: float,
    z: float,
    T: float,
    phi_deg: float,
    gamma: float,
    pileE: float,
    pileNu: float,
    n_node: int,
    H: float,
):
    """
    Build and run a lateral spreading pile analysis directly using OpenSeesPy.

    This function is the runtime equivalent of the original MATLAB/Tcl
    `makeInput` + running the generated .tcl file. It:
      - computes p-y spring parameters using `m_data_sort`
      - creates spring and pile nodes
      - defines PySimple1 springs and zeroLength elements
      - creates the elastic pile beam-column elements
      - applies boundary conditions and equalDOF constraints
      - defines recorders and lateral spreading displacement pattern
      - runs a static pushover analysis

    Parameters
    ----------
    D : float
        Pile diameter.
    z : float
        Depth of the center of the liquefied (weaker) layer.
    T : float
        Thickness of the liquefied (weaker) layer.
    phi_deg : float
        Soil friction angle in degrees.
    gamma : float
        Soil unit weight.
    pileE : float
        Young's modulus of the pile.
    pileNu : float
        Poisson's ratio of the pile.
    n_node : int
        Number of nodes along the pile.
    H : float
        Total embedded length of the pile (analysis depth).

    Returns
    -------
    results : dict
        Dictionary with basic information:
        - 'ok'         : OpenSees analysis return code
        - 'time'       : analysis execution time (seconds)
        - 'd'          : numpy array of nodal depths
        - 'pu'         : numpy array of ultimate spring resistance
        - 'y50'        : numpy array of reference displacement
        - 'mPro'       : numpy array of imposed displacement profile
        - 'nodeList6'  : list of pile node tags
        - 'beamElems'  : list of pile element tags
    """

    # ---------------------------------------------------------------
    # 1. Compute p-y curve parameters (Python version of mDataSort)
    # ---------------------------------------------------------------
    phi_rad = np.deg2rad(phi_deg)
    pu, y50, d, mPro = m_data_sort(D, z, T, H, phi_rad, gamma, n_node)

    print("============================================================")
    print(f"Pile: D = {D}, E = {pileE}, nu = {pileNu}")
    print(f"Soil: z = {z}, T = {T}, phi = {phi_deg}, gamma = {gamma}")
    print("Basic units: kN, m, s")
    print("============================================================\n")

    # ---------------------------------------------------------------
    # 2. Initialize OpenSeesPy model
    # ---------------------------------------------------------------
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # ---------------------------------------------------------------
    # 3. Create p-y spring nodes and boundary conditions
    # ---------------------------------------------------------------
    # Soil-side spring nodes (fully fixed)
    # Node tags: 2 .. n_node
    for k in range(2, n_node + 1):
        ops.node(k, 0.0, 0.0, d[k - 1])
        # Fix all 6 DOFs at soil side
        ops.fix(k, 1, 1, 1, 1, 1, 1)

    # Pile-side spring nodes (free in horizontal DOF only)
    # Node tags: 202 .. 200 + n_node
    for k in range(2, n_node + 1):
        ops.node(k + 200, 0.0, 0.0, d[k - 1])
        # Only DOF 1 (horizontal) free
        ops.fix(k + 200, 0, 1, 1, 1, 1, 1)

    print("Created p-y spring nodes and boundary conditions.")

    # ---------------------------------------------------------------
    # 4. Create PySimple1 materials (p-y springs)
    # ---------------------------------------------------------------
    # Material tags: 2 .. n_node
    for k in range(2, n_node + 1):
        ops.uniaxialMaterial("PySimple1", k, 2, pu[k - 1], y50[k - 1], 0.0)

    print("Created PySimple1 spring materials.")

    # ---------------------------------------------------------------
    # 5. Create zeroLength spring elements
    # ---------------------------------------------------------------
    # Element tags: 2 .. n_node
    for k in range(2, n_node + 1):
        ops.element("zeroLength", k, k, k + 200, "-mat", k, "-dir", 1)

    print("Created zeroLength p-y spring elements.")

    # ---------------------------------------------------------------
    # 6. Create pile nodes and boundary conditions
    # ---------------------------------------------------------------
    nodeList6 = []

    # Pile nodes: 501 .. 500 + n_node
    for k in range(1, n_node + 1):
        ops.node(k + 500, 0.0, 0.0, d[k - 1])
        nodeList6.append(k + 500)

    # Pile boundary conditions
    # Interior nodes: fix vertical and some rotations (0 1 0 1 0 1)
    for k in range(1, n_node):
        ops.fix(k + 500, 0, 1, 0, 1, 0, 1)

    # Tip (last node): also fix DOF 3 (vertical translation) -> (0 1 1 1 0 1)
    ops.fix(500 + n_node, 0, 1, 1, 1, 0, 1)

    print("Created pile nodes and boundary conditions.")

    # equalDOF constraints: tie pile nodes to pile-side spring nodes in DOF 1
    for k in range(2, n_node + 1):
        ops.equalDOF(k + 500, k + 200, 1)

    print("Applied equalDOF constraints between pile and spring nodes.")

    # ---------------------------------------------------------------
    # 7. Create elastic pile section and material
    # ---------------------------------------------------------------
    pi = 4.0 * np.arctan(1.0)
    radius = D / 2.0
    area = pi * radius**2
    I = pi * radius**4 / 4.0
    J = 2.0 * I
    E = pileE
    nu = pileNu
    G = E / (2.0 * (1.0 + nu))
    nIntPts = 3
    secTag = 1
    secTag3D = 3

    # Elastic beam section (no explicit torsion in this section)
    ops.section("Elastic", secTag, E, area, I, I, G, J)

    # Very stiff torsional material (for T DOF)
    ops.uniaxialMaterial("Elastic", 100, 1.0e10)

    # Aggregated 3D section with torsion
    ops.section("Aggregator", secTag3D, 100, "T", "-section", secTag)

    # Geometric transformation for vertical pile
    transTag = 1
    ops.geomTransf("Linear", transTag, 0.0, -1.0, 0.0)

    #  integrationTag for beam-column elements
    integrationTag = 1
    ops.beamIntegration("Legendre", integrationTag, secTag3D, 3)

    print("Created elastic pile section and transformation.")

    # ---------------------------------------------------------------
    # 8. Create pile beam-column elements
    # ---------------------------------------------------------------
    beamElementList = []

    # Element tags: 501 .. 500 + (n_node-1)
    for k in range(1, n_node):
        eleTag = 500 + k
        ops.element("dispBeamColumn", eleTag, 500 + k, 501 + k, transTag, integrationTag)
        beamElementList.append(eleTag)

    print("Created pile beam-column elements.")

    # # ---------------------------------------------------------------
    # # 9. Define recorders
    # # ---------------------------------------------------------------
    # dt = 0.5

    # ops.recorder(
    #     "Node",
    #     "-file",
    #     "pyDisplace.out",
    #     "-time",
    #     "-nodeRange",
    #     202,
    #     200 + n_node,
    #     "-dof",
    #     1,
    #     "-dT",
    #     dt,
    #     "disp",
    # )

    # ops.recorder(
    #     "Node",
    #     "-file",
    #     "pyForces.out",
    #     "-time",
    #     "-nodeRange",
    #     2,
    #     n_node,
    #     "-dof",
    #     1,
    #     "-dT",
    #     dt,
    #     "reaction",
    # )

    # ops.recorder(
    #     "Node",
    #     "-file",
    #     "Displacements.out",
    #     "-time",
    #     "-node",
    #     *nodeList6,
    #     "-dof",
    #     1,
    #     2,
    #     3,
    #     "-dT",
    #     dt,
    #     "disp",
    # )

    # ops.recorder(
    #     "Node",
    #     "-file",
    #     "Reaction.out",
    #     "-time",
    #     "-node",
    #     *nodeList6,
    #     "-dof",
    #     1,
    #     2,
    #     3,
    #     "-dT",
    #     dt,
    #     "reaction",
    # )

    # ops.recorder(
    #     "Element",
    #     "-file",
    #     "globalForces.out",
    #     "-time",
    #     "-ele",
    #     *beamElementList,
    #     "-dT",
    #     dt,
    #     "globalForce",
    # )

    # print("Recorders have been defined.")

    # ---------------------------------------------------------------
    # 10. Apply lateral spreading displacement profile
    # ---------------------------------------------------------------
    # mPro was computed in m_data_sort for depths d < z
    # Here we apply a static displacement pattern in DOF 1 on soil-side nodes
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 10, 1)

    # Node tags for soil-side springs: 2..len(mPro)+1
    for k in range(2, len(mPro) + 1):
        ops.sp(k, 1, float(mPro[k - 1]))

    print("Applied lateral spreading displacement profile.")

    # # ---------------------------------------------------------------
    # # 11. Run static analysis
    # # ---------------------------------------------------------------
    # ops.integrator("LoadControl", 0.05)
    # ops.numberer("RCM")
    # ops.system("BandGeneral")
    # ops.constraints("Transformation")
    # ops.test("NormDispIncr", 1e-5, 20, 1)
    # ops.algorithm("Newton")
    # ops.analysis("Static")

    # startT = time.time()
    # ok = ops.analyze(205)
    # endT = time.time()

    # print(f"Analysis finished with return code: {ok}")
    # print(f"Analysis execution time: {endT - startT:.3f} seconds")

    # # Clean up OpenSees model
    # ops.wipe()

    # ---------------------------------------------------------------
    # 12. Collect results summary
    # ---------------------------------------------------------------
    results = {
        # "ok": ok,
        # "time": endT - startT,
        "d": d,
        "pu": pu,
        "y50": y50,
        "mPro": mPro,
        "nodeList6": nodeList6,
        "beamElems": beamElementList,
    }

    return results


if __name__ == "__main__":
    model_info = run_lateral_spreading_model(
        D=0.6,  # pile diameter (m)
        z=10.0,  # depth to center of liquefied layer (m)
        T=2.0,  # thickness of liquefied layer (m)
        phi_deg=36.0,  # friction angle (degrees)
        gamma=17.0,  # unit weight (kN/m^3)
        pileE=3.0e7,  # pile Young's modulus (kN/m^2)
        pileNu=0.30,  # pile Poisson's ratio (-)
        n_node=80,  # number of pile nodes (you can adjust)
        H=40.0,  # analysis depth / embedded length (m) – adjust as needed
    )
