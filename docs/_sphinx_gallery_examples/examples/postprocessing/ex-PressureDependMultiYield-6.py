r"""
Dry single BbarBrick element with pressure dependent material
================================================================

See `PressureDependMultiYield-Example 6 <https://opensees.berkeley.edu/wiki/index.php?title=PressureDependMultiYield-Example_6>`_ example for more details.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

import opstool as opst

# %%
# MODEL CONFIGURATION & PARAMETERS
# ---------------------------------

# ===============================================================
#  MODEL CONFIGURATION & PARAMETERS
#  This script converts the original Tcl model into OpenSeesPy.
#  A dry bbarBrick element with PressureDependMultiYield material
#  is subjected to 1D sinusoidal base excitation.
# ===============================================================

ops.wipe()

# --- Material & physical parameters ---
friction = 31.40  # Friction angle of the soil
phaseTransform = 26.50  # Phase transformation angle
E1 = 93178.4  # Young's modulus
poisson1 = 0.40  # Poisson’s ratio

# Derived elastic constants
G1 = E1 / (2.0 * (1.0 + poisson1))  # Shear modulus
B1 = E1 / (3.0 * (1.0 - 2.0 * poisson1))  # Bulk modulus

gamma = 0.600  # Newmark gamma parameter

dt = 0.01  # Time step for dynamic analysis
numSteps = 1600  # Number of analysis steps

rhoS = 2.00  # Solid mass density (saturated)
rhoF = 0.00  # Fluid density (dry case → 0)
densityMult = 1.0  # Density multiplier

Bfluid = 2.2e6  # Fluid bulk modulus (unused in dry case)
fluid1 = 1  # Tag for fluid material (not used)
solid1 = 10  # Tag for solid material

accMul = 4.0  # Input acceleration scaling factor

pi = np.pi
inclination = 0.0  # Inclination angle of the element

massProportionalDamping = 0.0
InitStiffnessProportionalDamping = 0.001

# Body forces in global coordinates
bUnitWeightX = (rhoS - 0.0) * 9.81 * np.sin(inclination / 180.0 * pi) * densityMult
bUnitWeightY = 0.0
bUnitWeightZ = -(rhoS - rhoF) * 9.81 * np.cos(inclination / 180.0 * pi)

ndm = 3  # Spatial dimensions
ndf = 3  # Degrees of freedom per node for bricks

# ===============================================================
#  MODEL GENERATION
# ===============================================================
ops.model("Basic", "-ndm", ndm, "-ndf", ndf)

# --- PressureDependMultiYield material definition ---
ops.nDMaterial(
    "PressureDependMultiYield",
    solid1,
    ndm,
    rhoS * densityMult,
    G1,
    B1,
    friction,
    0.1,
    80,
    0.5,
    phaseTransform,
    0.17,
    0.4,
    10,
    10,
    0.015,
    1.0,
)

# --- Nodes of the brick element ---
ops.node(1, 0, 0, 0)
ops.node(2, 0, 0, 1)
ops.node(3, 0, 1, 0)
ops.node(4, 0, 1, 1)
ops.node(5, 1, 0, 0)
ops.node(6, 1, 0, 1)
ops.node(7, 1, 1, 0)
ops.node(8, 1, 1, 1)

# --- bbarBrick element ---
ops.element("bbarBrick", 1, 1, 5, 7, 3, 2, 6, 8, 4, solid1, bUnitWeightX, bUnitWeightY, bUnitWeightZ)

# Initially, set the material to "elastic stage"
ops.updateMaterialStage("-material", solid1, "-stage", 0)

# ===============================================================
#  BOUNDARY CONDITIONS
# ===============================================================
# Note: Brick elements use 3 DOFs (UX, UY, UZ). Rotations are not present.

ops.fix(1, 1, 1, 1)
ops.fix(2, 0, 1, 0)
ops.fix(3, 1, 1, 1)
ops.fix(4, 0, 1, 0)
ops.fix(5, 1, 1, 1)
ops.fix(6, 0, 1, 0)
ops.fix(7, 1, 1, 1)
ops.fix(8, 0, 1, 0)

# Equal DOF constraints (to tie boundary nodes)
ops.equalDOF(2, 4, 1, 3)
ops.equalDOF(2, 6, 1, 3)
ops.equalDOF(2, 8, 1, 3)

# %%
fig = opst.vis.pyvista.plot_model()
fig.show()

# %%
# ANALYSIS
# -----------------

# ===============================================================
#  STATIC GRAVITY ANALYSIS (Elastic stage)
# ===============================================================
ops.system("ProfileSPD")
ops.test("NormDispIncr", 1e-10, 25, 2)
ops.constraints("Transformation")
ops.integrator("LoadControl", 1.0, 1, 1, 1)
ops.algorithm("Newton")
ops.numberer("RCM")
ops.analysis("Static")

ops.analyze(2)

# ===============================================================
#  SWITCH MATERIAL TO PLASTIC STAGE
# ===============================================================
ops.updateMaterialStage("-material", solid1, "-stage", 1)
# ops.updateMaterials("-material", solid1, "bulkModulus", G1 * 2.0 / 3.0)
ops.setParameter("-val", G1 * 2.0 / 3.0, "-ele", *[1], "bulkModulus")

ops.analyze(2)

# Reset time for dynamic analysis
ops.setTime(0.0)
ops.wipeAnalysis()


# ===============================================================
#  DYNAMIC ANALYSIS CONFIGURATION
# ===============================================================
# Sinusoidal base excitation: Sine(start, end, period)
ops.timeSeries("Sine", 1, 0.0, 10.0, 1.0, "-factor", accMul)
ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

ops.rayleigh(massProportionalDamping, 0.0, InitStiffnessProportionalDamping, 0.0)

beta = (gamma + 0.5) ** 2 / 4.0
ops.integrator("Newmark", gamma, beta)

ops.constraints("Penalty", 1e18, 1e18)
ops.test("NormDispIncr", 1e-10, 25, 0)
ops.algorithm("ModifiedNewton")
ops.system("ProfileSPD")
ops.numberer("Plain")
ops.analysis("VariableTransient")

# %%
# RUN TIME-HISTORY ANALYSIS
# --------------------------------

startT = time.time()
ODB = opst.post.CreateODB(
    odb_tag=1,
    compute_mechanical_measures=True,
    project_gauss_to_nodes="copy",  # "extrapolate", "copy", "average"
)
for _ in range(numSteps):
    ops.analyze(1, dt)
    ODB.fetch_response_step()
ODB.save_response()
endT = time.time()

print(f"Execution time: {endT - startT:.3f} seconds.")


# %%
#  POST-PROCESSING
# -----------------
# Nodal Response
# +++++++++++++++

node_resp = opst.post.get_nodal_responses(odb_tag=1)
print(node_resp)


# %%
# node 3 displacement relative to node 1

disp1 = node_resp["disp"].sel(nodeTags=1, DOFs="UX")
disp8 = node_resp["disp"].sel(nodeTags=8, DOFs="UX")
times = node_resp["time"].data

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(times, disp8 - disp1, "b")
ax.set_title("Lateral displacement at element top")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
plt.show()

# %%
# Elemental response
# +++++++++++++++++++

ele_resp = opst.post.get_element_responses(odb_tag=1, ele_type="brick")
print("Data Variables in Element Responses:", ele_resp.data_vars)

# %%
print(ele_resp.coords)

# %%
for key, value in ele_resp.attrs.items():
    print(f"{key}: {value}")

# %%
# Gauss point responses
# *************************

sigma11 = ele_resp["Stresses"].sel(stressDOFs="sigma11", eleTags=1)
sigma22 = ele_resp["Stresses"].sel(stressDOFs="sigma22", eleTags=1)
sigma33 = ele_resp["Stresses"].sel(stressDOFs="sigma33", eleTags=1)
sigma12 = ele_resp["Stresses"].sel(stressDOFs="sigma12", eleTags=1)
sigma13 = ele_resp["Stresses"].sel(stressDOFs="sigma13", eleTags=1)
sigma23 = ele_resp["Stresses"].sel(stressDOFs="sigma23", eleTags=1)

# %%
# Calculate confinement p and deviatoric stress q

po = ele_resp["StressMeasures"].sel(measures="sigma_oct", eleTags=1)
qo = ele_resp["StressMeasures"].sel(measures="tau_oct", eleTags=1)
print(qo)

# %%
# strain components
eps11 = ele_resp["Strains"].sel(strainDOFs="eps11", eleTags=1)
eps22 = ele_resp["Strains"].sel(strainDOFs="eps22", eleTags=1)
eps33 = ele_resp["Strains"].sel(strainDOFs="eps33", eleTags=1)
eps12 = ele_resp["Strains"].sel(strainDOFs="eps12", eleTags=1)
eps13 = ele_resp["Strains"].sel(strainDOFs="eps13", eleTags=1)
eps23 = ele_resp["Strains"].sel(strainDOFs="eps23", eleTags=1)

# %%
eps13_ele1_1 = eps13.sel(GaussPoints=1)
sigma13_ele1_1 = sigma13.sel(GaussPoints=1)
po_ele1_1 = po.sel(GaussPoints=1)
qo_ele1_1 = qo.sel(GaussPoints=1) * np.sign(sigma13_ele1_1)

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].plot(eps13_ele1_1, sigma13_ele1_1, "b")
axs[0].set_title("shear stress $\\tau_{xz}$ VS. shear strain $\\epsilon_{xz}$ \n at integration point 1")
axs[0].set_xlabel(r"Shear strain $\epsilon_{xz}$")
axs[0].set_ylabel(r"Shear stress $\tau_{xz}$ (kPa)")
axs[1].plot(-po_ele1_1, qo_ele1_1, "r")
axs[1].set_title("confinement p VS. deviatoric stress q \n at integration point 1")
axs[1].set_xlabel("confinement p (kPa)")
axs[1].set_ylabel("q (kPa)")
plt.show()

# %%
eps13_ele1_1 = eps13.sel(GaussPoints=5)
sigma13_ele1_1 = sigma13.sel(GaussPoints=5)
po_ele1_1 = po.sel(GaussPoints=5)
qo_ele1_1 = qo.sel(GaussPoints=5) * np.sign(sigma13_ele1_1)

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].plot(eps13_ele1_1, sigma13_ele1_1, "b")
axs[0].set_title("shear stress $\\tau_{xz}$ VS. shear strain $\\epsilon_{xz}$ \n at integration point 5")
axs[0].set_xlabel(r"Shear strain $\epsilon_{xz}$")
axs[0].set_ylabel(r"Shear stress $\tau_{xz}$ (kPa)")
axs[1].plot(-po_ele1_1, qo_ele1_1, "r")
axs[1].set_title("confinement p VS. deviatoric stress q \n at integration point 5")
axs[1].set_xlabel("confinement p (kPa)")
axs[1].set_ylabel("q (kPa)")
plt.show()

# %%
