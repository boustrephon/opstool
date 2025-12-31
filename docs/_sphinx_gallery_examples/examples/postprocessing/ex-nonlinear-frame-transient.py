r"""
3D Nonlinear beam-column elements Gravity load analysis followed by transient analysis
==========================================================================================

Reinforced concrete one-bay, three-story frame Distributed vertical load on girder

Example Objectives:

3D building with rigid diaphragms
Nonlinear beam-column ops.elements
Gravity load analysis followed by transient analysis

Units: kips, in, sec

Examples code see
`OpenSees Example5.1.py <https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/ExamplePython/Example5.1.py>`_
"""

# %%
# Start of model generation
# --------------------------
import numpy as np
import openseespy.opensees as ops
import utils._RCsection as RCsection  # local module for RC section properties

import opstool
import opstool.vis.plotly as opsvis

# remove existing model
ops.wipe()

# create ModelBuilder (with three-dimensions and 6 DOF/node)
ops.model("BasicBuilder", "-ndm", 3, "-ndf", 6)

# set default units
ops.defaultUnits("-force", "kip", "-length", "in", "-time", "sec", "-temp", "F")

# %%
# Define geometry
# ++++++++++++++++

# Set parameters for model geometry
h = 144.0  # Story height
by = 240.0  # Bay width in Y-direction
bx = 240.0  # Bay width in X-direction

# Create nodes
#       tag    X        Y        Z
ops.node(1, -bx / 2.0, by / 2.0, 0.0)
ops.node(2, bx / 2.0, by / 2.0, 0.0)
ops.node(3, bx / 2.0, -by / 2.0, 0.0)
ops.node(4, -bx / 2.0, -by / 2.0, 0.0)

ops.node(5, -bx / 2.0, by / 2.0, h)
ops.node(6, bx / 2.0, by / 2.0, h)
ops.node(7, bx / 2.0, -by / 2.0, h)
ops.node(8, -bx / 2.0, -by / 2.0, h)

ops.node(10, -bx / 2.0, by / 2.0, 2.0 * h)
ops.node(11, bx / 2.0, by / 2.0, 2.0 * h)
ops.node(12, bx / 2.0, -by / 2.0, 2.0 * h)
ops.node(13, -bx / 2.0, -by / 2.0, 2.0 * h)

ops.node(15, -bx / 2.0, by / 2.0, 3.0 * h)
ops.node(16, bx / 2.0, by / 2.0, 3.0 * h)
ops.node(17, bx / 2.0, -by / 2.0, 3.0 * h)
ops.node(18, -bx / 2.0, -by / 2.0, 3.0 * h)

# Retained nodes for rigid diaphragm
#        tag   X    Y    Z
ops.node(9, 0.0, 0.0, h)
ops.node(14, 0.0, 0.0, 2.0 * h)
ops.node(19, 0.0, 0.0, 3.0 * h)

# Set base constraints
#      tag DX DY DZ RX RY RZ
ops.fix(1, 1, 1, 1, 1, 1, 1)
ops.fix(2, 1, 1, 1, 1, 1, 1)
ops.fix(3, 1, 1, 1, 1, 1, 1)
ops.fix(4, 1, 1, 1, 1, 1, 1)

# Define rigid diaphragm multi-point constraints
#              normalDir retained constrained
ops.rigidDiaphragm(3, 9, 5, 6, 7, 8)
ops.rigidDiaphragm(3, 14, 10, 11, 12, 13)
ops.rigidDiaphragm(3, 19, 15, 16, 17, 18)

# Constraints for rigid diaphragm retained nodes
#      tag DX DY DZ RX RY RZ
ops.fix(9, 0, 0, 1, 1, 1, 0)
ops.fix(14, 0, 0, 1, 1, 1, 0)
ops.fix(19, 0, 0, 1, 1, 1, 0)

# %%
# Define materials for nonlinear columns
# ---------------------------------------
# CONCRETE
fc = 4.0
Ec = 57000.0 * np.sqrt(fc * 1000.0) / 1000.0
# Core concrete (confined)
#                                 tag  f'c   epsc0  f'cu  epscu
ops.uniaxialMaterial("Concrete01", 1, -5.0, -0.005, -3.5, -0.02)

# Cover concrete (unconfined)
#                                 tag  f'c   epsc0  f'cu  epscu
ops.uniaxialMaterial("Concrete01", 2, -fc, -0.002, 0.0, -0.006)

# STEEL
fy = 60.0  # Yield stress
Es = 30000.0  # Young's modulus
# Reinforcing steel
#                              tag fy  E0  b
ops.uniaxialMaterial("Steel01", 3, fy, Es, 0.02)

# Column parameters
h = 18.0
GJ = 1.0e10
colSec = 1

# Call the RCsection procedure to generate the column section
#                        id  h  b cover core cover steel nBars barArea nfCoreY nfCoreZ nfCoverY nfCoverZ GJ
RCsection.create(colSec, h, h, 2.5, 1, 2, 3, 3, 0.79, 8, 8, 10, 10, GJ)

# %%
# Define column ops.elements
# ----------------------------
PDelta = "OFF"
# PDelta = "ON"

# Geometric transformation for columns
if PDelta == "OFF":
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
else:
    ops.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)

# Number of column integration points (sections)
np = 4
ops.beamIntegration("Lobatto", colSec, colSec, np)

# Create the nonlinear column elements
eleType = "forceBeamColumn"
#                   tag ndI ndJ transfTag integrationTag
ops.element(eleType, 1, 1, 5, 1, colSec)
ops.element(eleType, 2, 2, 6, 1, colSec)
ops.element(eleType, 3, 3, 7, 1, colSec)
ops.element(eleType, 4, 4, 8, 1, colSec)

ops.element(eleType, 5, 5, 10, 1, colSec)
ops.element(eleType, 6, 6, 11, 1, colSec)
ops.element(eleType, 7, 7, 12, 1, colSec)
ops.element(eleType, 8, 8, 13, 1, colSec)

ops.element(eleType, 9, 10, 15, 1, colSec)
ops.element(eleType, 10, 11, 16, 1, colSec)
ops.element(eleType, 11, 12, 17, 1, colSec)
ops.element(eleType, 12, 13, 18, 1, colSec)

# Define beam ops.elements
# --------------------------
# Define material properties for elastic beams
# Using beam depth of 24 and width of 18
Abeam = 18.0 * 24.0
# "Cracked" second moments of area
Ibeamzz = 0.5 * 1.0 / 12.0 * 18.0 * pow(24.0, 3)
Ibeamyy = 0.5 * 1.0 / 12.0 * 24.0 * pow(18.0, 3)
beamSec = 2

# Define elastic section for beams
#                       tag     E    A      Iz       Iy     G    J
ops.section("Elastic", beamSec, Ec, Abeam, Ibeamzz, Ibeamyy, GJ, 1.0)

# Geometric transformation for beams
ops.geomTransf("Linear", 2, 1.0, 1.0, 0.0)

# Number of beam integration points (sections)
np = 3
ops.beamIntegration("Lobatto", beamSec, beamSec, np)

# Create the beam ops.elements
eleType = "forceBeamColumn"
#                   tag ndI ndJ transfTag integrationTag
ops.element(eleType, 13, 5, 6, 2, beamSec)
ops.element(eleType, 14, 6, 7, 2, beamSec)
ops.element(eleType, 15, 7, 8, 2, beamSec)
ops.element(eleType, 16, 8, 5, 2, beamSec)

ops.element(eleType, 17, 10, 11, 2, beamSec)
ops.element(eleType, 18, 11, 12, 2, beamSec)
ops.element(eleType, 19, 12, 13, 2, beamSec)
ops.element(eleType, 20, 13, 10, 2, beamSec)

ops.element(eleType, 21, 15, 16, 2, beamSec)
ops.element(eleType, 22, 16, 17, 2, beamSec)
ops.element(eleType, 23, 17, 18, 2, beamSec)
ops.element(eleType, 24, 18, 15, 2, beamSec)

# %%
# Define gravity loads
# ---------------------

# Gravity load applied at each corner node
# 10% of column capacity
p = 0.1 * fc * h * h
g = 386.09

# Mass lumped at retained nodes
m = (4.0 * p) / g

# Rotary inertia of floor about retained node
i = m * (bx * bx + by * by) / 12.0

# Set mass at the retained nodes
#        tag MX MY MZ   RX   RY   RZ
ops.mass(9, m, m, 0.0, 0.0, 0.0, i)
ops.mass(14, m, m, 0.0, 0.0, 0.0, i)
ops.mass(19, m, m, 0.0, 0.0, 0.0, i)

# Define gravity loads
# create a Constant TimeSeries
ops.timeSeries("Constant", 1)
# create a Plain load pattern
ops.pattern("Plain", 1, 1, "-fact", 1.0)

for i in [5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]:
    ops.load(i, 0.0, 0.0, -p, 0.0, 0.0, 0.0)

# %%
fig = opsvis.plot_model(show_nodal_loads=True)
fig

# %%
fig = opsvis.plot_eigen(mode_tags=[1, 4], subplots=False)
fig

# %%
# Dynamic Loads
# ----------------

# set rayleigh damping factors
ops.rayleigh(0.0, 0.0, 0.0, 0.0018)

# Define earthquake excitation
# ----------------------------
dt = 0.02
# Set up the acceleration records for Tabas fault normal and fault parallel
ops.timeSeries("Path", 2, "-filePath", "utils/tabasFN.txt", "-dt", dt, "-factor", g)
ops.timeSeries("Path", 3, "-filePath", "utils/tabasFP.txt", "-dt", dt, "-factor", g)

# Define the excitation using the Tabas ground motion records
#                         tag dir         accel series args
ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
ops.pattern("UniformExcitation", 3, 2, "-accel", 3)

# %%
# Smart Analysis
# --------------

# create the system of equation
ops.system("UmfPack")
# create the DOF numberer
ops.numberer("Plain")
# create the constraint handler
ops.constraints("Transformation")
# create the convergence test
ops.test("EnergyIncr", 1.0e-8, 20)
# create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm("Newton")
# create the integration scheme, the Newmark with gamma=0.5 and beta=0.25
ops.integrator("Newmark", 0.5, 0.25)
# create the analysis object
ops.analysis("Transient")

# %%
ODB = opstool.post.CreateODB(odb_tag=1, save_every=500, interpolate_beam_disp=11)  # Create ODB object

analysis = opstool.anlys.SmartAnalyze(
    "Transient",
    tryAddTestTimes=True,  # add test times to the analysis
    testIterTimesMore=[50, 100],
    tryAlterAlgoTypes=True,  # try different algorithms
    algoTypes=[40, 10, 20, 30],  # algorithm types to try
    minStep=1e-6,  # minimum step size for substepping
    debugMode=True,  # False for progress bar, True for debug info
)
segs = analysis.transient_split(npts=2000)  # Tell the analysis how to split the steps, and how many steps to take

for _ in segs:
    analysis.TransientAnalyze(dt=0.01)
    ODB.fetch_response_step()  # fetch response for the current step
ODB.save_response()  # save response to ODB
analysis.close()  # Close the analysis object

# %%
# Post-processing
# ----------------
# print info
node_info = opstool.post.get_nodal_responses_info()
ele_info = opstool.post.get_element_responses_info(ele_type="Frame")


# %%
# Get nodal responses

nodal_resp = opstool.post.get_nodal_responses(odb_tag=1)
nodal_resp

# %%
# Get frame element responses

frame_resp = opstool.post.get_element_responses(odb_tag=1, ele_type="Frame")
frame_resp

# %%
# Visualization with Plotly
# ******************************

opsvis.set_plot_props(point_size=3.0)
opsvis.set_plot_colors(frame="gray")

# %%
fig = opsvis.plot_frame_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    resp_type="sectionDeformations",
    resp_dof="My",
    unit_symbol="N·m",
    scale=3.0,
    show_values=True,  # hover to show values
    line_width=5,
    show_bc=True,
    bc_scale=2,
    style="surface",
    opacity=1.0,
)
# fig.show()  # for auto
# fig.write_html("sectionDeformations.html", full_html=False, include_plotlyjs="cdn")
fig

# %%
framerate = int(2000 / 20)  # Set framerate for animation, 2000 steps, 20 seconds

fig = opsvis.plot_nodal_responses_animation(
    odb_tag=1,
    framerate=framerate,
    resp_type="disp",
    resp_dof=["UX", "UY", "UZ"],
    defo_scale=3,
    interpolate_beam_disp=True,
    unit_symbol="in",
    show_bc=True,
    bc_scale=5,
)
# fig.show()
# fig.write_html("nodal_responses_animation.html", full_html=False, include_plotlyjs="cdn")
fig

# %%
# Visualization with PyVista
# ******************************

# %%
fig = opstool.vis.pyvista.plot_nodal_responses(
    odb_tag=1,
    slides=False,
    resp_type="disp",
    resp_dof=["UX", "UY", "UZ"],
    defo_scale=5,
    interpolate_beam_disp=True,
    unit_symbol="in",
    show_bc=True,
    bc_scale=5,
)
fig.show()

# %%
framerate = int(2000 / 20)  # Set framerate for animation, 2000 steps, 20 seconds

fig = opstool.vis.pyvista.plot_nodal_responses_animation(
    odb_tag=1,
    framerate=framerate,
    savefig="images/NodalRespAnimation.mp4",  # or ".mp4" (recommended)
    resp_type="disp",
    resp_dof=["UX", "UY", "UZ"],
    defo_scale=3,
    interpolate_beam_disp=True,
    unit_symbol="in",
    show_bc=True,
    bc_scale=5,
)
fig.show()

# %%
# Don't forget to close the figure
fig.close()
