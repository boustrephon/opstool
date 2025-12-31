"""
Soil-Structure Interaction
============================

This example originates from the GitHub repository maintained by Professor Quan Gu of Xiamen University.
`OpenSeesXMU <https://github.com/OpenSeesXMU/OpenSees-examples-Chinese/tree/master/1.5-SSI>`_
"""

# %%
# The TCL script has been translated into a Python script using
# `opstool.pre.tcl2py <https://opstool.readthedocs.io/en/latest/src/api/_autosummary/opstool.pre.tcl2py.html#opstool.pre.tcl2py>`_
# For details, see:
# `model.py <https://github.com/yexiang1992/opstool/blob/master/docs/examples/post/SSI-GUQUAN/model.py>`_
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from utils.SSI_Gu_model import Model

import opstool as opst
import opstool.vis.pyvista as opsvis

Model()

# %%
# Visualize the model
# -----------------------
fig = opsvis.plot_model()
fig.show()

# %%
# Gravity analysis
# -----------------------
ops.constraints("Transformation")
ops.numberer("RCM")
ops.test("NormDispIncr", 1e-06, 25, 2)
ops.integrator("LoadControl", 1, 1, 1, 1)
ops.algorithm("Newton")
ops.system("BandGeneral")
ops.analysis("Static")
ops.analyze(3)
print("soil gravity nonlinear analysis completed ...")

# %%
# Earthquake analysis
# -----------------------
ops.timeSeries("Path", 1, "-dt", 0.01, "-filePath", "utils/elcentro.txt", "-factor", 3)
ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

# %%
ops.wipeAnalysis()
ops.constraints("Transformation")
ops.test("NormDispIncr", 1e-06, 25)
ops.algorithm("Newton")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.integrator("Newmark", 0.55, 0.275625)
ops.analysis("Transient")

# %%
ODB = opst.post.CreateODB(
    odb_tag=1,
    save_every=500,  # save response every n steps
    compute_mechanical_measures=True,  # compute stress measures, strain measures, etc.
    project_gauss_to_nodes="copy",  # project gauss point responses to nodes, optional ["copy", "average", "extrapolate"]
)  # Create ODB object

for _ in range(2400):
    ops.analyze(1, 0.005)
    ODB.fetch_response_step()  # Fetch response for the current step
ODB.save_response()  # Save response

# %%
# Post-processing
# -----------------------
# Frame Element Response
# ++++++++++++++++++++++++++

FrameResp = opst.post.get_element_responses(odb_tag=1, ele_type="Frame")

# %%
f = FrameResp["sectionForces"].sel(eleTags=7, secDofs="MZ", secPoints=1)
d = FrameResp["sectionDeformations"].sel(eleTags=7, secDofs="MZ", secPoints=1)

plt.plot(d, f)
plt.show()

# %%
# Nodal response
# +++++++++++++++

NodalResp = opst.post.get_nodal_responses(odb_tag=1)

# %%
time = NodalResp.time
disp = NodalResp["disp"].sel(nodeTags=1, DOFs="UX")

plt.plot(time, disp)
plt.show()

# %%
# Plane Soil Element
# ++++++++++++++++++++++++++

PlaneResp = opst.post.get_element_responses(odb_tag=1, ele_type="Plane")
print(PlaneResp.coords)

# %%
print(PlaneResp["Stresses"])

# %%
s = PlaneResp["Stresses"].sel(eleTags=37, GaussPoints=1, stressDOFs="sigma12")
e = PlaneResp["Strains"].sel(eleTags=37, GaussPoints=1, strainDOFs="eps12")

plt.plot(e, s, c="blue")
plt.xlabel("eps12")
plt.ylabel("sigma12")
plt.title("Stress-Strain Curve for Plane Element 37 at Gauss Point 1")
plt.show()

# %%
# Plotting the nodal responses with deformed shape
# +++++++++++++++++++++++++++++++++++++++++++++++++++

opsvis.set_plot_props(
    cmap="Spectral_r",
    point_size=0.0,
    scalar_bar_kargs={"title_font_size": 12, "label_font_size": 12, "position_x": 0.865},
    show_mesh_edges=False,
)

# %%
fig = opst.vis.pyvista.plot_nodal_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    resp_type="disp",
    resp_dof=("UX", "UY"),
    show_defo=True,
    defo_scale=5,
    show_undeformed=True,
)
fig.show()

# %%
# Plotting the plane element stresses
# +++++++++++++++++++++++++++++++++++++++++++++++++++
fig = opst.vis.pyvista.plot_unstruct_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    ele_type="Plane",
    resp_type="StressesAtNodes",
    resp_dof="sigma_oct",  # hydrostatic pressure
    show_defo=True,
    defo_scale="auto",
    show_model=True,
)
fig.show()

# %%
# Plotting the deformation animation
# +++++++++++++++++++++++++++++++++++++++++++++++++++

# sphinx_gallery_thumbnail_number = 7
fig = opst.vis.pyvista.plot_nodal_responses_animation(
    odb_tag=1,
    framerate=50,  # Frames per second
    resp_type="disp",
    resp_dof=("UX", "UY"),
    show_defo=True,
    defo_scale=5,
    savefig="images/nodal_disp_animation_ssi.mp4",
)
fig.show()

# %%
# Clean up
fig.close()

# %%
fig = opst.vis.pyvista.plot_unstruct_responses_animation(
    odb_tag=1,
    framerate=50,  # Frames per second
    ele_type="Plane",
    resp_type="StressesAtNodes",
    resp_dof="sigma_oct",
    show_defo=True,
    defo_scale=10,
    show_model=True,
    savefig="images/stress_animation_ssi.mp4",
)
fig.show()

# %%
# Clean up
fig.close()
