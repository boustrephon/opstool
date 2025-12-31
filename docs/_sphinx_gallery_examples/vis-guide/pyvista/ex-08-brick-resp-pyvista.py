r"""
Solid Element Responses
=======================
"""

# %%
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.pyvista as opsvis

# %%
opst.load_ops_examples("Dam-Brick")

ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
_ = opst.pre.gen_grav_load(direction="Z", factor=-9.81)

# %%
# Model
# -------
opsvis.set_plot_props(point_size=0, line_width=3)
fig = opsvis.plot_model(show_nodal_loads=True, show_ele_loads=True, show_outline=True)
fig.show()

# %%
# Results visualization
# ------------------------

ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-12, 6, 2)
ops.algorithm("Linear")
ops.integrator("LoadControl", 0.1)
ops.analysis("Static")


# %%
ODB = opst.post.CreateODB(
    odb_tag=1,
    compute_mechanical_measures=True,  # compute stress measures, strain measures, etc.
    project_gauss_to_nodes="copy",  # project gauss point responses to nodes, optional ["copy", "average", "extrapolate"]
)
for _ in range(10):
    ops.analyze(1)
    ODB.fetch_response_step()
ODB.save_response()

# %%
opsvis.set_plot_props(cmap="Spectral_r", point_size=0.0)

fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    ele_type="Brick",
    resp_type="StressesAtNodes",  # or "stressesAtGauss", "strainsAtNodes", project_gauss_to_nodes needs to be set prior
    resp_dof="sigma_vm",
    show_defo=True,
    defo_scale="auto",
    show_model=True,
)
fig.show()

# %%
fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=True,
    ele_type="Brick",
    resp_type="stresses",  # at Gauss points, it will be averaged over the element
    resp_dof="sigma_vm",
    show_model=False,
    show_defo=True,
    defo_scale="auto",
)
fig.show()

# %%
fig = opsvis.plot_unstruct_responses_animation(
    odb_tag=1,
    ele_type="Brick",
    resp_type="stressesAtNodes",  # at nodes
    resp_dof="sigma_vm",
    savefig="images/BrickRespAnimation.gif",
    framerate=2,
    show_model=True,
    show_defo=True,
    defo_scale="auto",
)
fig.close()

# %%
# Interacting with Pyvista
# ----------------------------
# Since version 1.0.18, opstool provides a function ``get_unstruct_responses_dataset``
# that returns a pyvista `UnstructuredGrid <https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgrid#pyvista.UnstructuredGrid>`_
# so that you can take advantage of all the functionality on it.

import pyvista as pv

ugrid = opsvis.get_unstruct_responses_dataset(
    odb_tag=1, step="absMax", ele_type="Brick", resp_type="stressesAtNodes", resp_dof="sigma_vm", defo_scale=0.0
)
print(ugrid)
print(ugrid.active_scalars_name)

# %%
ugrid["StressMeasuresAtNodes"]

# %%
ugrid.plot()

# %%
# Plot on line
# ~~~~~~~~~~~~~~~~~
pa = (0, 30, 0)
pb = (0, 40, 30)
# Preview how this line intersects this mesh
line = pv.Line(pa, pb)

p = pv.Plotter()
p.add_mesh(ugrid, style="wireframe", color="w")
p.add_mesh(line, color="b")
p.show()

# %%
ugrid.plot_over_line(pa, pb)

# %%
# Thresholding
# ~~~~~~~~~~~~~~~~~~~~~~
threshed = ugrid.threshold([300, 600])

# %%

# sphinx_gallery_thumbnail_number = 8
p = pv.Plotter()
p.add_mesh(threshed, style="wireframe", cmap="Spectral_r")
p.camera_position = [-2, 5, 3]
p.show()
