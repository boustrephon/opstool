r"""
Shell Element Responses
=======================
"""

# %%
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.plotly as opsvis

# %%
# Model and gravity load
# ------------------------
opst.load_ops_examples("Shell3D")

ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
_ = opst.pre.gen_grav_load(direction="Z", factor=-9810)

# %%
opsvis.set_plot_props(point_size=0, line_width=3)  # notebook=False for practical use
fig = opsvis.plot_model(show_nodal_loads=True, show_ele_loads=True, show_outline=True)
fig
# fig.show()

# %%
# Gravity analysis
# -----------------
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-8, 6, 2)
ops.algorithm("Linear")
ops.integrator("LoadControl", 0.1)
ops.analysis("Static")

# %%
# Save the responses
ODB = opst.post.CreateODB(
    odb_tag=1,
    project_gauss_to_nodes="copy",  # project gauss point responses to nodes, optional ["copy", "average", "extrapolate"]
)
for _ in range(10):
    ops.analyze(1)
    ODB.fetch_response_step()
ODB.save_response()


# %%
# Visualize the results
# ------------------------
# Nodal responses, ``project_gauss_to_nodes`` needs to be set to "copy", "average", or "extrapolate" when creating the ODB
opsvis.set_plot_props(cmap="plasma", show_mesh_edges=True)

fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    ele_type="Shell",
    resp_type="sectionForcesAtNodes",  # nodal response, "AtNodes"
    resp_dof="FXX",
)
# fig.show()
fig

# %%
# Display the responses at each element, all gauss points will be averaged to the element level.
fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=True,
    ele_type="Shell",
    resp_type="sectionForces",  # element response, "AtGaussPoints", will be averaged to each element
    resp_dof="FXX",
)
# fig.show()
fig

# %%
# Fiber point stress can be plotted as well, but it requires a ``shell_fiber_loc`` to be assigned.
# sphinx_gallery_thumbnail_number = 5
fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    ele_type="Shell",
    resp_type="StressesAtNodes",  # nodal stress response, "AtNodes"
    resp_dof="sigma11",  # sigma11, sigma22, sigma12, sigma13, sigma23
    shell_fiber_loc="top",  # shell_fiber_loc can be "top", "bottom", or "mid" for shell elements, also int
)
# fig.show()
fig
