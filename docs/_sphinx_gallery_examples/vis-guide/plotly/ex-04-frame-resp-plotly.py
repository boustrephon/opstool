r"""
Frame Element Responses
=======================
"""

# %%
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.plotly as opsvis

# %%
opst.load_ops_examples("Frame3D2")
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
for etag in [5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21]:
    ops.eleLoad("-ele", etag, "-type", "-beamUniform", 0.0, -10)  # wy=0.0, wz=-10.0

# %%
opsvis.set_plot_props(point_size=0, line_width=3)
fig = opsvis.plot_model(show_nodal_loads=True, show_ele_loads=True, show_local_axes=True, show_ele_numbering=True)
fig
# fig.show()

# %%
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-8, 6, 2)
ops.algorithm("Linear")
ops.integrator("LoadControl", 0.1)
ops.analysis("Static")

# %%
ODB = opst.post.CreateODB(odb_tag=1)
for i in range(10):
    ops.analyze(1)
    ODB.fetch_response_step()
ODB.save_response()

# %%
# sphinx_gallery_thumbnail_number = 2
opsvis.set_plot_props(point_size=3.0)
opsvis.set_plot_colors(frame="gray")

fig = opsvis.plot_frame_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    resp_type="sectionForces",
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
fig
# fig.show(renderer="jupyterlab")
# fig.show(renderer="notebook")
# fig.show(renderer="browser")
# fig.show()  # for auto
# fig.write_html("**.html")

# %%
fig = opsvis.plot_frame_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    resp_type="sectionForces",
    resp_dof="Mz",
    unit_symbol="N·m",
    scale=3.0,
    show_values=True,  # hover to show values
    line_width=5,
    show_bc=True,
    bc_scale=2,
    style="surface",
    opacity=1.0,
)
fig
# fig.show(renderer="jupyterlab")
# fig.show(renderer="notebook")
# fig.show(renderer="browser")
# fig.show()  # for auto
# fig.write_html("**.html")

# %%
fig = opsvis.plot_frame_responses(
    odb_tag=1,
    slides=True,
    resp_type="sectionForces",
    resp_dof="My",
    unit_symbol="MN·m",
    unit_factor=1e-6,
    scale=2.0,
    show_values=True,  # hover to show values
    line_width=5,
    show_bc=True,
    bc_scale=2,
    style="wireframe",
    color="blue",
)
fig
# fig.show(renderer="jupyterlab")
# fig.show(renderer="notebook")
# fig.show(renderer="browser")
# fig.show()  # for auto

# %%
opsvis.set_plot_props(point_size=0.0)

fig = opsvis.plot_frame_responses_animation(
    odb_tag=1,
    framerate=2,  # frames per second
    resp_type="sectionForces",
    resp_dof="My",
    unit_symbol="kN·m",
    unit_factor=1e-3,
    scale=2.0,
    show_values=False,  # hover to show values
    line_width=5,
    show_bc=True,
    bc_scale=2,
    style="wireframe",
)
fig
# fig.show()
