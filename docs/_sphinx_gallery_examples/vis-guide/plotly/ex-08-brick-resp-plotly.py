r"""
Solid Element Responses
========================
"""

# %%
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.plotly as opsvis

# %%
opst.load_ops_examples("Dam-Brick")  # or your own model

# Create a gravity load pattern
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
node_loads = opst.pre.create_gravity_load(direction="Z", factor=-9.81)

# %%
fig = opsvis.plot_model(show_nodal_loads=True)
fig
# fig.show()
# fig.write_html("**.html")

# %%
# Define the analysis parameters
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-6, 6, 2)
ops.algorithm("Linear")
ops.integrator("LoadControl", 0.1)
ops.analysis("Static")

# %%
# Perform the analysis and record the responses
ODB = opst.post.CreateODB(
    odb_tag=1,
    compute_mechanical_measures=True,  # compute stress measures, strain measures, etc.
    project_gauss_to_nodes="copy",  # project gauss point responses to nodes, optional ["copy", "average", "extrapolate"]
    interpolate_beam_disp=False,
)
for _ in range(10):
    ops.analyze(1)
    ODB.fetch_response_step()
ODB.save_response()

# %%
# Visualize the brick element responses using Plotly

opsvis.set_plot_props(point_size=0.0)

fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=True,
    ele_type="Brick",
    resp_type="stressesAtNodes",  # or "stressesAtGauss", "strainsAtNodes", project_gauss_to_nodes needs to be set prior
    resp_dof="sigma_vm",
    show_values=True,
    unit_symbol="kPa",
    show_defo=True,
    defo_scale="auto",
    show_model=True,
)
fig
# fig.show()

# %%
fig = opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=False,
    step="absMax",
    ele_type="Brick",
    resp_type="stresses",  # stresses at gauss points will be averaged to rach element
    resp_dof="sigma_vm",
    show_values=True,
    unit_symbol="kPa",
    show_defo=True,
    defo_scale="auto",  # "auto"
    show_model=False,
)
fig
# fig.show()

# %%
fig = opsvis.plot_unstruct_responses_animation(
    odb_tag=1,
    framerate=4,
    ele_type="Brick",
    resp_type="stressesAtNodes",  # or "stressesAtGauss", "strainsAtNodes"
    resp_dof="sigma_vm",
    show_values=False,
    unit_symbol="kPa",
    show_defo=True,
    defo_scale=20000,
    show_model=False,
)
fig
# fig.show()
