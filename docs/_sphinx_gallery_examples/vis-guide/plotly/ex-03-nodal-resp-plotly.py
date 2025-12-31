r"""
Nodal Responses Visualization
=============================
"""

# %%
import openseespy.opensees as ops

import opstool as opst
import opstool.vis.plotly as opsvis

# %%
# Here, we use a built-in example from ``opstool``, which is an example of a deck arch bridge model primarily composed of frame elements and shell elements.

# %%
opst.load_ops_examples("ArchBridge2")
# or your model code here

# %%
# We use the :func:`opstool.vis.plotly.set_plot_props` function to predefine some common visualization properties, which will affect all subsequent visualizations of models, eigenvalues, and responses.

# %%
# Model Geometry
# ---------------
opsvis.set_plot_props(point_size=0, line_width=3)

fig = opsvis.plot_model(show_outline=True)
fig
# fig.show()

# %%
# Gravity Analysis
# ----------------
# Apply the gravity load according to the mass in the model:
ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
_ = opst.pre.gen_grav_load(factor=-9810)

# %%
# Analysis Parameters:
ops.system("BandGeneral")
# Create the constraint handler, the transformation method
ops.constraints("Transformation")
# Create the DOF numberer, the reverse Cuthill-McKee algorithm
ops.numberer("RCM")
# Create the convergence test, the norm of the residual with a tolerance of
# 1e-12 and a max number of iterations of 10
ops.test("NormDispIncr", 1.0e-12, 10, 3)
# Create the solution algorithm, a Newton-Raphson algorithm
ops.algorithm("Newton")
# Create the integration scheme, the LoadControl scheme using steps of 0.1
ops.integrator("LoadControl", 0.1)
# Create the analysis object
ops.analysis("Static")

# %%
# Analysis and Saving Results

ODB = opst.post.CreateODB(odb_tag=1)
for i in range(10):
    ops.analyze(1)
    ODB.fetch_response_step()
ODB.save_response()

# %%
# Nodal Responses Visualization
# -----------------------------
# via Slides
# ~~~~~~~~~~~
opsvis.set_plot_props(point_size=3, line_width=2)
fig = opsvis.plot_nodal_responses(
    odb_tag=1, slides=True, defo_scale="auto", resp_type="disp", resp_dof=["UX", "UY", "UZ"], unit_symbol="mm"
)
fig
# fig.show()

# %%
# Change the unit dispaly
# ~~~~~~~~~~~~~~~~~~~~~~~~
fig = opsvis.plot_nodal_responses(
    odb_tag=1,
    slides=False,
    resp_type="disp",
    resp_dof=["UX", "UY", "UZ"],
    defo_scale=100,
    unit_symbol="m",
    unit_factor=1e-3,
)
fig
# fig.show()

# %%
# Animation
# ~~~~~~~~~~~~~
fig = opsvis.plot_nodal_responses_animation(
    odb_tag=1, framerate=2, resp_type="disp", resp_dof=["UX", "UY", "UZ"], defo_scale=100, unit_symbol="mm"
)
fig
# fig.show()
