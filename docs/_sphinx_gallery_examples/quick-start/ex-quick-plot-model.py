r"""
Quick Model and Eigen Visualization
======================================

At any point during model creation, you can run the :func:`opstool.vis.plotly.plot_model` or :func:`opstool.vis.pyvista.plot_model` to visualize the current model's geometric details.

Although both are based on different engines, ``PyVista`` and ``Plotly``,
they provide almost identical APIs.
Therefore, you can choose either for visualization.
"""


# %%
# Plotly-based Visualization
# -------------------------------

import opstool as opst
import opstool.vis.plotly as opsvis

opst.load_ops_examples("ArchBridge2")  # Built-in example model in opstool
# or your model code here

# %%
# Here, we use a built-in example from ``opstool``, which is an example of a deck arch bridge model primarily composed of frame elements and shell elements.
# For your own model script, simply place the :func:`opstool.vis.plotly.plot_model` function anywhere you need visualization.

# %%
# We use the :func:`opstool.vis.plotly.set_plot_props` function to predefine some common visualization properties, which will affect all subsequent visualizations of models, eigenvalues, and responses.
opsvis.set_plot_props(point_size=0, line_width=3)

# %%
# Model Geometry Visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Visualization is straightforward; simply call the :func:`opstool.vis.plotly.plot_model` function.
# For details on various parameters, refer to the :func:`opstool.vis.plotly.plot_model`.

fig = opsvis.plot_model(show_outline=True)
fig
# fig.show(renderer="browser")  # for interactive use

# %%
# The function :func:`opstool.vis.plotly.plot_model` returns a `Plotly Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_ object, which you can visualize using the `show` method.
# In this example, the rendering mode is set to `notebook`, allowing the figure to be embedded directly within a Jupyter Notebook.
# If you want it to automatically choose the rendering mode, simply call `show` without providing any arguments.
# For more details on the available `renderer`, please refer to `Displaying Figures in Python <https://plotly.com/python/renderers/>`_

# %%
# Eigenmode Visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Subplots
# +++++++++
fig = opsvis.plot_eigen(mode_tags=5, subplots=True, show_outline=False)
fig
# fig.show(renderer="browser")  # for auto

# %%
# Slider
# +++++++++
fig = opsvis.plot_eigen(mode_tags=5, subplots=False, show_outline=False)
fig
# fig.show(renderer="browser")  # for automatic use

# %%
# Pyvista-based Visualization
# -------------------------------

import opstool as opst
import opstool.vis.pyvista as opsvis

opst.load_ops_examples("Frame3D")
# or your model code here


# %%
# Model Geometry Visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Model Geometry Visualization
opsvis.set_plot_props(point_size=0, line_width=3)

plotter = opsvis.plot_model()
plotter.show()  # for interactive use


# %%
# Eigenmode Visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

plotter = opsvis.plot_eigen(mode_tags=6, subplots=True)
plotter.show()  # for interactive use
