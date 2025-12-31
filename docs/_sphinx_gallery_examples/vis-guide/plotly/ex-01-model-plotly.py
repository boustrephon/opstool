r"""
Model Geometry
=======================
"""

# %%
# A Bridge Example
# ----------------
# At any point during model creation, you can run the :func:`opstool.vis.plotly.plot_model` function to visualize the current model's geometric details.
# This function operates based on post-processed data saved during the modeling process, ensuring accurate visualization.
#
# You can use :func:`opstool.vis.plotly.plot_model` anywhere within your model script.
# It reads the predefined data, such as nodes and elements, and generates a visual representation of the model.

import opstool as opst
import opstool.vis.plotly as opsvis

opst.load_ops_examples("ArchBridge2")
# or your model code here

# %%
# Here, we use a built-in example from ``opstool``, which is an example of a deck arch bridge model primarily composed of frame elements and shell elements.
# For your own model script, simply place the :func:`opstool.vis.plotly.plot_model` function anywhere you need visualization.

# %%
# QUick visualization
# ------------------------
fig = opsvis.plot_model()
fig
# fig.show(renderer="jupyterlab")
# fig.show(renderer="notebook")
# fig.show(renderer="browser")
# fig.show()  # for auto
# fig.write_html("**.html")

# %%
# Saving Model Data (Optional)
# -----------------------------
# We can first save the model geometry data by assigning a unique label to the result database using the parameter **odb_tag**.
# This label is used to identify the data.
# For more details, see :func:`opstool.post.save_model_data`.
opst.post.save_model_data(odb_tag=1)

# %%
# Visualization
# ~~~~~~~~~~~~~~~
# We use the :func:`opstool.vis.plotly.set_plot_props` function to predefine some common visualization properties, which will affect all subsequent visualizations of models, eigenvalues, and responses.
opsvis.set_plot_props(point_size=0, line_width=3)

# %%
# Visualization is straightforward; simply call the :func:`opstool.vis.plotly.plot_model` function.
# The **odb_tag** parameter helps you identify which dataset to use.
# For details on various parameters, refer to the :func:`opstool.vis.plotly.plot_model`.

fig = opsvis.plot_model(odb_tag=1, show_outline=True)
fig
# fig.show(renderer="jupyterlab")
# fig.show(renderer="notebook")
# fig.show(renderer="browser")
# fig.show()  # for auto
# fig.write_html("**.html")

# %%
# The function :func:`opstool.vis.plotly.plot_model` returns a `Plotly Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_ object, which you can visualize using the `show` method.
# In this example, the rendering mode is set to `notebook`, allowing the figure to be embedded directly within a Jupyter Notebook.
# If you want it to automatically choose the rendering mode, simply call `show` without providing any arguments.
# For more details on the available `renderer`, please refer to `Displaying Figures in Python <https://plotly.com/python/renderers/>`_
