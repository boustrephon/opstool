r"""
Bridge superstructure section
====================================================
This example shows how to create a bridge superstructure section using
the `opstool.pre.section` module.
"""

# %%
import matplotlib.pyplot as plt

import opstool as opst

# %%
# We create a polygonal patch (a single I-beam) by incrementing the x,y coordinates:

incrs = [
    (0.72, 0.0),
    (0, 0.22),
    (-0.26, 0.23),
    (0, 0.84),
    (0.1, 0.1),
    (0.33, 0.07),
    (0, 0.13),
    (-1.06, 0.0),
    (0, -0.13),
    (0.33, -0.07),
    (0.1, -0.1),
    (0, -0.84),
    (-0.26, -0.23),
    (0, -0.22),
]

points = opst.pre.section.create_polygon_points(
    start=(0, 0),
    incrs=incrs,
)
patch = opst.pre.section.create_polygon_patch(outline=points)

# %%
# Returns a `Geometry <https://sectionproperties.readthedocs.io/en/latest/gen/sectionproperties.pre.geometry.Geometry.html#sectionproperties.pre.geometry.Geometry>`_ class object of ``sectionproperties``, we can call any of its attributes and methods, such as visualizing geometry:
patch.plot_geometry()
plt.show()

# %%
# We can offset this patch to get a new patch to get multiple I-beams:
patches = [patch]
for i in range(5):
    patches.append(patches[-1].shift_section(x_offset=2.21))

# %%
# `Geometry <https://sectionproperties.readthedocs.io/en/latest/gen/sectionproperties.pre.geometry.Geometry.html#sectionproperties.pre.geometry.Geometry>`_ class overloads the ``+`` operator for merging, so you can call the ``sum`` function to merge all patches into one:
#
# you can try patches[0] + patches[1] + patches[2] + patches[3] + patches[4] + patches[5]

girder = sum(patches[1:], start=patches[0])
girder.plot_geometry()
plt.show()

# %%
# We create the bridge deck:
points = [(0, 0), (13.2, 0), (13.2, 0.165), (0, 0.165)]
deck = opst.pre.section.create_polygon_patch(outline=points)

deck.plot_geometry()
plt.show()

# %%
# We can first align the centers of the two:
deck = deck.align_center(align_to=girder)
(deck + girder).plot_geometry()
plt.show()

# %%
# Then align it to the top:
deck = deck.align_to(girder, on="top")
(deck + girder).plot_geometry()
plt.show()

# %%
# Combination and fiber meshing:
SEC_MESH = opst.pre.section.FiberSecMesh()
SEC_MESH.add_patch_group(deck + girder)
SEC_MESH.set_mesh_size(0.15)
SEC_MESH.set_mesh_color("#87ae73")
SEC_MESH.mesh()

# %%
props = SEC_MESH.get_frame_props(display_results=True)

# %%

# sphinx_gallery_thumbnail_number = 6
SEC_MESH.view(show_legend=False)
plt.gca().set_aspect(0.75)
plt.show()
