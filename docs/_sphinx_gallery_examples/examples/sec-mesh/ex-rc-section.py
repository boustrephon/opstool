r"""
Hollow Rectangular RC Section
==============================

This example demonstrates how to create a mesh for a hollow rectangular reinforced concrete section.
"""

# %%
import matplotlib.pyplot as plt
import openseespy.opensees as ops

import opstool as opst

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)
coverID, coreID, rebarID = 1, 2, 3
ops.uniaxialMaterial("Concrete01", coverID, -30, -0.002, -15, -0.005)
ops.uniaxialMaterial("Concrete01", coreID, -40, -0.006, -30, -0.015)
ops.uniaxialMaterial("Steel01", rebarID, 200, 2.0e5, 0.02)

# %%

# the points of the outer contour line, only the turning point of the line is needed, counterclockwise or clockwise.
outlines = [
    [0.5, 0],
    [7.5, 0],
    [8, 0.5],
    [8, 4.5],
    [7.5, 5],
    [0.5, 5],
    [0, 4.5],
    [0, 0.5],
]
# cover thick
cover_d = 0.08
# Offset to get the inner boundary of the cover layer
coverlines = opst.pre.section.offset(outlines, d=cover_d)

# Generate polygonal geometry object for cover layer
cover_geo = opst.pre.section.create_polygon_patch(outlines, holes=[coverlines])

# Creating core with voids
holelines1 = [[1, 1], [3.5, 1], [3.5, 4], [1, 4]]
holelines2 = [[4.5, 1], [7, 1], [7, 4], [4.5, 4]]
core_geo = opst.pre.section.create_polygon_patch(coverlines, holes=[holelines1, holelines2])

# %%

SEC_MESH = opst.pre.section.FiberSecMesh()
SEC_MESH.add_patch_group({"cover": cover_geo, "core": core_geo})
SEC_MESH.set_mesh_size({"cover": 0.25, "core": 0.25})
SEC_MESH.set_ops_mat_tag({"cover": coverID, "core": coreID})  # add opensees mat tag !!!
SEC_MESH.set_mesh_color({"cover": "#dbb40c", "core": "#88b378"})
SEC_MESH.mesh()

# %%
dia = 0.06
rebars_outer = opst.pre.section.offset(coverlines, d=dia / 2)
SEC_MESH.add_rebar_line(
    points=rebars_outer,
    dia=dia,
    n=200,
    ops_mat_tag=rebarID,  # add opensees mat tag
    group_name="rebar #30",
    color="#580f41",
)

dia = 0.045
rebars_inner = opst.pre.section.offset(holelines1, d=-dia / 2 - 0.05)
SEC_MESH.add_rebar_line(
    points=rebars_inner,
    dia=dia,
    gap=0.1,
    ops_mat_tag=rebarID,  # add opensees mat tag !!!
    group_name="rebar #25",
    color="blue",
)

rebars_inner = opst.pre.section.offset(holelines2, d=-dia / 2 - 0.05)
SEC_MESH.add_rebar_line(
    points=rebars_inner,
    dia=dia,
    gap=0.1,
    ops_mat_tag=rebarID,  # add opensees mat tag !!!
    group_name="rebar #25",
    color="blue",
)

# %%
SEC_MESH.centring()
SEC_MESH.view(fill=False, show_legend=True)
plt.show()

# %%
props = SEC_MESH.get_frame_props(display_results=True)

# %%
G = 10000  # shear modulus of the concrete
J = props["J"]  # get the torsional constant
GJ = G * J
GJ

# %%
SEC_MESH.centring()

sec_tag = 1
SEC_MESH.to_opspy_cmds(secTag=sec_tag, GJ=GJ)  # to the opensees commands

# %%
ops.node(1, 0, 0, 0)
ops.node(2, 0, 0, 0)

sec_tag = 1
ops.element("zeroLengthSection", sec_tag, 1, 2, sec_tag)
