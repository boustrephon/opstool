r"""
Steel-Concrete Composite Section Meshing
=======================================================
This example demonstrates how to create a mesh for a steel-concrete composite section.
"""

import opstool as opst

# %%
# The following materials are intended for the purpose of calculating section properties and are not related to OpenSeesPy.
Ec = 3.45e7
Es = 2.0e8
Nus = 0.3
Nuc = 0.2
pho_c = 2.55
pho_s = 7.86
steel_mat = opst.pre.section.create_material(name="steel", elastic_modulus=Es, poissons_ratio=Nus, density=pho_s)
conc_mat = opst.pre.section.create_material(name="conc", elastic_modulus=Ec, poissons_ratio=Nuc, density=pho_c)

# %%
outlines = [[0, 0], [2, 0], [2, 2], [0, 2]]
coverlines = opst.pre.section.offset(outlines, d=0.05)
cover_geo = opst.pre.section.create_polygon_patch(outlines, holes=[coverlines], material=conc_mat)
bonelines = [
    [0.5, 0.5],
    [1.5, 0.5],
    [1.5, 0.7],
    [1.1, 0.7],
    [1.1, 1.3],
    [1.5, 1.3],
    [1.5, 1.5],
    [0.5, 1.5],
    [0.5, 1.3],
    [0.9, 1.3],
    [0.9, 0.7],
    [0.5, 0.7],
    [0.5, 0.5],
]

core_geo = opst.pre.section.create_polygon_patch(coverlines, holes=[bonelines], material=conc_mat)

bone_geo = opst.pre.section.create_polygon_patch(bonelines, material=steel_mat)

# %%
SEC_MESH = opst.pre.section.FiberSecMesh()
SEC_MESH.add_patch_group(dict(cover=cover_geo, core=core_geo, bone=bone_geo))
SEC_MESH.set_mesh_size(dict(cover=0.1, core=0.2, bone=0.1))
SEC_MESH.set_mesh_color(dict(cover="gray", core="#b84592", bone="#ffc168"))
SEC_MESH.set_ops_mat_tag(dict(cover=1, core=2, bone=4))
SEC_MESH.mesh()

# %%
# add rebars
rebar_lines1 = opst.pre.section.offset(outlines, d=0.05 + 0.032 / 2)
SEC_MESH.add_rebar_line(points=rebar_lines1, dia=0.032, gap=0.1, color="black", ops_mat_tag=3)

# %%
# Since it is a composite section, we use the elastic modulus of concrete as the reference modulus to obtain equivalent properties based on concrete material.

SEC_MESH.centring()
props = SEC_MESH.get_sec_props(Eref=Ec, display_results=True)

# %%
# Or use the faster property calculation method for frame elements
frame_props = props = SEC_MESH.get_frame_props(Eref=Ec, display_results=True)

# %%
SEC_MESH.view(fill=True, show_legend=True)
