r"""
Concrete-Filled Steel Tube (CFST) Section Meshing
=======================================================
This example demonstrates how to create a mesh for a Concrete-Filled Steel Tube (CFST) section. The CFST section consists of an outer steel tube and an inner concrete core.
"""

import opstool as opst

# %%
# The following materials are intended for the purpose of calculating section properties and are not related to OpenSeesPy.
Ec = 3.45e7
Es = 2.0e8
Nus = 0.3
Nuc = 0.2
tube_d = 0.05
steel_mat = opst.pre.section.create_material(name="steel", elastic_modulus=Es, poissons_ratio=Nus)
conc_mat = opst.pre.section.create_material(name="conc", elastic_modulus=Ec, poissons_ratio=Nuc)

# %%
all_ = opst.pre.section.create_circle_patch(xo=[0, 0], radius=0.75, n_sub=30)
conc_geo = opst.pre.section.create_circle_patch(xo=[0, 0], radius=0.75 - tube_d, n_sub=30)
tube_geo = all_ - conc_geo
opst.pre.section.set_patch_material([conc_geo, tube_geo], [conc_mat, steel_mat])

# %%
SEC_MESH = opst.pre.section.FiberSecMesh()
SEC_MESH.add_patch_group({"tube": tube_geo, "conc": conc_geo})
SEC_MESH.set_mesh_size({"tube": 0.1, "conc": 0.1})
SEC_MESH.set_mesh_color({"tube": "#ffa756", "conc": "#40a368"})
SEC_MESH.set_ops_mat_tag({"tube": 1, "conc": 2})
SEC_MESH.mesh()

# %%
SEC_MESH.centring()
props = SEC_MESH.get_sec_props(Eref=Es, display_results=True)

# %%
frame_props = SEC_MESH.get_frame_props(Eref=Es, display_results=True)

# %%
SEC_MESH.view(fill=True, show_legend=True)
