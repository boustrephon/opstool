r"""
Excavation Supported by Cantilevered Sheet Pile Wall
=======================================================

This example is from file
`Excavation Supported by Cantilevered Sheet Pile Wall <https://opensees.berkeley.edu/wiki/index.php?title=Excavation_Supported_by_Cantilevered_Sheet_Pile_Wall>`_
on the OpenSees website and has been converted using
`opst.pre.tcl2py <https://opstool.readthedocs.io/en/latest/src/api/pre.html#opstool.pre.tcl2py>`_
The python model script can be found here
`excavation.py <https://ln5.sync.com/dl/41ad07bf0#ice8pjty-62im9fq9-qegdsiq6-rubwni3n>`_
"""

# %%
import openseespy.opensees as ops

# %%
# Load the FEM model function ``FEMmodel`` from file `excavation.py`.
# [excavation.py](https://github.com/yexiang1992/opstool/blob/master/docs/examples/post/excavation/excavation.py)
from utils.excavation import FEMmodel

import opstool as opst
import opstool.vis.pyvista as opsvis

FEMmodel()

# %%
opsvis.set_plot_props(point_size=1, font_size=9)
opsvis.plot_model(show_node_numbering=True, show_ele_numbering=True).show()

# %%
# Create output database (ODB) file.
# Since some elements and nodes will be removed in subsequent analyses, ensure that ``model_update=True``.

ODB = opst.post.CreateODB(
    odb_tag=1,
    model_update=True,
    save_every=8,  # every 8 analysis steps will be saved
    compute_mechanical_measures=True,
    project_gauss_to_nodes="copy",
)

# %%
# GRAVITY ANALYSIS (w/ INITIAL STATE ANALYSIS TO RESET DISPLACEMENTS)
# -----------------------------------------------------------------------
# define analysis parameters for gravity phase
ops.constraints("Transformation")
ops.test("NormDispIncr", 1e-05, 50, 0)
ops.algorithm("Newton")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.integrator("LoadControl", 1)
ops.analysis("Static")

# %%
# Perform an initial state analysis, where elements with tags 1001--1042 are
# [BeamContact2D](https://opensees.berkeley.edu/wiki/index.php?title=BeamContact2D).

# turn on initial state analysis feature
ops.InitialStateAnalysis("on")
# ensure soil material intially considers linear elastic behavior
ops.updateMaterialStage("-material", 1, "-stage", 0)
# set contact elements to be frictionless for gravity analysis
ops.setParameter("-val", 0, "-eleRange", 1001, 1042, "friction")
# analysis 4 steps, and fetch response
for _ in range(4):
    ops.analyze(1)
    ODB.fetch_response_step()


# %%
# Update soil material to consider elastoplastic behavior and analyze a few more steps:

# update soil material to consider elastoplastic behavior and analyze a few more steps
ops.updateMaterialStage("-material", 1, "-stage", 1)
# analysis 4 steps, and fetch response
for _ in range(4):
    ops.analyze(1)
    ODB.fetch_response_step()
# designate end of initial state analysis (zeros displacements, keeps state variables)
ops.InitialStateAnalysis("off")
# turn on frictional behavior for beam contact elements
ops.setParameter("-val", 1, "-eleRange", 1001, 1042, "friction")

# %%
# REMOVE ELEMENTS TO SIMULATE EXCAVATION
# ---------------------------------------

# define analysis parameters for excavation phase
ops.wipeAnalysis()
ops.constraints("Transformation")
ops.test("NormDispIncr", 0.0001, 60)
ops.algorithm("KrylovNewton")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.integrator("LoadControl", 1)
ops.analysis("Static")

# %%
# We first define a function to avoid repetitive removal of elements and nodes,
# and then proceed with several steps of analysis.


def remove_components(ele_tags, node_tags, nsteps=4):
    for etag in ele_tags:
        ops.remove("element", etag)
    for ntag in node_tags:
        ops.remove("node", ntag)
    # run analysis after object removal
    for _ in range(nsteps):
        ops.analyze(1)
        ODB.fetch_response_step()


# %%
# Remove objects associated with lift 1:

# soil elements
ele_tags = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
ele_tags += [1042]  # contact element
# soil nodes
node_tags = [430, 437, 446, 455, 461, 468, 473, 476, 480, 482, 484]
node_tags += [1042]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 1 removed")

# %%
# We can then remove the remaining 9 lifts

# remove objects associated with lift 2
# soil elements
ele_tags = [181, 182, 183, 184, 185, 186, 187, 188, 189, 190]
ele_tags += [1040]  # contact element
# soil nodes
node_tags = [412, 424, 433, 444, 453, 460, 466, 471, 475, 479, 483]
node_tags += [1040]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 2 removed")

# %%
# remove objects associated with lift 3

# soil elements
ele_tags = [171, 172, 173, 174, 175, 176, 177, 178, 179, 180]
ele_tags += [1038]  # contact element
# soil nodes
node_tags = [387, 405, 418, 429, 442, 450, 458, 464, 470, 477, 481]
node_tags += [1038]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 3 removed")

# %%
# remove objects associated with lift 4

# soil elements
ele_tags = [161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
ele_tags += [1036]  # contact element
# soil nodes
node_tags = [363, 380, 398, 414, 427, 439, 448, 457, 465, 472, 478]
node_tags += [1036]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 4 removed")

# %%
# remove objects associated with lift 5

# soil elements
ele_tags = [151, 152, 153, 154, 155, 156, 157, 158, 159, 160]
ele_tags += [1034]  # contact element
# soil nodes
node_tags = [336, 353, 378, 395, 411, 425, 440, 449, 459, 467, 474]
node_tags += [1034]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 5 removed")

# %%
# remove objects associated with lift 6

# soil elements
ele_tags = [141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
ele_tags += [1032]  # contact element
# soil nodes
node_tags = [308, 326, 347, 369, 392, 408, 426, 441, 452, 462, 469]
node_tags += [1032]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 6 removed")

# %%
# remove objects associated with lift 7

# soil elements
ele_tags = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140]
ele_tags += [1030]  # contact element
# soil nodes
node_tags = [281, 304, 322, 345, 370, 394, 415, 428, 443, 454, 463]
node_tags += [1030]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 7 removed")

# %%
# remove objects associated with lift 8

# soil elements
ele_tags = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
ele_tags += [1028]  # contact element
# soil nodes
node_tags = [260, 278, 302, 325, 346, 374, 399, 419, 434, 447, 456]
node_tags += [1028]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 8 removed")

# %%
# remove objects associated with lift 9

# soil elements
ele_tags = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
ele_tags += [1026]  # contact element
# soil nodes
node_tags = [241, 253, 277, 306, 327, 350, 379, 406, 422, 438, 451]
node_tags += [1026]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 9 removed")

# %%
# remove objects associated with lift 10

# soil elements
ele_tags = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
ele_tags += [1024]  # contact element
# soil nodes
node_tags = [218, 239, 259, 282, 307, 335, 364, 389, 413, 431, 445]
node_tags += [1024]  # lagrange multiplier node

remove_components(ele_tags, node_tags, nsteps=4)

print("Lift 10 removed")

# %%
# We can save all previous responses to a file:
# zlib compression is used to reduce file size.
ODB.save_response()

# %%
# Post-processing
# ---------------
import matplotlib.pyplot as plt

import opstool as opst
import opstool.vis.pyvista as opsvis

# %%
# Since the result data has already been saved, we can read it at any time for post-processing:
opsvis.set_plot_props(point_size=0, line_width=5, cmap="turbo")
opsvis.set_plot_props(
    scalar_bar_kargs={
        "label_font_size": 12,
        "title_font_size": 13,
        "position_x": 0.85,  # 0--1
    }
)

# %%
# Nodal responses
# +++++++++++++++++

# sphinx_gallery_thumbnail_number = 2
opsvis.plot_nodal_responses(
    odb_tag=1,
    slides=True,
    defo_scale=20,
    resp_type="disp",
    resp_dof=["UX", "UY"],
    unit_symbol="m",
).show()

# %%
# We can create animations:

fig = opsvis.plot_nodal_responses_animation(
    odb_tag=1,
    framerate=20,
    defo_scale=25,
    savefig="images/NodalRespAnimation-excavation.gif",
    resp_type="disp",
    resp_dof=["UX", "UY"],
    unit_symbol="m",
)
fig.close()


# %%
# Frame elements responses
# +++++++++++++++++++++++++++
plotter = opsvis.plot_frame_responses(
    odb_tag=1,
    resp_type="sectionForces",
    resp_dof="MZ",
    unit_symbol="kN·m",
    show_values="eleMaxMin",
    scale=3,
    slides=True,
    style="surface",
    show_model=False,  # plot all model
    opacity=1.0,
    show_bc=False,
)
plotter.show()

# %%
opsvis.plot_frame_responses_animation(
    odb_tag=1,
    resp_type="sectionForces",
    resp_dof="MZ",
    unit_symbol="kN·m",
    show_values=False,
    framerate=20,
    scale=3,
    style="surface",
    opacity=1.0,
    show_model=True,  # plot all model
    show_bc=False,
    savefig="images/FrameForcesMZ-excavation.gif",
).close()


# %%
# Plane elements response
# +++++++++++++++++++++++++++
pl = opsvis.plot_unstruct_responses(
    odb_tag=1, slides=True, ele_type="Plane", resp_type="StressesAtNodes", resp_dof="sigma22", unit_symbol="kPa"
)
pl.show()

# %%
opsvis.set_plot_props(show_mesh_edges=False)
opsvis.plot_unstruct_responses(
    odb_tag=1,
    slides=True,
    ele_type="Plane",
    resp_type="StressesAtNodes",
    resp_dof="sigma12",
    show_defo=True,
    defo_scale=30,
    unit_symbol="MPa",
).show()

# %%
opsvis.plot_unstruct_responses(
    odb_tag=1, slides=True, ele_type="Plane", resp_type="StressesAtNodes", resp_dof="sigma_vm"
).show()

# %%
# Read data from ODB
# +++++++++++++++++++++++++++

# %%
# Reading the nodal displacements
data = opst.post.get_nodal_responses(odb_tag=1, resp_type="disp")
data.isel(time=41).dropna(dim="nodeTags", how="all")  # drop nodes with all NaN values


# %%
# Reading the response of the contact element
data = opst.post.get_element_responses(odb_tag=1, ele_type="Contact")
data

# %%
data["localForces"].sel(eleTags=1001).plot.line(x="time")
plt.show()

# %%
# Let's examine the response of contact element #1034.
# Since it is removed during the fifth lift, its response is truncated at time=16,
# and subsequent data will be filled with ``numpy.nan``.
data["localForces"].sel(eleTags=1034).plot.line(x="time")
plt.show()

# %%
data["localForces"].sel(eleTags=1034).data

# %%
# Reading the response of the beam element
data = opst.post.get_element_responses(odb_tag=1, ele_type="Frame")
print(data)

# %%
data = opst.post.get_element_responses(odb_tag=1, ele_type="Plane")
data

# %%
