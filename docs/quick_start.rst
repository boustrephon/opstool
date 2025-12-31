.. _quickstart:

.. role:: blue
.. role:: blue-bold

.. include:: _sphinx_gallery_examples/quick-start/index.rst

.. nbgallery::
    :caption: Performance testing of megatall building model
    :name: vis-megatall

    quick-start/vis-megatall.ipynb

**opstool** is a powerful and user-friendly library designed to simplify and enhance structural analysis workflows 
with **OpenSees** and **OpenSeesPy**. 
It provides advanced tools for preprocessing, postprocessing, and visualization, making structural 
simulations more efficient and accessible.

Features
########

.. card:: :octicon:`graph;1.5em;sd-mr-1 fill-primary` Preprocessing

   - :blue-bold:`Fiber Section Meshing`: Generate detailed fiber meshes for various geometries. An example is shown below:
     `Fiber Section Mesh <src/pre/sec_mesh.ipynb>`_
   - :blue-bold:`GMSH Integration`: Import and convert `Gmsh <https://gmsh.info/>`__ models, including geometry, mesh, and physical groups.
     An example is shown below: `Converting GMSH to OpenSeesPy <src/pre/read_gmsh.ipynb>`_
   - :blue-bold:`Unit System Management`: Ensure consistency with automatic unit conversions.
     An example is shown below: `Automatic Unit Conversion <src/pre/unit_system.rst>`_
   - :blue-bold:`Quickly apply gravity loads and obtain mass and stiffness matrices`:
     An example is shown below: `Model data retrieval <src/pre/model_data.ipynb>`_
   - :blue-bold:`Load Transformation`: Simplify the application of loads.
     An example is shown below: `Load Transformation <src/pre/loads.ipynb>`_
   - :blue-bold:`Tcl script to OpenSeesPy`: Convert Tcl scripts to OpenSeesPy scripts.
     An example is shown below: `Tcl to OpenSeesPy <src/pre/tcl2py.rst>`_


.. card:: :octicon:`database;1.5em;sd-mr-1 fill-primary` Postprocessing
   
  Easy retrieval and interpretation of analysis results using `xarray <https://docs.xarray.dev/en/stable/index.html#>`__.

  All examples are shown below: `Postprocessing with xarray <src/post/index.ipynb>`_

  The following is a non-exhaustive list of response types supported by this library:

  .. tab-set::
    :sync-group: category

    .. tab-item:: Nodal
        :sync: Nodal

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "disp"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "vel"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "accel"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "reaction"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "reactionIncInertia"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "rayleighForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "pressure"

    .. tab-item:: Frame
        :sync: Frame

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "localForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "basicForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "basicDeformations"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "plasticDeformation"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionDeformations"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionLocs"

    .. tab-item:: Truss
        :sync: Truss

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "axialForce"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "axialDefo"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Stress"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Strain"

    .. tab-item:: Link
        :sync: Link

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "basicDeformation" 
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "basicForce"
    
    .. tab-item:: Fiber Section
        :sync: Fiber Section

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Stresses" 
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Strains" 
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "secForce" 
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "secDefo"
    
    .. tab-item:: Shell
        :sync: Shell

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionDeformations"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Stresses"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Strains"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionForcesAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "sectionDeformationsAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressesAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainsAtNodes"

    .. tab-item:: Plane
        :sync: Plane

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Stresses"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Strains"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressesAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressAtNodesErr"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainsAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainsAtNodesErr"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressMeasures"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainMeasures"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressMeasuresAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainMeasuresAtNodes"

    .. tab-item:: Solid
        :sync: Solid

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Stresses"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "Strains"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressesAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressAtNodesErr"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainsAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainsAtNodesErr"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressMeasures"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainMeasures"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StressMeasuresAtNodes"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "StrainMeasuresAtNodes"

    .. tab-item:: Contact
        :sync: Contact

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "globalForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "localForces"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "localDisp"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "slips"

    .. tab-item:: Sensitivity Analysis
        :sync: Sensitivity Analysis

        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "disp"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "vel"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "accel"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "pressure"
        * :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` "lambdas"

.. card:: :octicon:`eye;1.5em;sd-mr-1 fill-primary` Visualization
   
   - Powered by `PyVista <https://docs.pyvista.org/>`__ (VTK-based) and `Plotly <https://plotly.com/python/>`__ (web-based).
   - Nearly identical APIs for flexible visualization of model geometry, modal analysis, and simulation results.
   - Supports most common OpenSees elements.
   - An example is shown below: `Visualization <src/vis/index.ipynb>`_

.. card:: :octicon:`paper-airplane;1.5em;sd-mr-1 fill-primary` Intelligent Analysis

   - Features like :blue-bold:`automatic step size adjustment` and :blue-bold:`algorithm switching` to optimize simulation workflows.
     An example is shown below: `Intelligent Analysis <src/analysis/smart_analysis.rst>`_
   - :blue-bold:`Moment-Curvature Analysis`: Generate moment-curvature curves for various sections.
     An example is shown below: `Moment-Curvature Analysis <src/analysis/mc_analysis.ipynb>`_
   - :blue-bold:`Linear Buckling Analysis`: Perform linear buckling analysis for stability assessment.
     An example is shown below: `Linear Buckling Analysis <src/analysis/buckling_analysis_linear.ipynb>`_

