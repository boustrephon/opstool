.. This theme is based on the `sphinx-needs`.

.. role:: blue


.. grid::
   :gutter: 2 3 3 3
   :margin: 4 4 1 2
   :class-container: architecture-bg
   :class-row: sd-w-100

   .. grid-item::
      :columns: auto
      :child-align: justify
      :class: sd-fs-3

      .. div:: sd-font-weight-bold
         
         Making structural and geotechnical systems simulation with OpenSees and OpenSeesPy effortless and straightforward

      .. div:: sd-fs-5 sd-font-italic

         Welcome to the documentation for opstool, a thoughtfully crafted preprocessing and postprocessing package 
         designed for OpenSeesPy. 

      .. grid:: 1 1 2 2
         :gutter: 2 2 3 3
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: auto

            .. button-ref:: quick_start
               :ref-type: doc
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               Get Started
      
         .. grid-item::
            :columns: auto

            .. button-link:: https://opensees.berkeley.edu/
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               OpenSees
         
         .. grid-item::
            :columns: auto

            .. button-link:: https://openseespydoc.readthedocs.io/en/latest/index.html#
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               OpenSeesPy
         
         .. grid-item::
            :columns: auto

            .. button-link:: https://portwooddigital.com/
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               Portwood Digital

----------------


Design Philosophy of opstool
-------------------------------

.. grid:: 2

    .. grid-item-card:: Philosophy
      
         - OpenSeesPy as the core analysis and solution engine
         - Modular, end-to-end workflow
         - xarray-driven data organization
         - Unified element-type classification
         - Consistent response handling across elements
         - Interchangeable visualization backends
         - Minimal intrusion into user scripts

    .. grid-item-card::  Architecture diagram of the opstool package

         .. figure:: _static/opstool-architecture.png
            :width: 400px
            :align: center

``opstool`` is a modular, data-oriented framework built on ``OpenSeesPy`` (It can also be a version compiled by yourself) to support the complete structural analysis workflow, including *pre-processing, analysis, post-processing, and visualization.*

*OpenSeesPy is used strictly as the analysis and solution engine*, while opstool focuses on structuring and managing analysis data. 
Simulation results are organized using xarray-based labeled N-dimensional arrays, enabling clear, consistent handling of time-dependent and high-dimensional responses.

A key design choice is that most OpenSees element types are systematically classified (e.g., nodal, truss, frame, shell, plane, and solid elements). 
This classification allows responses from different element formulations to be stored, queried, and visualized in a unified manner, independent of the underlying element implementation.

By separating numerical solving from data organization and visualization, opstool enhances usability and extensibility while preserving the flexibility of script-based OpenSeesPy modeling.

.. note::

   **For v1.0.20 and earlier document versions**  See `https://opstool.readthedocs.io/en/v1.0.20/ <https://opstool.readthedocs.io/en/v1.0.20/>`_.


Citing opstool
---------------

If you find ``opstool`` useful in your research or projects, please consider citing the following publication:

.. grid:: 2

    .. grid-item-card::

         .. _cards-clickable:

         .. card:: :octicon:`project-roadmap;1.5em;sd-mr-1 fill-primary` OPSTOOL Publication
            :link: https://www.sciencedirect.com/science/article/pii/S2352711025000937

            Yexiang Yan and Yazhou Xie. *"opstool: A Python library for OpenSeesPy analysis automation, streamlined pre-and post-processing, and enhanced data visualization."* SoftwareX 30 (2025): 102126. DOI: `https://doi.org/10.1016/j.softx.2025.102126 <https://www.sciencedirect.com/science/article/pii/S2352711025000937>`_.


    .. grid-item-card::  Bibtex entry

         .. code-block:: bibtex

            @article{YAN2025102126,
               title = {opstool: A Python library for OpenSeesPy analysis automation, streamlined pre- and post-processing, and enhanced data visualization},
               author = {Yexiang Yan and Yazhou Xie},
               journal = {SoftwareX},
               volume = {30},
               pages = {102126},
               year = {2025},
               issn = {2352-7110},
               doi = {https://doi.org/10.1016/j.softx.2025.102126},
               }

Features Overview
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`stack;1.5em;sd-mr-1 fill-primary` Preprocessing

      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Fiber Section Meshing and Property Calculation
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Easily translate Tcl-based OpenSees models into Python scripts
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Unified Unit Conversion System
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Integration with `Gmsh <https://gmsh.info/>`__ Meshing
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Fast gravity load application
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` System matrix extraction
      - :octicon:`zap;1.0em;sd-mr-1 fill-primary` Features in Development

   .. grid-item-card:: :octicon:`eye;1.5em;sd-mr-1 fill-primary` Elegant Visualization

      - Powered by `PyVista <https://docs.pyvista.org/>`__ and `Plotly <https://plotly.com/python/>`__
      - Supporting most commonly used OpenSees element types.
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Truss Elements
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Frame Elements   
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Link Elements
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Shell Elements
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Plane Elements
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Solid Elements

   .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1 fill-primary` Results Post-Processing

      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Powered by `xarray <https://docs.xarray.dev/en/stable/index.html#>`_
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Nodal Responses, Various Element Responses (Truss, Frame, Link, Shell, Plane, Solid, Contact).
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Static and Dynamic Analysis
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Sensitivity Analysis
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Linear Buckling Analysis
   
   .. grid-item-card:: :octicon:`dependabot;1.5em;sd-mr-1 fill-primary` Analysis Assistance

      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Smart Analysis, features include automatic step size division, algorithm switching, and other advanced functionalities
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Moment-Curvature Analysis of Sections
      - :octicon:`checkbox;1.0em;sd-mr-1 fill-primary` Linar Buckling Analysis

---------------------------------------------------

Why Choose opstool?
--------------------

- **Efficiency**: Streamlines complex workflows, reducing time spent on repetitive tasks.
- **Flexibility**: Provides nearly identical interfaces for different visualization engines.
- **Accessibility**: Makes advanced structural analysis tools like OpenSeesPy more approachable to users of all levels.

``opstool`` is actively evolving, with continuous additions of new features planned for the future.
With ``opstool``, you can focus on what matters most: 
understanding and solving your structural engineering challenges. 
Whether you are building models, visualizing results, or interpreting data, 
``opstool`` is your go-to solution for OpenSeesPy workflows.


                                                                                                       
Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   installation.rst
   known_issues.rst
   quick_start.rst

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   src/pre/index
   src/post/index.rst
   _sphinx_gallery_examples/vis-guide/index
   src/analysis/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   _sphinx_gallery_examples/examples/index


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   
   src/api/global
   src/api/pre
   src/api/post
   src/api/vis
   src/api/analysis

.. toctree::
   :maxdepth: 1
   :caption: Theory Reference

   src/theory/index.rst        

Acknowledgements
-----------------
I sincerely thank `EOS <https://www.eurasianopensees.com/>`_ for the *Honorable Mention* in *Pre/Post-processing & Implementations* at the *1st EOS OpenSees Research and Innovation Awards*.

`🎉 Announcing the Winners of the 1st EOS OpenSees Research and Innovation Awards! 🎉 <https://www.linkedin.com/feed/update/urn:li:activity:7398770387381219328/>`_

.. :octicon:`copilot;1.25em;sd-mr-1 fill-primary` This document theme is adapted from `sphinx-needs <https://github.com/useblocks/sphinx-needs/>`__ with modifications. We sincerely thank the original author(s) for their contribution.`