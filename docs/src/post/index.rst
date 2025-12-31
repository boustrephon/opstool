Post-Processing
==================

The ``opstool`` framework adopts `xarray <https://xarray.pydata.org/en/stable/>`_ as its core data structure, leveraging its powerful capabilities for handling labeled multi-dimensional data.
This choice provides a flexible and intuitive way to manage and process structural analysis results.

You can learn ``xarray`` through the following resources:

`Xarray in 45 minutes <https://tutorial.xarray.dev/overview/xarray-in-45-min.html>`_

`Project Pythia <https://foundations.projectpythia.org/core/xarray/>`_

In a nutshell
------------------

In ``opstool``, saving responses is very simple. 
You only need to run the analysis ``iteratively`` and use the following concise code:

.. code-block:: pycon

   >>> ODB = opst.post.CreateODB(odb_tag=1, **kargs)
   >>> for i in range(Nsteps):  # you need to run the analysis iteratively
   >>>    # perform analysis for one step
   >>>    ops.analyze(1)  # or other analysis commands
   >>>    ODB.fetch_response_step()  # fetch response at this step
   >>>    # if you want to fetch response every 10 steps
   >>>    #  if i % 10 == 0:
   >>>    #     ODB.fetch_response_step()
   >>> ODB.save_response()  # save the responses after the analysis is done

:class:`opstool.post.CreateODB` is used to create a database object, where ``odb_tag`` specifies the label. 
Then, the method :meth:`opstool.post.CreateODB.fetch_response_step` is called to iteratively save the data at each step. 
Once the analysis is complete, all responses are saved by the method :meth:`opstool.post.CreateODB.save_response`. 
That's it.

The above code will automatically retrieve and save most of the commonly used output quantities of OpenSees.

Other args in ``kargs`` can be used to customize the output, such as specifying which response types to save, the frequency of saving, computing derived stress measures, and project Gaussian points responses to nodes, and more.

.. note::

   - ``odb_tag`` is used to label different datasets (different sets of responses). Subsequent post-processing and visualization will rely on this ``odb_tag`` for data retrieval.
   - The analysis must be performed iteratively (e.g., using a loop) to enable the step-by-step fetching of responses.
   - Static and dynamic analyses are both performed step-by-step, so the post-processing is consistent.

Supported response types
-------------------------

The following is a non-exhaustive list of response types supported by opstool:

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

Gallery of Examples
-------------------

The following examples all deal with response data. For visualization, please refer to visualization instructions. They share datasets tagged with ``odb_tag``.

.. nbgallery::
    :caption: 👉 👉 👉
    :name: post-gallery

    eigen.ipynb
    nodal_resp.ipynb
    frame_resp.ipynb
    fiber_sec_resp.ipynb
    truss_resp.ipynb
    shell_resp.ipynb
    plane_up_resp.ipynb
    brick_resp.ipynb
    sensitivity_resp.ipynb
    unit_transform.ipynb

.. * [Eigenvalue Analysis](eigen.ipynb)
.. * [Node Responses](nodal_resp.ipynb)
.. * [Frame Responses](frame_resp.ipynb)
.. * [Fiber Section Responses](fiber_sec_resp.ipynb)
.. * [Truss Responses](truss_resp.ipynb)
.. * [Shell Responses](shell_resp.ipynb)
.. * [Plane response](plane_up_resp.ipynb)
.. * [Brick Responses](brick_resp.ipynb)
.. * [Sensitivity Analysis Responses](sensitivity_resp.ipynb)
.. * [Unit conversion in post-processing](unit_transform.ipynb)