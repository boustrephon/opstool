:orphan:

###############
Visualization
###############


The ``opstool`` currently offers two engines for visualizing models and responses, 
each with its own strengths and weaknesses.
Both engines are custom-tailored to provide nearly identical visualization functionalities for ``OpenSeesPy``.

**Supports all response types available in the post-processing modules.**


.. tip::
   In Plotly, only surface rendering is supported. 
   As a result, solid elements like tetrahedra and hexahedra are visualized using triangulated meshes, 
   meaning that Plotly does not natively support solid meshes as PyVista does.

.. important::
    If your model is large or you are demonstrating a dynamic analysis with a very large number of steps, 
    do not use Plotly's ``slide`` module or attempt to create animations, as it will be extremely slow.
    Ensure that ``slides=False``.

With these two engines, ``opstool`` ensures flexibility and robust visualization options 
tailored for ``OpenSeesPy`` users, enabling effective analysis and presentation of structural models and 
simulation results.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Plotyly-Based Visualization
=============================

Refer to the module :py:mod:`opstool.vis.plotly` for details.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Model Geometry">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-01-model-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-01-model-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model Geometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The eigenvalue (modal) visualization provides insights into the dynamic characteristics of the structure. It includes the following features:">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-02-eigen-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-02-eigen-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Eigen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nodal Responses Visualization">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-03-nodal-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-03-nodal-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nodal Responses Visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Frame Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-04-frame-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-04-frame-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Frame Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Truss Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-05-truss-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-05-truss-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Truss Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shell Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-06-shell-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-06-shell-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Shell Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This model code  from Plot stress distribution of a plane stress quad model.">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-07-plane-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-07-plane-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plane stress quad model visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Solid Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/plotly/images/thumb/sphx_glr_ex-08-brick-resp-plotly_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_plotly_ex-08-brick-resp-plotly.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Solid Element Responses</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Pyvista-Based Visualization
============================

Refer to the module :py:mod:`opstool.vis.pyvista` for details.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At any point during model creation, you can run the opstool.vis.pyvista.plot_model function to visualize the current model&#x27;s geometric details. This function operates based on post-processed data saved during the modeling process, ensuring accurate visualization.">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-01-model-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-01-model-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model Geometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The eigenvalue (modal) visualization provides insights into the dynamic characteristics of the structure. It includes the following features:">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-02-eigen-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-02-eigen-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Eigen</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nodal Responses Visualization">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-03-nodal-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-03-nodal-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nodal Responses Visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Frame Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-04-frame-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-04-frame-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Frame Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Truss Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-05-truss-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-05-truss-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Truss Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shell Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-06-shell-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-06-shell-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Shell Element Responses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This model code  from Plot stress distribution of a plane stress quad model.">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-07-plane-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-07-plane-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plane stress quad model visualization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Solid Element Responses">

.. only:: html

  .. image:: /_sphinx_gallery_examples/vis-guide/pyvista/images/thumb/sphx_glr_ex-08-brick-resp-pyvista_thumb.png
    :alt:

  :ref:`sphx_glr__sphinx_gallery_examples_vis-guide_pyvista_ex-08-brick-resp-pyvista.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Solid Element Responses</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /_sphinx_gallery_examples/vis-guide/plotly/index.rst
   /_sphinx_gallery_examples/vis-guide/pyvista/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
