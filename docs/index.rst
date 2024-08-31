``dendritex`` documentation
===========================

`dendritex <https://github.com/chaoming0625/dendritex>`_ provides dendritic modeling capabilities in JAX for brain dynamics.




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U dendritex[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U dendritex[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U dendritex[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `BDP ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_:




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/dendritex.rst
   apis/dendritex.neurons.rst
   apis/dendritex.ions.rst
   apis/dendritex.channels.rst
   apis/integration.rst



