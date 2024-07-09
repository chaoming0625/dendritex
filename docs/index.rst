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


- `brainstate <https://github.com/chaoming0625/brainstate>`_: A ``State``-based transformation system for brain dynamics programming.

- `brainunit <https://github.com/chaoming0625/brainunit>`_: Physical units and unit-aware mathematical system in JAX for brain dynamics and AI4Science.

- `braintaichi <https://github.com/chaoming0625/braintaichi>`_: Leveraging Taichi Lang to customize brain dynamics operators.

- `brainscale <https://github.com/chaoming0625/brainscale>`_: The scalable online learning framework for biological neural networks.

- `dendritex <https://github.com/chaoming0625/dendritex>`_: The dendritic modeling in JAX.

- `braintools <https://github.com/chaoming0625/braintools>`_: The toolbox for the brain dynamics simulation, training and analysis.




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/dendritex.rst
   apis/dendritex.ions.rst
   apis/dendritex.channels.rst



