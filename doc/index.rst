VORTEX
======

*vortex* is a high-performance C++ library with Python bindings for building real-time OCT engines.
It provides:

* modular components out of which an OCT engine is built
* a flexible OCT engine that accommodates most use cases
* extensive Python bindings that interoperate with NumPy_ and CuPy_

.. _NumPy: https://numpy.org/
.. _CuPy: https://cupy.dev/

*vortex* is designed from the ground up to:

- promote reproducibility in data acquisition and processing
- guarantee synchronization between data streams
- support on-the-fly changes for interactive use cases
- integrate cleanly with user interfaces
- facilitate rapid prototyping with high-level abstractions and sane defaults
- minimize restrictions and assumptions

*vortex* is developed and maintained by `Mark Draelos`_ in the `Image-Guided Medical Robotics Lab`_ at the `University of Michigan`_.

.. note::

    *vortex* is an acronym for **V**\ ariable **O**\ CT for **R**\ eal-\ **T**\ ime **Ex**\ ecution.
    Its logo is a warning sign that depicts a cyclone, emblematic of the goal to provide maximum throughput.

.. _`Mark Draelos`: https://websites.umich.edu/~mdraelos/
.. _`Image-Guided Medical Robotics Lab`: https://medical.robotics.umich.edu/
.. _`University of Michigan`: https://umich.edu/

Notable Features
----------------

*vortex* has several multiple features that make it particularly suitable for the OCT research and development community.

-   **Motion Planner for Scan Generation**

    Generate dynamically feasible scan patterns for your hardware.
    *vortex* will transparently insert the minimum number of inactive samples to avoid scan distortion and automatically crop out flyback.

-   **Scan Markers for Flexible Patterns**

    The scan pattern is encoded by markers that annotate the scan waveforms rather than a rigid structure.
    *vortex* reconstructs the scan after the acquisition by decoding the markers.
    This decouples scan pattern generation from the core OCT engine.

-   **High-performance Processing Pipeline**

    *vortex* implements a reconfigurable engine that performs your acquisition and processing once given a description of your system.
    The engine pre-allocates resources for quick startup after initialization.

-   **Multiple GPU Support**

    *vortex*'s engine can easily distribute large workloads over multiple GPUs.
    The engine will infer the necessary transfers between GPUs or rely on peer-to-peer access, if supported.

-   **Extensive Python Bindings**

    *vortex* is fully usable from Python and C++.
    The detailed Python bindings provide nearly a one-to-one correspondence with low-level C++ capabilities without needing to recompile between application changes.

-   **Full OCT Application Feature Set**

    *vortex* supports UI inactivity, real-time data storage to disk is popular file formats, synchronized I/O, and other usability features necessary to build a complete OCT application.

-   **Permissive BSD-3 License**

    *vortex* is released under the permissive BSD-3 license that supports commercial and research use.

Quickstart
----------

#.  Follow the :ref:`getting-started` guide for Python or C++.

#.  Configure and run some :ref:`demos`.

#.  Check out the :ref:`build-guide` to customize *vortex*.

.. toctree::
    :hidden:

    getting-started
    release

.. toctree::
    :caption: Review
    :hidden:

    overview
    data-model
    concepts
    module
    api

.. toctree::
    :caption: Learn
    :hidden:

    tutorial
    how-to
    demo
    troubleshoot

.. toctree::
    :caption: Develop
    :hidden:

    develop/build
    develop/debug
    develop/profiler
    develop/hardware

.. Indices and Tables
.. ------------------

.. * :ref:`search`
