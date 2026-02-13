.. _demos:

Demos
=====

.. _demo/python:

Setting Up with Python
----------------------

#.  See :ref:`Getting Started <getting-started/python>` to install *vortex* and its dependencies, if not done so already.

#.  Clone the repository or `download the source <download-source_>`_.
    The demos are located in the `demo <https://gitlab.com/vortex-oct/vortex/-/tree/develop/demo>`_ directory.

    .. code-block:: powershell

        > git clone https://gitlab.com/vortex-oct/vortex.git

#.  Install dependencies common to most demos.
    Make sure that you choose the version of :mod:`cupy` that corresponds to your CUDA version.

    .. code-block:: powershell

        > pip install numpy cupy-cuda12x qtpy matplotlib rainbow-logging-handler

#.  Most demos retrieve their settings from the shared file `demo/_common/engine.py <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/_common/engine.py>`_.
    Edit this file as necessary for your OCT system.
    Note that these settings represent only a fraction of those available in *vortex*.
    You may need to edit the ``BaseEngine`` class in `demo/_common/engine.py <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/_common/engine.py>`_ to configure additional options (e.g., trigger delay).

    .. code-block:: python

        DEFAULT_ENGINE_PARAMS = StandardEngineParams(
            scan_dimension=5,
            bidirectional=False,
            ascans_per_bscan=500,
            bscans_per_volume=500,
            galvo_delay=95e-6,

            clock_samples_per_second=int(800e6),
            # zero blocks to acquire means infinite acquisition
            blocks_to_acquire=0,
            ascans_per_block=500,
            samples_per_ascan=2752,

            blocks_to_allocate=128,
            preload_count=32,

            swept_source=source.Axsun200k,
            internal_clock=True,
            clock_channel=Channel.A,
            input_channel=Channel.B,

            process_slots=2,
            dispersion=(2.8e-5, 0),

            log_level=1,
        )

    You can adjust the source settings to match your own as follows.

    .. code-block:: python

        from vortex.engine import Source
        my_source = Source(
            triggers_per_second=30000,
            clock_rising_edges_per_trigger=1234,
            duty_cycle=0.4,
            imaging_depth_meters=0.05
        )

#.  You may now run a demo, such as :ref:`demo/live-view`.

    .. code-block:: powershell

        > python path/to/demo/live_view.py

    If you need to adjust the default widget colormaps, modify the ``range`` and ``cmap`` arguments to the widget constructors (e.g., :class:`RasterEnFaceWidget`).

Setting Up with C++
-------------------

See the :ref:`build-guide` for compiling *vortex* with ``ENABLE_DEMOS=ON`` during the configuration process.

List of Demos
-------------

.. toctree::
   :maxdepth: 1

   demo/live-view
   demo/live-view-simple
   demo/live-view-raster-aim
   demo/live-view-repeated
   demo/live-view-dynamic
   demo/live-view-alazar-fft

   demo/scan-explorer
   demo/acquire-to-disk

   demo/engine
   demo/generate-scan
