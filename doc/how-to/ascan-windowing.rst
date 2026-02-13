Apply Spectral Windowing
========================

.. warning::

    This document is under construction.

This document is coming soon.
In the meantime, please see this excerpt from `demo/_common/engine.py <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/_common/engine.py>`_.

.. code-block::

    # spectral filter with dispersion correction
    window = numpy.hamming(pc.samples_per_ascan)
    phasor = dispersion_phasor(len(window), cfg.dispersion)
    pc.spectral_filter = window * phasor
