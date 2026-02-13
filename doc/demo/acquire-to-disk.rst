.. _demo/acquire-to-disk:

Acquire to Disk
===============

.. warning::

    This document is under construction.

In the meantime, please see `demo/volume_to_disk.py <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/volume_to_disk.py>`_.

.. code:: powershell

    > python demo/volume_to_disk.py -h
    usage: volume_to_disk.py [-h] [--format {matlab,hdf5,numpy}] [--no-save-ascans] [--no-save-spectra] [--prefix PREFIX]

    save volume to disk

    optional arguments:
    -h, --help            show this help message and exit
    --format {matlab,hdf5,numpy}
                            file format for saving (default: numpy)
    --no-save-ascans      Do not save A-scans (default: False)
    --no-save-spectra     Do not save spectra (default: False)
    --prefix PREFIX       prefix for output file names (default: )
