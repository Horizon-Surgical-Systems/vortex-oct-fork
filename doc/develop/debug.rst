Debugging
=========

Compilation Options in C++
--------------------------

*vortex* provides compilation options to facilitate C++ debugging.
These include capabilities to disable exception handling and CUDA kernel pipelining.
See :ref:`build-guide-debug` in the :ref:`build-guide` regarding compile-time debugging options.

.. _debug-dynamic:

Dynamic Modules in Python
-------------------------

When compiled in modular build mode (``ENABLE_MODULE_BUILD=ON``), *vortex* functionality that is driver- or runtime-specific is isolated into its own shared library.
When imported by Python, *vortex* attempts to load and initialize each bundled module.
To enable debug output from this process, set the environment variable ``VORTEX_LOADER_LOG`` to a non-zero value, such as ``"1"``.
This can be used to identify missing driver or runtime dependencies or to guide investigation into other shared library issues.

.. code-block:: powershell

    > python -c "import os; os.environ['VORTEX_LOADER_LOG']='1'; import vortex; print('OK', vortex.__version__)"
    [...] loader     (I) request to activate module "alazar"
    [...] loader     (I) loading "alazar" from "...\lib\site-packages\vortex\vortex-python-alazar.dll"
    [...] loader     (I) invoking initialization function "bind_alazar" at 0x7ff8e4b41080
    [...] loader     (I) module "alazar" activated successfully
    [...] loader     (I) request to activate module "daqmx"
    [...] loader     (I) loading "daqmx" from "...\lib\site-packages\vortex\vortex-python-daqmx.dll"
    [...] loader     (I) invoking initialization function "bind_daqmx" at 0x7ff8e0e31080
    [...] loader     (I) module "daqmx" activated successfully
    [...] loader     (I) request to activate module "simple"
    [...] loader     (I) loading "simple" from "...\lib\site-packages\vortex\vortex-python-simple.dll"
    [...] loader     (I) invoking initialization function "bind_simple" at 0x7ff8df3310d0
    [...] loader     (I) module "simple" activated successfully
    OK 0.4.4 ['reflexxes', 'cuda', 'hdf5', 'alazar', 'daqmx', 'simple', 'cuda_dynamic_resampling', 'exception_guards', 'pybind11_optimizations']

If a specific driver or runtime component is absent, its corresponding module will fail to load with an error message.
In the example below, the AlazarTech driver has been removed.

.. code-block:: powershell

    > python -c "import os; os.environ['VORTEX_LOADER_LOG']='1'; import vortex; print('OK', vortex.__version__)"
    [...] loader     (I) request to activate module "alazar"
    [...] loader     (I) loading "alazar" from "...\lib\site-packages\vortex\vortex-python-alazar.dll"
    [...] loader     (E) module "alazar" failed to load: failed to load library "...\lib\site-packages\vortex\vortex-python-alazar.dll": The specified module could not be found. (0x0000007e)
    [...] loader     (I) request to activate module "daqmx"
    [...] loader     (I) loading "daqmx" from "...\lib\site-packages\vortex\vortex-python-daqmx.dll"
    [...] loader     (I) invoking initialization function "bind_daqmx" at 0x7ff8e4d31080
    [...] loader     (I) module "daqmx" activated successfully
    [...] loader     (I) request to activate module "simple"
    [...] loader     (I) loading "simple" from "...\lib\site-packages\vortex\vortex-python-simple.dll"
    [...] loader     (E) module "simple" failed to load: failed to load library "...\lib\site-packages\vortex\vortex-python-simple.dll": The specified module could not be found. (0x0000007e)
    OK 0.4.4 ['reflexxes', 'cuda', 'hdf5', 'daqmx', 'cuda_dynamic_resampling', 'exception_guards', 'pybind11_optimizations']

The ``alazar`` and ``simple`` modules subsequently fail to load and the corresponding features are now absent.
Functionality of the successfully loaded modules remains available.
