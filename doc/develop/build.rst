.. _build-guide:

Build Guide
===========

Although *vortex* supports CMake, its many dependencies make it somewhat cumbersome to build.
This guide attempts to ease the difficulty in building *vortex*.

.. warning::

  If you only plan to use *vortex* via Python (and do not need to run a C++ debugger), use of a binary wheel is strongly recommended over setting up your own build.

Build Options
-------------

*vortex*'s build options are controlled exclusively through CMake flags.
There are flags that tailor the build to available dependencies, that enable *vortex*-specific features, and support debugging.

.. _dependency-control:

Dependency Control
^^^^^^^^^^^^^^^^^^

*   ``WITH_ALAZAR`` (default: ``OFF``):
    Provide support for AlazarTech digitizers.

*   ``WITH_ALAZAR_GPU`` (default: ``OFF``):
    Provide support for AlazarTech digitizers using their proprietary CUDA support.
    This option is not recommended as *vortex* provides its own CUDA support at a comparable performance level, at least in preliminary testing.

*   ``WITH_ALAZAR_DAC`` (default: ``OFF``):
    Provide support for analog output modules on certain AlazarTech digitizers.

*   ``WITH_ASIO`` (default: ``ON``):
    Provide support for network and serial interfaces (e.g., MachDSP).

*   ``WITH_BACKWARD`` (default: ``ON``):
    Use Backward exception stack traces.

*   ``WITH_CUDA`` (default: ``ON``):
    Use CUDA for nVIDIA GPU-accelerated OCT processing.
    This option is strongly recommended.

*   ``WITH_DAQMX`` (default: ``OFF``):
    Provide support for signal I/O using National Instruments DAQmx-compatible modules.

*   ``WITH_FFTW`` (default: ``ON``):
    Use FFTW for fast Fourier transforms, primarily in the CPU OCT processor.

*   ``WITH_HDF5`` (default: ``ON``):
    Provide support for the MATLAB v7.3 file format.

*   ``WITH_IMAQ`` (default: ``OFF``):
    Provide support for acquisition using National Instruments IMAQ-compatible cameras.

*   ``WITH_PYTHON`` (default: ``ON``):
    Build Python bindings.

*   ``WITH_REFLEXXES`` (default: ``ON``):
    Enable time-optimal scan trajectory generation with the `Reflexxes <https://github.com/Reflexxes/RMLTypeII>`_ library.
    If this option is disabled, scan generation is not available.
    This option is strongly recommended.

*   ``WITH_TELEDYNE`` (default: ``OFF``):
    Provide support for Teledyne digitizers.

.. seealso::

    For the *vcpkg* manifest features that correspond to each of these options, see the top-level `CMakeLists.txt <https://gitlab.com/vortex-oct/vortex/-/blob/develop/CMakeLists.txt?ref_type=heads>`_.
    In general, a CMake option ``WITH_ABC_XYZ`` would correspond to *vcpkg* feature ``abc-xyz``.

Feature Control
^^^^^^^^^^^^^^^

*   ``ENABLE_DEMOS`` (default: ``OFF``):
    Build C++ demo applications.

*   ``BUILD_PYTHON_WHEEL`` (default ``ON``):
    On Python-enabled builds, generate a binary wheel that installs *vortex*.
    The wheel will be deposited in the build output folder alongside the *vortex* library.

*   ``INSTALL_PYTHON_WHEEL`` (default ``OFF``):
    If the Python wheel is built as above, automatically install the wheel into the current Python environment.

*   ``ENABLE_DEPENDENCY_PACKAGING`` (default ``ON``):
    If the Python wheel is built as above, binary dependencies of *vortex* are analyzed after the build and packaged alongside *vortex*.
    This helps reduce the burden of installing *vortex* in simple deployment scenarios (i.e., those that do not provide binary dependency management).

*   ``ENABLE_OUT_OF_TREE_PACKAGING`` (default ``OFF``): By default, only binary dependencies within the *vortex* root directory are packaged if dependency packaging in enabled.
    This helps avoid deploying closed-source libraries in the wheel.
    It is thus recommended to clone *vcpkg* into the *vortex* root folder so that the dependencies that it provides are automatically packaged.
    Turn on this option to include all non-system binary dependencies in the wheel.

*   ``ENABLE_MODULAR_BUILD`` (default: ``${WITH_PYTHON}``):
    Isolate functionality for specific hardware (e.g., AlazarTech digitizer) in its own shared library.
    This enables dynamic loading of driver- or runtime-specific functionality and avoids the need to compile *vortex* for every specific combination of hardware.
    Modular building is strongly recommended for building Python wheels as the bindings include support for dynamically loading modules based on detected hardware.

.. _build-guide-debug:

Debugging Support
^^^^^^^^^^^^^^^^^

*   ``ENABLE_CUDA_KERNEL_SERIALIZATION`` (default ``OFF``):
    Synchronize the GPU with ``cudaDeviceSynchronize`` after every CUDA kernel launch for debugging purposes.
    This option may adversely affect performance and is not recommended for regular use.
    Python modules compiled with this flag report the ``serial_cuda_kernels`` feature.

*   ``ENABLE_EXCEPTION_GUARDS`` (default ``ON``):
    Surround all callback invocations in *vortex* with ``try``-``catch`` blocks to handle exceptions.
    Turn off this option to propagate exceptions in callbacks to the debugger for debugging purposes.
    This option is strongly recommended for regular use.
    Python modules compiled with this flag report the ``exception_guards`` feature.

*   ``ENABLE_PYBIND11_OPTIMIZATIONS`` (default: ``ON``):
    Allow *pybind11* to enable link-time optimization and, on MacOS and Linux, strip symbols from the Python module.
    Turn off this option to faciliate debugging, especially on MacOS and Linux.
    This option is strongly recommended for regular use.
    Python modules compiled with this flag report the ``pybind11_optimizations`` feature.

.. warning::

    The *vortex* Python bindings will emit warnings if any of these options are left in their non-recommended state.

Dependencies
------------

Open Source via *vcpkg*
^^^^^^^^^^^^^^^^^^^^^^^

All of *vortex*'s open source dependencies are available in its *vcpkg* main registry or the custom *vortex* `registry <https://gitlab.com/vortex-oct/vortex-registry>`_.
The table below is based on *vortex*'s *vcpkg* `manifest <https://gitlab.com/vortex-oct/vortex/-/blob/develop/vcpkg.json?ref_type=heads>`_.

=========================  ======================================  =================
Component                  Packages                                CMake Option
=========================  ======================================  =================
core                       ``fmt spdlog xtensor[tbb,xsimd]``
scan generation            ``reflexxes``                           ``WITH_REFLEXXES``
MachDSP                    ``asio``                                ``WITH_ASIO``
CUDA                       ``cuda``                                ``WITH_CUDA``
CPU OCT processor          ``fftw3[avx2]``                         ``WITH_FFTW``
Python bindings            ``xtensor-python pybind11``             ``WITH_PYTHON``
MATLAB v7.3 file support   ``hdf5[cpp]``                           ``WITH_HDF5``
C++ demos                  ``bfgroup-lyra``                        ``ENABLE_DEMOS``
=========================  ======================================  =================

.. note::

    CUDA is also a :ref:`closed-source dependency <closed-source>`.

.. _closed-source:

Closed Source
^^^^^^^^^^^^^

The closed source-dependencies are provided by installers from the developer.
*vortex* will detect standard installation locations or you can provide a search path hint.

=====================   ==============  ===================  ====================
Component               Installer       CMake Option         Search Path Hint
=====================   ==============  ===================  ====================
Alazar card support     ATS-SDK_        ``WITH_ALAZAR``      ``AlazarTech_DIR``
Alazar CUDA library     ATS-GPU-BASE_   ``WITH_ALAZAR_GPU``  ``AlazarTech_DIR``
Teledyne card support   ADQAPI_         ``WITH_TELEDYNE``    ``Teledyne_DIR``
CUDA runtime            CUDA_           ``WITH_CUDA``        ``CUDAToolkit_ROOT``
DAQmx I/O               `NI DAQmx`_     ``WITH_DAQMX``       ``NIDAQmx_DIR``
IMAQ camera support     `NI IMAQ`_      ``WITH_IMAQ``        ``NIIMAQ_DIR``
=====================   ==============  ===================  ====================

.. note::

    If ATS-GPU-BASE is used, *vortex* will let Alazar manage data movement to the GPU.
    If ATS-SDK is used, *vortex* will handle data movement itself.
    The performance between using ATS-SDK and ATS-GPU-BASE is comparable in preliminary testing.

.. admonition:: Linux

    Installation of the closed source dependencies on Linux is somewhat more nuanced than it is on Windows.

    -   **ATS-SDK**

        The headers and drivers are installed via separate packages.

        .. code-block:: bash

            $ sudo dpkg -i libats_X.Y.Z_amd64.deb
            $ sudo dpkg -i ats-devel_X.Y.Z_amd64.deb

    -   **ADQAPI**

        The headers and drivers are installed via separate packages.

        .. code-block:: bash

            $ sudo dpkg -i libadq0_2023.2_amd64.deb
            $ sudo dpkg -i spd-adq-pci-dkms_1.23_all.deb

    -   **CUDA**

        Many Linux distributions can install CUDA support using their package managers.

        .. code-block:: bash

            $ sudo apt install nvidia-cuda-toolkit

        Make sure to activate the corresponding NVIDIA graphics driver.

    -   **NI DAQmx**

        The main package adds a repository to the system manager.

        .. code-block:: bash

            $ sudo dpkg -i ni-ubuntu2004firstlook-drivers-stream.deb

        The following setups are required to actually install the headers and drivers.

        .. code-block:: bash

            $ sudo apt update
            $ sudo apt install ni-daqmx
            $ sudo dkms autoinstall

        You may need to install the headers for your kernel in order for the NI kernel modules to build.

        .. attention::

            Due to a `known issue with DAQmx on Linux <https://www.ni.com/en-us/support/documentation/bugs/23/ni-linux-device-drivers-2023-q1-known-issues.html>`_, you may need to add ``iommu=off`` to your kernel command line.

.. _ATS-SDK: https://www.alazartech.com/en/product/ats-sdk/27/
.. _ATS-GPU-BASE: https://www.alazartech.com/en/product/ats-gpu-base/30/
.. _ADQAPI: https://www.spdevices.com/what-we-do/products/software
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _`NI DAQmx`: https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html
.. _`NI IMAQ`: https://www.ni.com/en-us/support/downloads/drivers/download.vision-acquisition-software.html

Build Systems
-------------

*vortex* is compatible with any build system that CMake supports.
Examples for common build systems are provided below.
The continuous integration scripts are based on the *vcpkg* example below.

Visual Studio
^^^^^^^^^^^^^

.. tab:: Windows

    Recent versions of Visual Studio (2019 or newer) and Visual Studio Code can open the *vortex* source root as a CMake project.
    Visual Studio will use ``vcpkg.json`` and ``vcpkg-configuration.json`` to install dependencies and use ``CMakePresets.json`` to configure and build the project.
    Make sure to install `Clang/LLVM support for Visual Studio <https://docs.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-160>`_.

    If you are using Visual Studio 2022, you may use the menu option File > Open > CMake... to browse and select ``CMakeLists.txt`` in the *vortex* root folder.
    The included ``CMakePresets.json`` will automatically configure your project correctly for most users.
    To customize the CMake options, use Project > Edit CMake Presets for vortex.
    You may wish to add your own `CMakeUserPresets.json` for further configuration, such as to specify the path to your Python installation.
    Next, select the desired build configuration from the toolbar, configure the project with Project > Configure vortex, and build.

    .. note::

        Visual Studio 2022 bundles *vcpkg* with recent versions.
        Visual Studio 2019 requires that you to clone *vcpkg* and point ``CMAKE_TOOLCHAIN_FILE`` to it.

    .. caution::

        Building *vortex* with the MSVC compiler may produce an internal compiler error.

.. tab:: Linux

    Visual Studio Code can open the *vortex* source root as a CMake project.
    It will then use ``CMakePresets.json`` to configure and build the project.

    .. note::

        Visual Studio Code requires that you to clone *vcpkg* and point ``CMAKE_TOOLCHAIN_FILE`` to it.

    For more advanced needs, add your own ``CMakeUserPresets.json``.
    Make sure to install *clang* and *ninja*.

CMake
^^^^^

To use CMake directly, use any of the existing presets after setting up *vcpkg*.
For more advanced needs, add your own ``CMakeUserPresets.json``.
List the available presets using ``cmake --list-presets``.

.. tab:: Windows

    .. code-block:: powershell

        > cmake --list-presets
        Available configure presets:

        "clang-win-x64-debug"     - Clang x64 Windows Debug
        "clang-win-x64-release"   - Clang x64 Windows Release
        "clang-win-x64-unopt"     - Clang x64 Windows Unoptimized

.. tab:: Linux

    .. code-block::

        $ cmake --list-presets
        Available configure presets:

        "clang-linux-x64-debug"     - Clang x64 Linux Debug
        "clang-linux-x64-release"   - Clang x64 Linux Release
        "clang-linux-x64-unopt"     - Clang x64 Linux Unoptimized

The examples below show a complete listing of commands required to use CMake to build *vortex* with Python support.
These commands may be run within a Python virtual environment.

.. tab:: Windows

    This example illustrates how to build and install *vortex* from a PowerShell prompt using *clang-cl*.
    It is assumed that Python has already been installed and is available on the path.
    After cloning *vortex*, customize the build by creating ``CMakeUserPresets.json`` or editing ``CMakePresets.json`` in *vortex*'s source tree.

    .. code-block:: powershell

        # setup vcpkg
        > git clone https://github.com/microsoft/vcpkg.git vcpkg
        > .\vcpkg\bootstrap-vcpkg.bat -disableMetrics
        > $env:CMAKE_TOOLCHAIN_FILE="$PWD\vcpkg\scripts\buildsystems\vcpkg.cmake"

        # setup vortex
        > git clone https://gitlab.com/vortex-oct/vortex.git
        > $Python = (Get-Command python).Path
        > & $Python -m pip install -r vortex/requirements.txt

        # build and install vortex
        > cmake -S vortex -B vortex/build --preset clang-win-x64-release
        > cmake --build vortex/build

.. tab:: Linux

    This example illustrates how to build and install *vortex* on Ubuntu 22.04.
    It is assumed that closed-source dependencies have been installed into standard locations.
    After cloning *vortex*, customize the build by creating ``CMakeUserPresets.json`` or editing ``CMakePresets.json`` in *vortex*'s source tree.

    .. code-block:: bash

        # install OS packages
        $ sudo apt install curl zip unzip tar pkg-config cmake ninja-build clang patchelf python3 python3-pip

        # setup vcpkg
        $ git clone https://github.com/microsoft/vcpkg.git vcpkg
        $ ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
        $ export CMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake"

        # setup vortex
        $ git clone https://gitlab.com/vortex-oct/vortex.git
        $ pip3 install -r vortex/requirements.txt

        # configure and build
        $ cmake -S vortex -B vortex/build --preset clang-linux-x64-release
        $ cmake --build vortex/build

    The ``curl``, ``zip``, ``unzip``, ``tar``, and ``pkg-config`` packages are required for *vcpkg*.
    The ``cmake``, ``ninja-build``, ``clang``, and ``patchelf`` are required for *vortex*'s build system.
    The ``python3`` and ``python3-pip`` packages are required for Python support with *vortex*.

.. _python-setup-script:

Python Setup Script
^^^^^^^^^^^^^^^^^^^

*vortex*'s Python setup script (``setup.py``) executes the CMake compilation procedure above.
If you plan to use *vortex* exclusively from Python, the instructions in :ref:`python-from-source` are recommended.
The continuous integration system uses this approach to create the pre-compiled Python wheels.
