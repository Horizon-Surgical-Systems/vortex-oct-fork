.. _getting-started:

Getting Started
===============

*vortex* is available in Python and C++.
The Python bindings via a binary wheel is the fastest way to get started.

.. tip::
    See the :ref:`Python demo guide <demo/python>` for high-level step-by-step instructions to get started quickly.

.. _getting-started/python:

Python from Binary Wheel
------------------------

Pre-built binary wheels for recent Python and CUDA versions on Windows are published on the *vortex* `website`_.
Ensure that you install the wheel that matches the CUDA version for your system.
*vortex* uses the suffix ``-cudaXXY`` to denote a build for CUDA ``XX.Y``.

.. note::

    For CUDA 11 and higher, *vortex* supports CUDA forward compatibility between minor versions.
    In these cases, ``Y`` has the value ``x`` (e.g., ``vortex-cuda12x``) to indicate support for all CUDA releases for a given major version.

Installation of ``vortex-oct-tools`` is also recommended as it provides Python support classes for interacting with *vortex* and is required for certain demos.

.. _website: https://www.vortex-oct.dev/#releases

.. tab:: Windows

    .. code-block:: powershell

        > pip install vortex-oct-cuda12x vortex-oct-tools --extra-index-url https://vortex-oct.dev/stable

.. tab:: Linux

    .. code-block:: bash

        $ pip3 install vortex-oct-cuda12x vortex-oct-tools --extra-index-url https://vortex-oct.dev/stable

You can install developments builds of *vortex* for testing new features by using ``--extra-index-url https://vortex-oct.dev/develop`` instead.

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

These binary wheels are compiled with support AlazarTech and Teledyne ADQ digitizers, CUDA-compatible GPUs, and National Instruments I/O and camera interface cards.
CUDA is a mandatory dependency for the binary wheels and must be installed for *vortex* to load.
All other hardware components are optional so you need install only the driver and runtime components for each of those specific devices that you plan to use.

.. tab:: Windows

    **Mandatory**

    -   CUDA_ runtime with version ``XX.Y`` for ``vortex-oct-cudaXXY``

    **Optional**

    -   `AlazarTech <AlazarTechWindows_>`_ drivers for the installed digitizer

        .. attention::

            The AlazarTech drivers are different from ATS-SDK, which is the software development kit only.
            ATS-SDK is required for building *vortex* and does not include the drivers.
            The AlazarTech drivers are required to run *vortex*.

    -   Teledyne ADQAPI_ runtime and drivers for the installed digitizer

    -   `NI DAQmx`_ runtime

    -   `NI IMAQ`_ runtime

.. tab:: Linux

    .. tip::

        See the :ref:`build guide <closed-source>` for help installing the Linux dependencies.

    **Mandatory**

    -   CUDA runtime with at least version ``XX.Y`` for ``vortex-oct-cudaXXY``

    **Optional**

    -   `AlazarTech <AlazarTechLinux_>`_ drivers for the installed digitizer

        .. attention::

            The AlazarTech drivers are different from ATS-SDK, which is the software development kit only.
            ATS-SDK is required for building *vortex* and does not include the drivers.
            The AlazarTech drivers are required to run *vortex*.

    -   Teledyne `ADQAPI`_ runtime and drivers for the installed digitizer

        .. attention::

            Binary releases of *vortex* are compiled with ADQAPI version 2023.2.
            Teledyne features may not load for runtime or driver versions other than 2023.2.

    -   `NI DAQmx`_ runtime

        .. attention::

            Due to a `known issue with DAQmx on Linux <https://www.ni.com/en-us/support/documentation/bugs/23/ni-linux-device-drivers-2023-q1-known-issues.html>`_, you may need to add ``iommu=off`` to your kernel command line.

With the exception of CUDA runtime, *vortex* will detect the presence of these driver and runtime components and will activate the corresponding functionality.
See the :ref:`build-guide` for instructions on building Python bindings that match your hardware availability.

.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _AlazarTechWindows: https://www.alazartech.com/en/
.. _AlazarTechLinux: ftp://release@ftp.alazartech.com/outgoing/linux
.. _ADQAPI: https://www.spdevices.com/what-we-do/products/software
.. _`NI DAQmx`: https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html
.. _`NI IMAQ`: https://www.ni.com/en-us/support/downloads/drivers/download.vision-acquisition-software.html

Check Installation
^^^^^^^^^^^^^^^^^^

Check that *vortex* and its dependencies are installed correctly using the following command.
The command outputs ``OK``, the *vortex* version, and available features if *vortex* is correctly installed.

.. tab:: Windows

    .. code-block:: powershell

        > python -c "import vortex; print('OK', vortex.__version__, vortex.__feature__)"
        OK 0.5.1 ['machdsp', 'reflexxes', 'fftw', 'cuda', 'hdf5', 'cuda_dynamic_resampling', 'alazar', 'alazar_dac', 'daqmx', 'imaq', 'teledyne', 'simple', 'exception_guards', 'pybind11_optimizations']

.. tab:: Linux

    .. code-block:: bash

        $ python3 -c "import vortex; print('OK', vortex.__version__, vortex.__feature__)"
        OK 0.5.1 ['machdsp', 'reflexxes', 'fftw', 'cuda', 'hdf5', 'cuda_dynamic_resampling', 'alazar', 'alazar_dac', 'daqmx', 'imaq', 'teledyne', 'simple', 'exception_guards', 'pybind11_optimizations']

If an expected feature is absent from the list, ensure that you installed the correct driver or runtime component above.
For more detailed investigation, see the :ref:`debugging steps <debug-dynamic>`.
Once everything is installed, :ref:`try running a demo <demo/python>`.

.. _python-from-source:

Python from Source
------------------

The *vortex* setup script is capable of installing all open-source build dependencies using *vcpkg* and generating wheels for installation.
Building *vortex* from source is only recommended if the binary wheels are not suitable for your hardware.

.. _configure-build-environment:

Configure Build Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab:: Windows

    The *vortex* setup script requires the Visual Studio Developer PowerShell (or Prompt) for building with *clang-cl*.
    Make sure to install `Clang/LLVM support for Visual Studio <https://docs.microsoft.com/en-us/cpp/build/clang-support-msbuild?view=msvc-160>`_.
    You can activate the Developer PowerShell from an existing PowerShell session as follows.

    .. code-block:: powershell

        > Import-Module C:"\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
        > Enter-VsDevShell -VsInstallPath "C:\Program Files\Microsoft Visual Studio\2022\Community" -DevCmdArguments "-arch=x64"

.. tab:: Linux

    The *vortex* build script requires ``cmake``, ``ninja``, and ``clang`` for building.
    These and other required packages are available through the system package manager.

    .. code-block:: bash

        $ sudo apt install curl zip unzip tar pkg-config cmake ninja-build clang patchelf python3 python3-pip

Configure Features
^^^^^^^^^^^^^^^^^^

You can enable or disable features using the ``VCPKG_MANIFEST_FEATURES`` environment variable, which is a semicolon-delimited list of *vcpkg* features, especially those that :ref:`enable or disable features <dependency-control>`.
The setup script parses the feature specification and automatically installs the required packages from *vcpkg*.
:ref:`Closed-source dependencies <closed-source>`, such as `CUDA`_ and `NI DAQmx`_, still require manual installation as for the binary wheel above.

.. tab:: Windows

    .. code-block:: powershell

        > $env:VCPKG_MANIFEST_FEATURES="asio;backward;cuda;reflexxes;fftw;hdf5;python"

.. tab:: Linux

    .. code-block:: bash

        $ export VCPKG_MANIFEST_FEATURES="asio;backward;cuda;reflexxes;fftw;hdf5;python"

If you do not want the setup script to install dependencies using *vcpkg*, define the environment variable ``VORTEX_DISABLE_AUTO_VCPKG`` or specify ``CMAKE_TOOLCHAIN_FILE`` or ``CMAKE_INSTALL_PREFIX`` as environment variables.

Build with Setup Script
^^^^^^^^^^^^^^^^^^^^^^^

*vortex* source distributions lack the ``-cudaXXY`` suffix and are thereby distinguished from the binary wheels.
You may clone the repository or `download the source <download-source_>`_, and then run the setup script.

.. tab:: Windows

    .. code-block:: powershell

        > git clone https://gitlab.com/vortex-oct/vortex.git
        > pip install ./vortex

.. tab:: Linux

    .. code-block:: bash

        $ git clone https://gitlab.com/vortex-oct/vortex.git
        $ pip3 install ./vortex

Alternatively, you may install directly from the repository.

.. tab:: Windows

    .. code-block:: powershell

        > pip install git+https://gitlab.com/vortex-oct/vortex.git

.. tab:: Linux

    .. code-block:: powershell

        $ pip3 install git+https://gitlab.com/vortex-oct/vortex.git



C++
---

The build procedure varies based on the use-case for *vortex*.

As an Application Dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most users will want to include *vortex* as a dependency in a larger application.
See the *vortex* `application example <https://gitlab.com/vortex-oct/vortex-app-example>`_ for a walkthrough.

As a Library for Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compile *vortex* directly for development or customization purposes, follow the steps below.

#.  Install and/or setup *vcpkg* for *vortex*'s C++ dependencies.
    Avoiding using shallow clones of *vcpkg* (e.g., with ``--depth``) as this can cause problems when installing packages with the specific package versions specified in *vortex*'s manifest file.

    .. tab:: Windows

        .. code-block:: powershell

            > git clone https://github.com/microsoft/vcpkg.git
            > ./vcpkg/bootstrap-vcpkg.bat -disableMetrics


    .. tab:: Linux

        .. code-block:: bash

            $ git clone https://github.com/microsoft/vcpkg.git
            $ ./vcpkg/bootstrap-vcpkg.sh -disableMetrics

#.  Export the path to *vcpkg*'s build system for CMake.
    Use an absolute path to avoid unexpected build failures in the future.

    .. tab:: Windows

        .. code-block:: powershell

            > $env:CMAKE_TOOLCHAIN_FILE=C:/.../vcpkg/scripts/buildsystems/vcpkg.cmake

    .. tab:: Linux

        .. code-block:: bash

            $ export CMAKE_TOOLCHAIN_FILE="/.../vcpkg/scripts/buildsystems/vcpkg.cmake"

    Alternatively, set ``CMAKE_TOOLCHAIN_FILE`` as a cache variable as shown below.

#.  Clone *vortex*.

    .. tab:: Windows

        .. code-block:: powershell

            > git clone https://gitlab.com/vortex-oct/vortex.git

    .. tab:: Linux

        .. code-block:: bash

            $ git clone https://gitlab.com/vortex-oct/vortex.git

#.  Edit the cache variables for the top-level CMake preset (``base-x64``) for building *vortex* in ``vortex/CMakePresets.json``.
    Configure the *vortex* features to build using the ``VCPKG_MANIFEST_FEATURES`` cache variable.
    Also, you can define the ``CMAKE_TOOLCHAIN_FILE`` cache variable here rather than setting it as an environment variable above.

    .. code-block:: json

        "cacheVariables": {
          "CMAKE_INSTALL_PREFIX": "${sourceDir}/install/${presetName}"
          "CMAKE_TOOLCHAIN_FILE": "/.../vcpkg/scripts/buildsystems/vcpkg.cmake",
          "VCPKG_MANIFEST_FEATURES": "asio;backward;cuda;reflexxes;fftw;hdf5;python"
        },

    Find a complete list of feature names using ``vcpkg search vortex --featurepackages`` or in the *vortex* `port manifest <https://gitlab.com/vortex-oct/vortex-registry/-/blob/master/ports/vortex/vcpkg.json?ref_type=heads>`_.
    See the documentation for additional `build options <https://www.vortex-oct.dev/rel/latest/doc/develop/build/#build-options>`_.

#.  Choose and customize the build preset for your system.
    *vortex* provides presets for building with *Clang-CL* on Windows and *clang* or *gcc* on Linux in debug, unoptimized, or release modes.

    -   **Debug** mode builds with debug versions of all libraries and includes symbols.
        This mode is generally not compatible with Python support since you will need to have a debug version of NumPy installed.
    -   **Unoptimized** mode builds with release versions of all libraries but turns debug symbols on and optimizations off for *vortex*.
        This is the recommended mode for debugging *vortex* if Python features are enabled.
    -   **Release** mode builds with release versions of all libraries and turns on all optimizations for *vortex*.
        Debug symbols are included for *vortex* if not optimized out.

    Edit the presets with names that start with ``base-`` so that changes propagate to debug, unoptimized, and release builds.

    .. code-block:: json

        "name": "base-clang-win-x64",
        "inherits": "base-x64",
        "generator": "Ninja",
        "environment": {
            "CC": "clang-cl",
            "CXX": "clang-cl",
            "CXXFLAGS": "-m64 -fdiagnostics-absolute-paths -fcolor-diagnostics -Wno-unused-command-line-argument /Zi",
            "CUDAFLAGS": "--expt-relaxed-constexpr -lineinfo -Xcudafe --diag_suppress=base_class_has_different_dll_interface -diag-suppress 27 -Wno-deprecated-gpu-targets -Xcompiler=/wd4984 --debug",
            "LDFLAGS": "/debug"
        },
        "cacheVariables": {
            "CMAKE_CUDA_STANDARD": "17"
        },

#.  If you plan to build with Python support, install the Python requirements first.

    .. tab:: Windows

        .. code-block:: powershell

            > pip install -r vortex/requirements.txt

    .. tab:: Linux

        .. code-block:: bash

            $ pip3 install -r vortex/requirements.txt

#.  Build *vortex* using your chosen environment.

    -   For Visual Studio, open the root folder for *vortex* as a CMake project.
        Select the correct preset from the configuration dropdown in the toolbar and then build.
    -   For Visual Studio Code, open the root folder for *vortex*.
        Make sure to install the C++ and CMake extensions.
        Select the correct configure preset using the CMake extension and then build.
    -   For CMake from the command line, configure using the selected preset.

        .. tab:: Windows

            .. code-block:: powershell

                > cmake -S vortex -B vortex/build --preset clang-win-x64-release
                > cmake --build vortex/build

        .. tab:: Linux

            .. code-block:: bash

                $ cmake -S vortex -B vortex/build --preset gcc-linux-x64-release
                $ cmake --build vortex/build

    If you edit the CMake preset, delete your CMake cache and reconfigure to ensure that the changes take effect.
