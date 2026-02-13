Vortex
======

`vortex` is a high-performance C++ library with Python bindings for building real-time OCT engines. It provides:

- modular components out of which an OCT engine is built
- a flexible OCT engine that accommodates most use cases
- extensive Python bindings that interoperate with NumPy and CuPy

`vortex` is designed from the ground up to:

- promote reproducibility in data acquisition and processing
- guarantee synchronization between data streams
- support on-the-fly changes for interactive use cases
- integrate cleanly with user interfaces
- facilitate rapid prototyping with high-level abstractions and sane defaults
- minimize restrictions and assumptions

Documentation
-------------

Documentation is available for `the latest release <https://www.vortex-oct.dev/rel/latest/doc/>`_ and for `specific versions <https://www.vortex-oct.dev/#releases>`_.

Getting Started
---------------

Python
^^^^^^

For an official release, install with ``pip``, where the suffix ``-cudaXXY`` denotes a version for CUDA ``XX.Y``.

.. code-block:: powershell

    > pip install vortex-oct-cuda12x --index-url https://vortex-oct.dev/stable

    $ pip3 install vortex-oct-cuda12x --index-url https://vortex-oct.dev/stable

To test a development build, use ``https://vortex-oct.dev/develop`` as the index URL for ``pip``.
Visit the `online documentation <https://www.vortex-oct.dev/rel/latest/doc/getting-started/#python-from-binary-wheel>`__ for more details.


C++
^^^

The build procedure varies based on the use-case for *vortex*.
Visit the `online documentation <https://www.vortex-oct.dev/rel/latest/doc/getting-started/#c>`_ for full details.

As an Application Dependency
++++++++++++++++++++++++++++

Most users will want to include *vortex* as a dependency in a larger application.
See the *vortex* `application example <https://gitlab.com/vortex-oct/vortex-app-example>`_ for a walkthrough.

As a Library for Development
++++++++++++++++++++++++++++

To compile *vortex* directly for development or customization purposes, follow the steps below.

#.  Install and/or setup *vcpkg* for *vortex*'s C++ dependencies.
    Avoiding using shallow clones of *vcpkg* (e.g., with ``--depth``) as this can cause problems when installing packages with the specific package versions specified in *vortex*'s manifest file.

    .. code-block:: powershell

        > git clone https://github.com/microsoft/vcpkg.git
        > ./vcpkg/bootstrap-vcpkg.bat -disableMetrics


    .. code-block:: bash

        $ git clone https://github.com/microsoft/vcpkg.git
        $ ./vcpkg/bootstrap-vcpkg.sh -disableMetrics

#.  Export the path to *vcpkg*'s build system for CMake.
    Use an absolute path to avoid unexpected build failures in the future.

    .. code-block:: powershell

        > $env:CMAKE_TOOLCHAIN_FILE=C:/.../vcpkg/scripts/buildsystems/vcpkg.cmake

    .. code-block:: bash

        $ export CMAKE_TOOLCHAIN_FILE="/.../vcpkg/scripts/buildsystems/vcpkg.cmake"

    Alternatively, set ``CMAKE_TOOLCHAIN_FILE`` as a cache variable as shown below.

#.  Clone *vortex*.

    .. code-block:: powershell

        > git clone https://gitlab.com/vortex-oct/vortex.git

    .. code-block:: bash

        $ git clone https://gitlab.com/vortex-oct/vortex.git


#.  Edit the cache variables for the top-level CMake preset (``base-x64``) for building *vortex* in ``vortex/CMakePresets.json``.
    Configure the *vortex* features to build using the ``VCPKG_MANIFEST_FEATURES`` cache variable.
    Also, you can define the ``CMAKE_TOOLCHAIN_FILE`` cache variable here rather than setting it as an environment variable above.

    .. code-block:: json

        "cacheVariables": {
          "CMAKE_INSTALL_PREFIX": "${sourceDir}/install/${presetName}"
          "CMAKE_TOOLCHAIN_FILE": "/.../vcpkg/scripts/buildsystems/vcpkg.cmake",
          "VCPKG_MANIFEST_FEATURES": "reflexxes;cuda;python;teledyne;alazar;alazar-dac;daqmx;asio;backward"
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

    .. code-block:: powershell

        > pip install -r vortex/requirements.txt

    .. code-block:: bash

        $ pip3 install -r vortex/requirements.txt

#.  Build *vortex* using your chosen environment.

    -   For Visual Studio, open the root folder for *vortex* as a CMake project.
        Select the correct preset from the configuration dropdown in the toolbar and then build.
    -   For Visual Studio Code, open the root folder for *vortex*.
        Make sure to install the C++ and CMake extensions.
        Select the correct configure preset using the CMake extension and then build.
    -   For CMake from the command line, configure using the selected preset.

        .. code-block:: powershell

            > cmake -S vortex -B vortex/build --preset clang-win-x64-release
            > cmake --build vortex/build

        .. code-block:: bash

            $ cmake -S vortex -B vortex/build --preset gcc-linux-x64-release
            $ cmake --build vortex/build

    If you edit the CMake preset, delete your CMake cache and reconfigure to ensure that the changes take effect.

License
-------

*vortex* is released under the permissive BSD-3 license to support research and commercial use.
