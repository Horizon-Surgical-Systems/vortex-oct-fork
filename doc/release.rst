Release Notes
=============

v0.5.1 -- 1/17/2026
--------------------

Changelog
+++++++++

New Features
^^^^^^^^^^^^

- Add experimental support for Teledyne on-board background subtraction, spectral filtering, and FFT with FWOCT firmware
- Add experimental support for using hugepages with Teledyne acquisitions on Linux :attr:`TeledyneConfig.enable_hugepages <vortex.acquire.TeledyneConfig.enable_hugepages>`
- Add experimental support for Alazar DAC modules with :class:`~vortex.io.AlazarIO`
- Add experimental support for digital servo drivers with :class:`~vortex.io.MachDSPIO`
- Add support for NRRD and NIfTI formats in :class:`~vortex.storage.SimpleStack` with :attr:`SimpleStackHeader.NRRD <vortex.storage.SimpleStackHeader.NRRD>` and :attr:`SimpleStackHeader.NIfTI <vortex.storage.SimpleStackHeader.NIfTI>`
- Add :class:`~vortex.log.Logger` for fine-grained control over log output from Python
- Add :attr:`CPUProcessorConfig.enable_magnitude <vortex.process.CPUProcessorConfig.enable_magnitude>` to control conversion of complex to real values
- Add :attr:`CPUProcessorConfig.levels <vortex.process.CPUProcessorConfig.levels>` for control of output quantization
- Add ``block_callback`` for endpoints that reports all segments which complete within a block
- Add :meth:`Engine.shutdown() <vortex.engine.Engine.shutdown>` to request engine shutdown without blocking

Improvements
^^^^^^^^^^^^

- Improve handling of dual-edge sampling configuration for Alazar digitizers
- Improve specificity and actionability of configuration error messages
- Show critical error messages in module loader even when ``VORTEX_LOADER_LOG`` is not set
- Add warning when Python wheel is compiled in debug mode
- Rename storage headers to avoid confusion
- Extract version detection routines from top-level CMake scripts into separate files

Build System
^^^^^^^^^^^^

- Update to *vcpkg* manifest mode for handling build dependencies
- Provide *vcpkg* manifest for building *vortex* as an external library
- Create *vcpkg* `registry <https://gitlab.com/vortex-oct/vortex-registry>`_ for *vortex* build dependencies not available in main *vcpkg* registry
- Add support for driving build feature flags from CMake or *vcpkg*
- Build Python wheels against updated versions of all dependencies
- Build Python wheels from standards-compliant source distributions
- Add Python wheels for Python 3.13 and 3.14
- Build Python wheels with support for NumPy 2 for Python 3.9+
- Drop builds for CUDA 10.2 and compute capability 3.5
- Update CUDA 11 builds to use CUDA 11.8
- Update stub generation to use ``mypy`` without patching
- Remove outdated CMake functions for wrapping compiler flags for CUDA
- Set CMake ``*_FOUND`` variables in dependency find scripts
- Simplify handling of include paths in CMake
- Change CMake-driven defaults to enable modular build only if Python support is built
- Change CMake-driven defaults to not automatically install Python wheels
- Support empty build suffixes

Documentation
^^^^^^^^^^^^^

- Add `example project <https://gitlab.com/vortex-oct/vortex-app-example>`_ for using *vortex* as a *vcpkg* dependency
- Add debugging help for scan pattern limit violations
- Add creation of documentation to build server
- Add documentation of :ref:`profiler events <develop/profiler/events>`
- Add version selector to documentation
- Add dynamics warnings for outdated or development documentation versions
- Add light theme variants of figures and plots
- Change diagrams from PNG to SVG

Assorted
^^^^^^^^

- Fix bug that used non-strided CUDA copies when saving strided spectral buffers
- Add tensor stride accessor that keeps singular dimensions
- Fix inefficient uses of background logging threads
- Fix potential deadlocks in Python logging sink
- Fix bug in scan formatting in which volume and scan boundaries could be ignored in interleaved scans
- Rename storage headers to avoid confusion
- Fix race condition in construction of logger for module loader
- Fix percent/ratio conversion bug that incorrectly configured Alazar external clock levels too low
- Fix incorrect block sizing in tutorials 1 and 2
- Fix incorrect handling of exceptions during module loading that could suppress error messages and/or produce program termination
- Update CUDA standard for C++17
- Reduce CUDA compilation warnings

Migration Guide
+++++++++++++++

Logging
^^^^^^^

For Python users, :meth:`~vortex.get_console_logger` and :meth:`~vortex.get_python_logger` previously returned an existing logger if a logger with the requested name already existed.
These functions now always create a new logger.
If you wish to reuse an existing logger, keep a reference to the :class:`~vortex.log.Logger` returned by these functions.
The same semantics apply to the new function :meth:`~vortex.get_file_logger`.
In general, prefer constructing a :class:`~vortex.log.Logger` directly instead of using these convenience functions.

Include Changes
^^^^^^^^^^^^^^^

For C++ users, the headers for ``stream_dump_t`` have moved as per the table below to avoid confusion.
No other code changes are required.

============================= ===========================
Prior Name                    New Name
============================= ===========================
``vortex/storage/stream.hpp`` ``vortex/storage/dump.hpp``
============================= ===========================

Build System
^^^^^^^^^^^^

- The build system uses *vcpkg* manifest mode to simplify dependency management.
  A custom *vcpkg* registry for *vortex* has replaced the prior *vcpkg* overlay.
  All packages previously included in the overlay and now in the registry.
  The bundled *vcpkg* `configuration <https://gitlab.com/vortex-oct/vortex/-/blob/develop/vcpkg-configuration.json?ref_type=heads>`_ will automatically retrieve packages from the registry.
  Consult the :ref:`build guide <build-guide>` for updated build instructions.

- The standardized ``VCPKG_MANIFEST_FEATURES`` environment variable has now replaced ``VORTEX_BUILD_FEATURES`` environment variable.
  Whereas ``VORTEX_BUILD_FEATURES`` was a semicolon-delimited list of CMake ``WITH_*`` flags, ``VCPKG_MANIFEST_FEATURES`` is a semicolon-delimited list of features described in *vortex*'s *vcpkg* `manifest <https://gitlab.com/vortex-oct/vortex/-/blob/develop/vcpkg.json?ref_type=heads>`_.
  In general, a CMake option ``WITH_ABC_XYZ`` would correspond to *vcpkg* feature ``abc-xyz``.

Configuration Defaults
^^^^^^^^^^^^^^^^^^^^^^

- For CMake-driven builds, ``INSTALL_PYTHON_WHEEL`` now defaults to ``OFF`` so any built Python wheels are not automatically installed.
- The CMake variable ``ENABLE_MODULAR_BUILD`` now defaults to the value of ``WITH_PYTHON`` instead of ``BUILD_PYTHON_WHEEL``.

Removed Functionality
^^^^^^^^^^^^^^^^^^^^^

Support for non-standard raw headers in storage objects has been removed.
See the table below for recommended alternative headers.
Changing the headers will necessitate minor changes to code that loads the saved data.

====================================================================== ==========================================================================
Removed Header                                                         Alternative Header
====================================================================== ==========================================================================
:attr:`SimpleStreamHeader.Raw <vortex.storage.SimpleStreamHeader.Raw>` :attr:`SimpleStreamHeader.Empty <vortex.storage.SimpleStreamHeader.Empty>`
:attr:`SimpleStackHeader.Raw <vortex.storage.SimpleStackHeader.Raw>`   :attr:`SimpleStackHeader.NumPy <vortex.storage.SimpleStackHeader.NumPy>`
====================================================================== ==========================================================================

v0.5.0 -- 1/25/2025
--------------------

Changelog
+++++++++

New Features
^^^^^^^^^^^^

- Add official support for Linux
- Add experimental support for Teledyne ADQ cards with :class:`~vortex.acquire.TeledyneAcquisition`
- Add :class:`~vortex.process.CPUProcessor` for FFTW-based OCT processing
- Add :ref:`engine profiler <develop/profiler>`
- Add stack trace reporting with unhandled exceptions
- Add support for strided memory operations
- Add experimental support for GPU k-clock resampling on per A-scan basis for CUDA 11 and higher (feature ``cuda_dynamic_resampling``)

Improvements
^^^^^^^^^^^^

- Support full set of callbacks from :class:`~vortex.engine.NullEndpoint`
- Issue callbacks for endpoints in order of registration
- Change stream storage endpoints (e.g., :class:`~vortex.engine.AscanStreamEndpoint`) to honor flags and issue callbacks
- Update demos to use new and faster display widgets
- Add engine-generated markers at scan start when using leading samples
- Add stream and stack storage endpoints for all available data streams
- Avoid de-interleaving multi-channel data prior to processing
- Allow disabling of post-FFT square in OCT processors
- Support signed raw data in OCT processors in Python with :attr:`CUDAProcessorConfig.interpret_as_signed <vortex.process.CUDAProcessorConfig.interpret_as_signed>`
- Allow disabling of *pybind11* optimizations with ``ENABLE_PYBIND11_OPTIMIZATIONS`` CMake option to facilitate debugging
- Add accessor for device index to CUDA device tensors
- Add option to release DAQmx tasks when acquisition stops with :attr:`~vortex.io.DaqmxIo.persistent_task`
- Reorganized acquisition component namespaces for consistency (see Migration Guide)
- Allow use of Alazar cards not present in feature database
- Include sample skipping support in Alazar feature database
- Provide access to Alazar board handles with :attr:`~vortex.acquire.AlazarAcquisition.board` and :attr:`~vortex.acquire.alazar.Board.handle`
- Open file during preparation phase instead of start phase in :class:`~vortex.acquire.FileAcquisition`
- Add recycling stage to engine processing to support non-owned memory buffers
- Increase default data type for resampling index from 16-bits to 32-bits to support long spectra
- Include compile-time options (e.g., ``ENABLE_EXCEPTION_GUARDS``) in Python module features (e.g., ``__feature__``) for introspection

Build System
^^^^^^^^^^^^

- Add support for Python 3.11 and 3.12
- Add support for CUDA 11.x and 12.x using forward compatibility
- Rewrite CI build system for more robust and flexibility building of wheels
- Add ``manylinux2014_x86_64`` compliance for Linux wheels for portability
- Detect CUDA version in Python-driven builds to produce correct package suffix
- Support builds with GCC on Linux
- Pull Python setup script flags from CMake presets
- Build against oldest supported ``numpy`` for compatibility
- Fix dependency deployment and rpaths on Linux
- Add tests for memory management and tensors
- Add tests for OCT resampling and filtering
- Add CMake presets for unoptimized builds with GCC and Clang
- Remove dependency of engine on Alazar and DAQmx support
- Fix handling of compile time definitions in modular builds that produced builds errors with certain feature combinations
- Fix bug in handling of Alazar SDK version for error messages

Assorted
^^^^^^^^

- Fix bug when formatting to closed storage objects
- Improve logging for storage objects
- Fix bug that disabled support for Alazar on-board FFT
- Fix whitespace typo in *vcpkg* ports overlay that corrupted TBB port
- Fix dimension promotion bug in memory copy utilities
- Fix type safety for FFTW wrapper
- Fix delayed segment processing at volume boundaries in host tensor endpoints (e.g., :class:`~vortex.engine.SpectraStackHostTensorEndpoint`).
- Fix bug in scan change workaround that could produce infinite loop
- Change defaults of :class:`~vortex.format.FormatPlanner` to satisfy the most common use cases
- Improve robustness of C++ exception translation for Python
- Fix bug in engine utilization computation
- Fix assorted bugs in CPU processor
- Fix bug that unconditionally disabled engine plan logging

Migration Guide
+++++++++++++++

Namespace Changes
^^^^^^^^^^^^^^^^^

For C++ users, classes and enumerations for Alazar, DAQmx, IMAQ support have moved from ``vortex::acquire`` or ``vortex::io`` to ``vortex::alazar``, ``vortex::daqmx``, and ``vortex::imaq``, respectively.
For example, ``vortex::acquire::clock::internal_t`` is now ``vortex::alazar::clock::internal_t``.
This change avoids name collision with other acquisition components.

========================================= ==============================================
Prior Namespace                           New Namespace
========================================= ==============================================
``vortex::acquire::clock::internal_t``    ``vortex::acquire::alazar::clock::internal_t``
``vortex::acquire::frame_trigger_t``      ``vortex::acquire::imaq::frame_trigger_t``
``vortex::io::channel::digital_output_t`` ``vortex::daqmx::channel::digital_output_t``
...                                       ...
========================================= ==============================================

For Python users, objects for Alazar, DAQmx, IMAQ support have moved from ``vortex.acquire`` or ``vortex.io`` to ``vortex.acquire.alazar``, ``vortex.io.daqmx``, and ``vortex.acquire.imaq``.
Objects previously in the driver module (e.g., ``vortex.driver.alazar``) have merged with these new modules.
For example, ``vortex.acquire.InternalClock`` is now ``vortex.acquire.alazar.InternalClock`` and ``vortex.driver.alazar.Channel`` is now ``vortex.acquire.alazar.Channel``.
This change parallels C++ namespaces and simplifies Python imports.

========================================= ==============================================
Prior Module                              New Module
========================================= ==============================================
``vortex.acquire.InternalClock``          ``vortex.acquire.alazar.InternalClock``
``vortex.acquire.FrameTrigger``           ``vortex.acquire.imag.FrameTrigger``
``vortex.io.AnalogOutput``                ``vortex.io.daqmx.AnalogOutput``
...                                       ...
========================================= ==============================================

Functionality Changes
^^^^^^^^^^^^^^^^^^^^^

Stream storage endpoints (e.g., :class:`~vortex.engine.AscanStreamEndpoint`) now honor the flags setting of their formatter, just like stack storage endpoints (e.g., :class:`~vortex.engine.AscanStackEndpoint`).
The formatter defaults have been changed such that the original behavior is provided by default unless flags are customized.
Users who relied upon these endpoints to unconditionally store all samples should ensure that the endpoints are associated with a formatter than matches all flags (i.e., the default).
Users who do not customize flags are not affected by this change.

Configuration Defaults
^^^^^^^^^^^^^^^^^^^^^^

The default values for several configuration fields has changed as listed below.
Users who do not set these fields and desire the original behavior should explicitly assign the prior default value.
Users who already set these fields are not affected.

======================================================================================= ============================================== ===============================================
Field                                                                                   Prior Default                                  New Default
======================================================================================= ============================================== ===============================================
:data:`FormatPlannerConfig.adapt_shape <vortex.format.FormatPlannerConfig.adapt_shape>` ``True``                                       ``False``
:data:`FormatPlannerConfig.shape <vortex.format.FormatPlannerConfig.shape>`             ``(0, 0)``                                     ``(2**64-1, 2**64-1)``
:data:`SimpleStreamConfig.header <vortex.storage.SimpleStreamConfig.header>`            :data:`~vortex.storage.SimpleStreamHeader.Raw` :data:`~vortex.storage.SimpleStreamHeader.Empty`
======================================================================================= ============================================== ===============================================

v0.4.3 -- 6/30/2022
-------------------

Changelog
+++++++++

Hardware
^^^^^^^^

- Add support for NI IMAQ cards with :class:`~vortex.acquire.ImaqAcquisition`
- Improve feature detection for Alazar cards
- Add calibration step to reduce ADC noise for select Alazar cards
- Fix dual edge sampling for Alazar cards
- Add strobe generation to engine with :data:`EngineConfig.strobes <vortex.engine.EngineConfig.strobes>`
- Transition to dynamically loaded hardware support modules

Documentation
^^^^^^^^^^^^^

- Reorganization of documentation with expansion for acquisition and processing components
- Parse docstrings from documentation and embed in compiled module
- Add galvo delay demo and :ref:`tuning tool <how-to/io-delay>`
- Add new demo for :ref:`saving data to disk <demo/acquire-to-disk>`
- Add tutorial for :ref:`UI development and scan switching <tutorial/display>`
- Improvements and bugfixes for demos

Build System
^^^^^^^^^^^^

- Drop builds for Python 3.6
- Add builds for CUDA 11.6
- Refactor CMake to support modular builds
- Fix issue that led to incorrect type stub output
- Add experimental support for Linux
- Explicitly set minimum required NumPy version
- Improve version detection and handling for Alazar and Vortex

Assorted
^^^^^^^^

- Fix issue where non-preloadable acquisition components produced incorrect startup sequences, leading to loss of synchronization
- Exceptions that cause premature engine shutdown are now propagated
- Add :data:`~vortex.engine.Block.StreamIndex.Counter` stream to support correct index bookkeeping in post-processing with :class:`~vortex.engine.CounterStackEndpoint` and related endpoints.
- Add endpoints for formatting and storing streams (e.g., :class:`~vortex.engine.GalvoActualStackHostTensorEndpoint`)
- Refactor spiral scan generation
- Fix bug that prevented change of repeated scan strategies
- Add multiple inactive segment generation policies
- Add CPU and GPU memory endpoints for spectra (e.g., :class:`~vortex.engine.SpectraStackDeviceTensorEndpointUInt16`)
- Assorted internal improvements and bugfixes

Migration Guide
+++++++++++++++

No migration required.

v0.4.2 -- 3/3/2022
------------------

Changelog
+++++++++

- Update database for newer Alazar cards
- Add support for tolerating unknown Alazar cards
- Add Python ``setup.py`` script for building *vortex*, including all dependencies with *vcpkg*
- Fix missing constructor for :class:`~vortex.engine.Source`
- Include imaging depth in :class:`~vortex.engine.Source`
- Build for Python 3.10
- Improvements and bugfixes for demos

Migration Guide
+++++++++++++++

Building
^^^^^^^^

*vortex* now includes its own *vcpkg* overlay for its dependencies in ``/.vcpkg``.
If you previously used this path for *vcpkg* (as suggested by earlier build guides), you will need to relocate it.

v0.4.1 -- 1/22/2022
-------------------

Changelog
+++++++++

- Add support for Alazar FFT acquisition with :class:`~vortex.acquire.AlazarFFTAcquisition` and :class:`~vortex.process.CopyProcessor`.
  See the :ref:`demo/alazar-fft` demo for an example.
- Move latency handling from formatter to engine.
- Add missing :func:`~vortex.scan.RasterScanConfig.to_segments` method for high-level scan classes.
- Add Python stubs for autocompletion and type checking.
- Add CMake presets.
- Add :class:`~vortex.format.PositionFormatExecutor`.
- Fix issues with :class:`~vortex.acquire.FileAcquisition`.
- Add ``sample`` to marker-associated callbacks.
- Assorted bug fixes and improvements.

Migration Guide
+++++++++++++++

Name Changes
^^^^^^^^^^^^

- ``Stack`` has replaced ``Cube`` in class names.  For example, ``CubeTensorEndpointX`` is now ``StackTensorEndpointX``.  Similarly, ``CubeFormatExecutor`` is now :class:`~vortex.format.StackFormatExecutor`.

IO Leading
^^^^^^^^^^

IO delays (e.g., for galvo response) are no longer handled in post-processing via the formatter. Instead, the engine now generates leading IO signals that cancel out the IO delay.
The delay in samples is passed via :meth:`~vortex.engine.EngineConfig.add_io` in the :class:`~vortex.engine.EngineConfig`.
Multiple IO delays are possible.
Change ``fc.stream_delay_samples = round(cfg.galvo_delay * ioc_out.samples_per_second)`` to ``ec.add_io(io_out, lead_samples=round(cfg.galvo_delay * ioc_out.samples_per_second))``.
