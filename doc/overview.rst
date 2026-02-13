Overview
========

*vortex* provides several categories of objects at the **driver**, **component**, and **system** levels.
This document provides an overview of these objects and how they interact.

.. tikz:: Structural organization of objects within *vortex* from high-level (top) to low-level (bottom).
   :alt: Diagram of vortex's components

    % ref: https://tex.stackexchange.com/questions/464213/rotate-node-and-use-relative-positioning-when-using-two-lines
    \tikzset{
      base box/.style={draw, thick, rounded corners = 2 mm, align=center, text width = 2.7 cm, inner sep=0, outer sep=0},
      box/.style={base box, minimum height = 1.5 cm},
      rotated box/.style={base box, minimum width = 3 cm, minimum height = 1.3 cm, rotate=90, anchor=north},
      double box/.style={base box, minimum width = 2.7 cm, minimum height = 3 cm},
      triple box/.style={base box, minimum width = 4.1 cm, minimum height = 3 cm}
    }

    \node[rotated box]                                (reflexxes) {Reflexxes};
    \node[right=1 mm of reflexxes.south, rotated box] (alazar) {ATS-API};
    \node[right=1 mm of alazar.south, rotated box]    (teledyne) {ADQAPI};
    \node[right=1 mm of teledyne.south, rotated box]  (imaq) {IMAQ};
    \node[right=1 mm of imaq.south, rotated box]      (daqmx) {DAQmx};
    \node[right=1 mm of daqmx.south, rotated box]     (galvo) {ATS-API};
    \node[right=1 mm of galvo.south, rotated box]     (cuda) {CUDA};
    \node[right=1 mm of cuda.south, rotated box]      (fftw) {FFTW};

    \node[right=1 mm of fftw.south, rotated box]      (sa) {Spectra, A-scan};
    \node[right=3 mm of sa.south, rotated box]        (srp) {Stack, Radial, Position};
    \node[right=1 mm of srp.south, rotated box]       (ct) {CPU, GPU};
    \node[right=1 mm of ct.south, rotated box]        (numpy) {NumPy, MATLAB};
    \node[right=1 mm of numpy.south, rotated box]     (broct) {BROCT};
    \node[right=1 mm of broct.south, rotated box]     (nrrd) {NRRD,\\NIfTI};

    \node[above=3.3 cm of reflexxes.north, rotated box] (scan) {Scan Pattern};
    \node[right=1 mm of scan.south, triple box] (acquire) {Acquisition};
    \node[right=1 mm of acquire.east, double box] (io) {I/O};
    \node[right=1 mm of io.east, double box] (process) {Processor};
    \node[right=1 mm of process.east, rotated box] (format) {Formatter};

    \node[right=3 mm of format.south, double box] (executor) {Executor};
    \node[right=1 mm of executor.east, triple box] (storage) {Storage};

    \coordinate (m) at ($(scan.north east)!0.5!(format.south east)$);
    \node[box, above=3 mm of m, minimum width=12.5 cm]   (engine) {Engine};
    \node[box, right=3 mm of engine, minimum width=6.9 cm]   (endpoints) {Endpoints};

    \node[above, rotate=90] at (reflexxes.north) {Driver\strut};
    \node[above, rotate=90] at (scan.north) {Component\strut};
    \node[above, rotate=90] at (engine.west) {System\strut};

    \coordinate (m) at ($(engine.north east)!0.5!(endpoints.north west)$);
    \draw[thick, dashed] (m) -- (m |- fftw.west);

Drivers
-------

**Drivers** provide convenient wrappers around low-level APIs.
Examples include drivers for NI DAQmx, Alazar ATS-SDK, and Reflexxes.
Applications that use *vortex* do not routinely interact with drivers except for use of configuration constants or introspection.
For example, an application may query available Alazar cards via the Alazar driver.

.. admonition:: C++

   C++ users may provide their own drivers for custom components, such as a new acquisition card.

Components
----------

**Components** encapsulate functionality that meets the requirements of specific roles within *vortex*.
Examples include :class:`~vortex.acquire.AlazarAcquisition`, which encapsulates an acquisition from an Alazar card, or :class:`~vortex.process.CUDAProcessor`, which encapsulates CUDA-based OCT processing.
Each component is paired with a configuration object which contains all information necessary for a component to operate.
For example, :class:`~vortex.process.CUDAProcessorConfig` indicates the A-scan shape and spectral filter for use with :class:`~vortex.process.CUDAProcessor`.

Most users will interact with *vortex* at the component level.
A typical application will create the components it requires and then assemble them into a **system** using the :class:`~vortex.engine.Engine`, as described further below.
However, certain advanced or specific applications may wish to manage the system themselves and create only components.
This is an intended and supported use case for *vortex* and is facilitated by the strict separation of components and systems.

.. admonition:: C++

   C++ users may provide their own components, optionally extending existing ones.

.. admonition:: Python

   Python users are limited to the standard *vortex* components, unless they build *vortex* with bindings for custom C++ components.

System
------

A **system** is a collection of components organized to accomplish a task.
For *vortex*, a system is built from an **engine** and group of **endpoints**.
The :class:`~vortex.engine.Engine` implements a flexible pipeline that organizes components into an application and manages data transfer between components.
Components within a particular stage of the pipeline meet specific requirements and interface with the engine via an adapter.
Endpoints determine the disposition of the data once it has transited the engine's pipeline.
Examples include storing data to disk (:class:`~vortex.engine.AscanStreamEndpoint`) or assembling a volume in GPU memory for further processing (:class:`~vortex.engine.StackDeviceTensorEndpointInt8`).

A typical *vortex* use-case is to configure and start the engine such that *vortex* itself manages the majority of the data processing.
The engine and endpoints provide callbacks that allow the user to respond to events in real time, such as by appending user-defined processing.
This arrangement enables the user to focus on the specific aspects of their application rather than the internals of the processing pipeline.

.. admonition:: C++

   C++ users may wish to implement a custom pipeline rather than using *vortex*'s engine in order to meet specific application needs.

.. admonition:: Python

   Python users interact primarily with *vortex* at the engine level by configuring a pipeline.
