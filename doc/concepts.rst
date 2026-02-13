.. _concepts:

Concepts
========

.. warning::

    This document is under construction.

Pipeline
--------

A **pipeline** is five stages through which data flows during a *vortex* acquisition or session.
The *vortex* pipeline is sufficiently flexible to accommodate most OCT use cases.
If a specific use case does not fit this format, the user can assemble their own pipeline from individual *vortex* components, usually at the C++ level to meet real-time requirements.

.. tikz::
    :alt: Diagram of vortex's pipeline

    \tikzset{
        base box/.style={draw, thick, rounded corners = 2 mm, align=center, text width = 2.75 cm, inner sep=0, outer sep=0},
        box/.style={base box, minimum height = 1.5 cm},
        arrow/.style={->, ultra thick, >=stealth'}
    }

    \node[box] (generate) {Generate};
    \node[box, right=1cm of generate] (acquire) {Acquire};
    \node[box, right=1cm of acquire] (process) {Process};
    \node[box, right=1cm of process] (format) {Format};
    \node[box, right=1cm of format] (store) {Store};

    \draw[arrow] (generate) -- (acquire);
    \draw[arrow] (acquire) -- (process);
    \draw[arrow] (process) -- (format);
    \draw[arrow] (format) -- (store);

-   **Generate**

    Calculate scan waveforms and other signals from the scan pattern.
    This stage corresponds to components in the :ref:`module/scan` module.

-   **Acquire**

    Collect data from digitizers or frame grabbers.
    Read/write synchronized I/O signals.
    This stage corresponds to components in the :ref:`module/acquire` and :ref:`IO <module/io>` modules.

-   **Process**

    Perform OCT processing, including de-averaging, resampling, complex filtering, FFT, and log normalization.
    Perform non-OCT processing using linear transformations.
    This stage corresponds to components in the :ref:`module/process` module.

-   **Format**

    Organize data into useful formats with inactive segment removal, assembly of volumes, or rectification of non-rectangular patterns.
    This stage corresponds to components in the :ref:`Format <module/format>` module.

-   **Store**

    Maintain data on host or GPU memory.
    Deliver data to disk for persistent storage.
    This stage corresponds to components in the :ref:`Endpoint <module/endpoint>` and :ref:`Storage <module/storage>` modules.

System
------

A **system** is a *vortex* pipeline that has been configured to accomplish a specific imaging task.
Each pipeline stage is populated with components related to that imaging task.
The whole system is managed by the :class:`~vortex.engine.Engine`, which coordinates flow of data through the pipeline in real-time.
The :class:`~vortex.engine.EngineConfig` encodes the pipeline configuration as well as the data management and flow control required for the system.

Application
-----------

An **application** is a system that is paired with user-facing features, such as graphical display or event-driven changes in scan patterns.
An application interacts with the engine through the :class:`~vortex.engine.ScanQueue`, which receives scan patterns to execute, and endpoints, which asynchronously issue callbacks.
Using callbacks, the application can chain custom processing onto that already performed in the pipeline.

.. tikz::
    :alt: Diagram of how vortex's pipeline interacts with an end-application

    \tikzset{
        base box/.style={draw, thick, rounded corners = 2 mm, align=center, text width = 2.75 cm, inner sep=0},
        box/.style={base box, minimum height = 1.5 cm},
        arrow/.style={->, ultra thick, >=stealth'}
    }

    \node[box] (generate) {Generate};
    \node[box, right=1cm of generate] (acquire) {Acquire};
    \node[box, right=1cm of acquire] (process) {Process};
    \node[box, right=1cm of process] (format) {Format};
    \node[box, right=1cm of format] (store) {Store};

    \draw[arrow] (generate) -- (acquire);
    \draw[arrow] (acquire) -- (process);
    \draw[arrow] (process) -- (format);
    \draw[arrow] (format) -- (store);

    \node[box, above=1cm of generate] (scan-queue) {Scan\\Queue};
    \node[box, above=1.5cm of scan-queue, yshift=3mm, xshift=3mm] (scan-patterns-top) {};
    \node[box, fill=backcolor, above=1.5cm of scan-queue, yshift=1.5mm, xshift=1.5mm] {};
    \node[box, fill=backcolor, above=1.5cm of scan-queue] (scan-patterns) {Scan\\Patterns};

    \draw[arrow] (scan-patterns) -- (scan-queue);
    \draw[arrow] (scan-queue) -- (generate);

    \node[box, above=1cm of store] (endpoint2) {Endpoint};
    \node[box, left=0.5cm of endpoint2] (endpoint1) {Endpoint};
    \node[box, right=0.5cm of endpoint2] (endpoint3) {Endpoint};

    \draw[arrow] (store) -- (endpoint1);
    \draw[arrow] (store) -- (endpoint2);
    \draw[arrow] (store) -- (endpoint3);

    \coordinate (midpt) at ($(scan-queue)!0.5!(scan-patterns)$);
    \draw[thick, dashed] (midpt -| scan-queue.west) -- (midpt -| endpoint3.east)
        coordinate[pos=0.35] (apppt);

    \node[above] at (apppt) {Application (Python or C+)};
    \node[below] at (apppt) {Vortex Engine (C++)};

    \node[box, above=3cm of apppt] (logic) {Application\\Logic};

    \node[box] at (scan-patterns -| endpoint2) (custom) {Custom\\Processing};

    \draw[arrow] (endpoint2) -- (custom);
    \draw[arrow] (logic) -| (scan-patterns-top.north -| scan-patterns.north);
    \draw[arrow] (custom) |- (logic) node[pos=0.85, above] {Callbacks};
    \draw[ultra thick] (endpoint1) -- (endpoint1 |- logic);
