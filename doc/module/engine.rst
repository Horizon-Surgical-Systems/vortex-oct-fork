Engine
======

.. warning::

    This document is under construction.

The :class:`~vortex.engine.Engine` implements a high-performance data processing pipeline composed of acquisition, processing, and formatting stages.
See :ref:`concepts` for more detail regarding this pipeline and its capabilities.

.. class:: vortex.engine.Engine

    A real-time engine that moves data through an acquisition, processing, and formatting pipeline.

    Once the pipeline is configured, the engine can operate the pipeline in multiple sessions, bounded by :meth:`start` and :meth:`stop` calls.
    The session is active until the :attr:`scan_queue` becomes empty, :attr:`EngineConfig.blocks_to_acquire <vortex.engine.EngineConfig.blocks_to_acquire>` have been acquired, :meth:`stop` is called, or an error occurs.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the engine using the supplied configuration.

        :param EngineConfig config:
            New configuration to apply.

    .. method:: prepare()

        Prepare to start the engine.
        The engine allocates all blocks, assigns and allocates transfer buffers, and notifies endpoints to allocate their internal buffers, if needed.

    .. method:: start()

        Start a new session with the engine.
        Acquisition and IO components are preloaded, if applicable, and started.
        Every call of this method must be paired with a call to :meth:`stop` at a later time.

    .. method:: wait()

        Wait for the active session to complete.
        This method will block indefinitely.

        :raises RuntimeError:
            If any error occurred to shut down the session prematurely.
            This is guaranteed to be the first of such errors in case a cascade of errors occurs.

    .. method:: wait_for(timeout)

        Wait the specified duration for the session to complete.

        :param float timeout:
            Maximum duration to wait in seconds.

        :returns bool:
            ``True`` if engine has finished and ``False`` otherwise.

        :raises RuntimeError:
            If any error occurred to shut down the session prematurely.
            This is guaranteed to be the first of such errors in case a cascade of errors occurs.

    .. method:: shutdown(interrupt=False)

        Request that the engine shut down without waiting for it do so.
        A call to :meth:`stop` is still required before then engine can be started again.

        :param bool interrupt:
            If `True`, abort any dispatched blocks.
            If `False`, wait for dispatched blocks to complete before exiting.

    .. method:: stop()

        Request that the engine shut down and wait for it to do so.
        This method must be called for every call to :meth:`start`.

        :raises RuntimeError:
            If any error occurred to shut down the engine prematurely.
            This is guaranteed to be the first of such errors in case a cascade of errors occurs.

    .. method:: status

        Query the engine status.

        :return EngineStatus:
            The current status of the engine.

    .. property:: config
        :type: EngineConfig

        Copy of the active configuration.

    .. property:: scan_queue
        :type: ScanQueue

        Access the engine's scan queue.

    .. property:: done
        :type: bool

        Return ``True`` if the engine is stopped and ``False`` otherwise.

        .. warning::

            This property will return ``True`` until :meth:`stop` is called.
            Do no rely on this property to determine if the engine has exited.

    .. property:: event_callback
        :type: Callable[[Engine.Event, Exception], None]

        Callback to receive status events from the engine.

    .. property:: job_callback
        :type: Callable[[int, EngineStatus, JobTiming], None]

        Callback to receive status and timing information for each block when it exits the pipeline.

        .. caution::

            Avoid computationally expensive tasks in this callback or the session may shut down prematurely due to delayed block recycling.

.. class:: vortex.engine.EngineConfig

    Configuration object for :class:`~vortex.engine.Engine`.

    .. property:: records_per_block
        :type: int

        Number of records (spectra or A-scans) in a block.
        Each block represents a slice of time, as determined by the number of records acquired per second.
        Default is ``1000``.

    .. property:: blocks_to_allocate
        :type: int

        Number of blocks to pre-allocate prior to starting the session.
        This determines the maximum session duration or size that can be buffered in memory.
        The maximum buffered duration in records is the product of :attr:`blocks_to_allocate` and :attr:`records_per_block`.
        Default is ``4``.

    .. property:: preload_count
        :type: int

        Number of blocks to commit to the hardware drivers prior to starting the session.
        This determines how far in advance the engine is generating signals and scheduling buffers for acquisition.
        Once the engine starts, the engine will never have more that this number of blocks pending for acquisition.
        The product of :attr:`preload_count` and :attr:`records_per_block` determines the number of records required for new inputs (e.g., scan pattern changes) to propagate through the pipeline.
        Default is ``2``.

    .. property:: blocks_to_acquire
        :type:

        The total acquisition duration as measured in blocks.
        Set to ``0`` for an indefinite acquisition, which ends only when the last scan reports that it is complete.
        Default is ``0``.

    .. property:: post_scan_records
        :type: int

        The number of records to acquire after the last scan reports that is is complete.
        This is useful if hardware latencies cause the final samples of the scan to physically occur after the acquisition would have otherwise ended.
        Default is ``0``.

    .. property:: scanner_warp
        :type: [~vortex.scan.warp.NoWarp | ~vortex.scan.warp.AngularWarp | ~vortex.scan.warp.TelecentricWarp]

        Scan warp to generate the sample waveforms from the galvo waveforms.
        Default is :class:`~vortex.scan.warp.NoWarp`.

    .. property:: galvo_output_channels
        :type: int

        Number of output channels to allocate for galvo waveforms.
        Default is ``2``.

    .. property:: galvo_input_channels
        :type: int

        Number of input channels to allocate for galvo waveforms.
        Default is ``2``.

    .. property:: strobes
        :type: List[~vortex.engine.SampleStrobe | ~vortex.engine.SegmentStrobe | ~vortex.engine.VolumeStrobe | ~vortex.engine.ScanStrobe | ~vortex.engine.EventStrobe]

        Scan pattern-derived strobes to generate for output.
        Default is ``[SampleStrobe(0, 2), SampleStrobe(1, 1000), SampleStrobe(2, 1000, Polarity.Low), SegmentStrobe(3), VolumeStrobe(4)]``.

    .. property:: lead_marker
        :type: ~vortex.marker.ScanBoundary

        Scan boundary marker inserted for leading samples.
        The primary purpose of this option is to apply :class:`~vortex.marker.Flags` to the leading samples for stream-based storage endpoints, such as :class:`~vortex.engine.AscanStreamEndpoint`.
        Default is `ScanBoundary(0, 0, 0, 0x00)`.

    .. property:: lead_strobes
        :type: int

        Value for the strobe output when generating leading samples.
        Default is ``0``.

    .. method:: add_acquisition(acquisition, processors, preload=True, master=True)

        Register the acquisition with the engine and route its output data to the listed processors.
        :func:`divide` and :func:`cycle` may be used to build arbitrarily nested schemes for distributing data between processors.
        For example, :func:`cycle` can be used to realize "ping-pong" GPU processing of data, such as below.

        .. code-block:: python

            engine.add_acquisition(acquisition, [
                cycle([
                    processorA,
                    processorB
                ])
            ])

        There is a single acquisition dispatch thread for the whole engine.
        Processing is dispatched sequentially to divided processing in the listed order but may complete out of order.

        :param Acquisition acquisition:
            The acquisition component to register with the engine.
            See the :ref:`list of acquisition components <module/acquire>` for supported object types.
        :param List[Processor | Cycle | Divide] processors:
            The graph of processing components to receive the data from this acquisition component.
            See the :ref:`list of processing components <module/process>` for supported object types.
        :param bool preload:
            Enable or disable preloading with this acquisition.
            If ``True``, :attr:`preload_count` blocks will be queued before the engine starts.
            If ``False``, the first block will be queued immediately after the engine starts.
        :param bool master:
            Control when this acquisition is started in relation to other acquisition or IO components.
            All components with ``master=True`` are started after all components with ``master=False``.

    .. method:: add_processor(processor, formatters)

        Register the processor with the engine and route its output to all listed formatters.
        The engine ensures that data arrives on the correct GPU device, if any, and deinterleaves multiple channels, if present, in preparation for processing.
        There is a single processing dispatch thread for the whole engine.
        Formatters are scheduled receive data in the order listed, although out-of-order completion may causes formatters to execute in different orders.

        :param Processor processor:
            The processor component to register with the engine.
            See the :ref:`list of processors <module/process>` for supported object types.
        :param List[Formatter] formatters:
            The list of formatting components to receive the data from this processor component and its parent acquisition component.

    .. method:: add_formatter(formatter, endpoints)

        Register the formatter component with the engine and apply its format plans to acquired and/or processed via the listed endpoints.
        Each formatter is allocated a dedicated thread for its endpoints.
        Endpoints receive data sequentially in the order listed.

        :param Formatter formatter:
            The formater component to register with the engine.
            See the :ref:`list of processors <module/process>` for supported object types.
        :param List[Endpoints] endpoints:
            The list of endpoints to receive this formatter's plan and the data associated with the parent acquisition and processor components.

    .. method:: add_io(io, preload=True, master=False, lead_samples=0)

        Register the IO component with the engine.

        :param IO io:
            The IO component to register with the engine.
        :param bool preload:
            Enable or disable preloading for this IO component.
            See :meth:`add_acquisition` for full explanation.
        :param bool master:
            Enable or disable master status for this IO component.
            See :meth:`add_acquisition` for full explanation.
        :param int lead_samples:
            The number of samples in advance to generate output waveforms in order to cancel out IO delay.
            This is done by looking ahead into the scan pattern-derived waveforms by the specified number of samples.
            Default is ``0``.

    .. method:: validate()

        Check the configuration for errors.

        :raises RuntimeError:
            If the configuration is invalid.

        .. warning::

            This method is not fully implemented yet.

.. function:: vortex.engine.divide(nodes)

    Return an object that informs the engine to divide the acquired data between multiple processors that execute in parallel.
    In Python, this is equivalent to creating a list of nodes so this function effectively returns ``nodes``.

    :param List[Processor | Cycle | Divide] nodes:
        The processor components to divide.

.. function:: vortex.engine.cycle(nodes)

    Return an object that informs the engine to rotate between the processors in order for each block of acquired data.

    :param List[Processor | Cycle | Divide] nodes:
        The processor components to cycle.

.. class:: vortex.engine.ScanQueue

    A queue of scans to execute with the engine.
    When the scan queue is empty, the engine initiates a graceful shutdown.

    .. method:: append(scan, callback=None, marker=ScanBoundary())

        Append the specified scan to the queue.

        :param Scan scan:
            The scan to append to the queue.
            See the :ref:`list of scans <module/scan>` for supported object types.
        :param Callable[[int, ScanQueue.Event], None] callback:
            Callback to receive notifications regarding this scan.
        :param ~vortex.marker.ScanBoundary marker:
            Boundary marker for the start of this scan.

        .. caution::

            This callback is executed in the acquisition dispatch thread of the engine, which is highly sensitive to timing delays.
            Avoid computationally expensive tasks in this callback or the session may shut down prematurely due to buffer underflows or overflows.

    .. method:: interrupt(scan, callback=None, marker=ScanBoundary())

        Clear the scan queue and immediately switch to the specified scan.
        The latency with which the transition propagates through the engine is determined by the engine's :attr:`~EngineConfig.preload_count`.

        :param Scan scan:
            The scan to append to the queue.
            See the :ref:`list of scans <module/scan>` for supported object types.
        :param Callable[[int, ScanQueue.Event], None] callback:
            Callback to receive notifications regarding this scan.
        :param ~vortex.marker.ScanBoundary marker:
            Boundary marker for the start of this scan.

        .. caution::

            This callback is executed in the acquisition dispatch thread of the engine, which is highly sensitive to timing delays.
            Avoid computationally expensive tasks in this callback or the engine may shut down prematurely due to buffer underflows or overflows.

    .. method:: reset()

        Clears the scan queue and internal scan state (i.e., last position and velocity).

    .. method:: clear()

        Clears the scan queue but maintains internal scan state (e.g., last position and velocity).

    .. property:: empty_callback
        :type: Callable[[~ScanQueue.OnlineScanQueue], None]

        Callback to execute when the scan queue is empty but before the engine initiates a graceful shutdown.
        This represents the last opportunity to prolong the session.
        The callback receives a single argument of :class:`~vortex.engine.ScanQueue.OnlineScanQueue` which exposes a single :meth:`OnlineScanQueue.append() <vortex.engine.ScanQueue.OnlineScanQueue.append>` method to provide a thread-safe mechanism to append another scan.
        This method is identical in operation to :meth:`ScanQueue.append() <vortex.engine.ScanQueue.append>`.
        If no scan is appended, the engine will initiate a graceful shutdown immediately after the callback returns.

        .. attention::

            Any attempt to call a method of the :class:`ScanQueue` during this callback will lead to deadlock.
            Only call methods of the :class:`~ScanQueue.OnlineScanQueue` provided as the callback's argument.
