.. _module/acquire:

Acquire
#######

Acquisition components are sources of data in *vortex*, such as a digitizer card.
Each component provides a similar API.

-   :class:`~vortex.acquire.NullAcquisition`
-   :class:`~vortex.acquire.FileAcquisition`
-   :class:`~vortex.acquire.AlazarAcquisition`
-   :class:`~vortex.acquire.AlazarFFTAcquisition`
-   :class:`~vortex.acquire.AlazarGPUAcquisition`
-   :class:`~vortex.acquire.TeledyneAcquisition`
-   :class:`~vortex.acquire.ImaqAcquisition`

Overview
========

An acquisition component is first initialized with a call to :func:`~vortex.acquire.AlazarAcquisition.initialize()`, supplying a matching configuration object.
Initialization brings the component into a state where multiple acquisitions can be subsequently performed.
Once the component is initialized, a preparation step is required for some acquisitions.
Tasks performed during preparation are those that required to start each acquisition, such as arming the digitizer.
This is performed by calling :func:`~vortex.acquire.AlazarAcquisition.prepare()`.
After the preparation step, some acquisitions permit the queuing (or "preloading") of blocks for a buffered asynchronous operation.

The acquisition begins producing data with a call to :func:`~vortex.acquire.AlazarAcquisition.start()`.
Data is received synchronously with :func:`~vortex.acquire.AlazarAcquisition.next()` or asynchronously with :func:`~vortex.acquire.AlazarAcquisition.next_async()`.
Both methods accept an ``id`` argument, which is provided only for user bookkeeping and logging.
A call to :func:`~vortex.acquire.AlazarAcquisition.stop()` requests that the acquisition complete.
Note that asynchronously queued blocks may continue to complete after :func:`~vortex.acquire.AlazarAcquisition.stop()` is called.
When used with the :class:`~vortex.engine.Engine`, the user is only responsible for calling :func:`~vortex.acquire.AlazarAcquisition.initialize()`.
The engine will call :func:`~vortex.acquire.AlazarAcquisition.prepare()`, :func:`~vortex.acquire.AlazarAcquisition.start()`, and :func:`~vortex.acquire.AlazarAcquisition.stop()` at the appropriate time.

Some components support preloading, in which :func:`~vortex.acquire.AlazarAcquisition.next()` or :func:`~vortex.acquire.AlazarAcquisition.next_async()` may be called prior to :func:`~vortex.acquire.AlazarAcquisition.start()`.
These components queue the provided buffers for immediate acquisition once :func:`~vortex.acquire.AlazarAcquisition.start()` is called.
Each component states below whether or not preloading is supported.
Preloading is primarily a feature of the :class:`~vortex.engine.Engine`, which is controlled with the ``preload`` parameter on a per-component basis when calling :meth:`EngineConfig.add_acquisition() <vortex.engine.EngineConfig.add_acquisition>`.

.. Example
.. -------

.. This `example <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/acquire/alazar.py>`_ uses :class:`~vortex.acquire.AlazarAcquisition` to asynchronously acquire and then display a handful of buffers.

.. .. literalinclude:: /../demo/acquire/alazar.py
..     :language: python

Components
==========

*vortex* provides components for simulated and physical acquisitions.

Null
----

.. class:: vortex.acquire.NullAcquisition

    Perform no acquisition.

    This class is provided as an engine placeholder for testing or mocking.
    The only necessary configuration is the expected output shape.

    .. note::

        This component supports preloading with the engine.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        Present for API uniformity but calling is necessary only if the configuration is accessed elsewhere.

        :param NullAcquisitionConfig config:
            New configuration to apply.

    .. method:: prepare()

        Prepare to imminently start the acquisition.
        Present for API uniformity, but calling is unnecessary.

    .. method:: start()

        Start the acquisition.
        Present for API uniformity, but calling is unnecessary.

    .. method:: stop()

        Stop the acquisition.
        Present for API uniformity, but calling is unnecessary.

    .. method:: next(buffer, id=0)

        Acquire the next buffer.
        Always successfully acquires ``buffer.shape[0]`` records.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            The number of records is always matches ``buffer.shape[0]``.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the next buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback is executed in the calling thread before this method returns.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            The number of records is always ``buffer.shape[0]``, and the exception is always ``None``.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: NullAcquisitionConfig

        Copy of the active configuration.

.. class:: vortex.acquire.NullAcquisitionConfig

    Configuration object for :class:`~vortex.acquire.NullAcquisition`.

    .. property:: shape
        :type: List[int[3]]

        Required shape of output buffers.
        Returns a list of [:data:`records_per_block`, :data:`samples_per_record`, :data:`channels_per_sample`].
        Read-only.

    .. property:: samples_per_record
        :type: int

        Number of samples per record.

    .. property:: records_per_block
        :type: int

        Number of records in each acquired buffer or block.

    .. property:: channels_per_sample
        :type: int

        Number of channels that comprise each sample.

    .. method:: validate()

        Check the configuration for errors.

        :raises RuntimeError:
            If the configuration is invalid.

    .. method:: copy()

        Create a copy of this configuration.

        :return NullAcquisitionConfig:
            The copy.

File
----

.. class:: vortex.acquire.FileAcquisition

    Acquire data from a file.

    Data is read from the file and returned in the requested shape.
    The file is read as raw bytes with no datatype or alignment considerations.
    File looping for infinite acquisitions is possible.
    This class is intended primarily for testing or offline post-processing.

    .. note::

        This component supports preloading with the engine.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.

        :param FileAcquisitionConfig config:
            New configuration to apply.

    .. method:: prepare()

        Prepare to imminently start the acquisition.
        Present for API uniformity, but calling is unnecessary.

    .. method:: start()

        Start the acquisition, and open the source file.

    .. method:: stop()

        Stop the acquisition, and close the source file.

    .. method:: next(buffer, id=0)

        Acquire the buffer and return the number of acquired records.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            If the number acquired is less than the number requested, the acquisition is complete.

        :raises RuntimeError:
            If the acquisition fails.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to fill with data according to the shape of the buffer, which must match :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            If the number of records acquired is less than the number requested, the acquisition is complete.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: FileAcquisitionConfig

        Copy of the active configuration.

.. class:: vortex.acquire.FileAcquisitionConfig

    Base: :class:`~vortex.acquire.NullAcquisitionConfig`

    Configuration object for :class:`~vortex.acquire.FileAcquisition`.

    .. property:: path
        :type: str

        Path to file that backs the acquisition.

    .. property:: loop
        :type: bool

        Loop the file to provide an infinite acquisition.
        Otherwise, the acquisition ends when the end of file is reached.

    .. method:: copy()

        Return a copy of this configuration.

        :return FileAcquisitionConfig:
            The copy.

AlazarTech
----------

Host
^^^^

.. class:: vortex.acquire.AlazarAcquisition

    Acquire data using an AlazarTech digitizer.

    Once :func:`prepare()` is called, the acquisition may be started and stopped as many times as necessary.

    .. note::

        This component supports preloading with the engine.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        The Alazar card is fully configured when this method returns.

        :param AlazarConfig config:
            New configuration to apply.

    .. method:: prepare()

        Prepare to imminently start the acquisition.
        The Alazar card is armed for capture.

    .. method:: start()

        Start the acquisition.

    .. method:: stop()

        Stop the acquisition.

        .. caution::
            Asynchronously acquired buffers that completed before the acquisition was stopped may continue to result after this method returns.

    .. method:: next(buffer, id=0)

        Acquire the buffer and return the number of acquired records.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            If the number acquired is less than the number requested, the acquisition is complete.

        :raises RuntimeError:
            If the acquisition fails.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            If the number of records acquired is less than the number requested, the acquisition is complete.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: AlazarConfig

        Copy of the active configuration.

.. class:: vortex.acquire.AlazarConfig

    Base: :class:`~vortex.acquire.NullAcquisitionConfig`

    Configuration object for :class:`~vortex.acquire.AlazarAcquisition`.

    .. property:: device
        :type: AlazarDevice

        Alazar device for acquisition.
        Defaults to ``AlazarDevice()``.

    .. property:: clock
        :type: InternalClock | ExternalClock

        Internal or external clock configuration.
        Defaults to ``InternalClock()``.

    .. property:: trigger
        :type: SingleExternalTrigger | DualExternalTrigger

        Single or dual trigger configuration.
        Defaults to ``SingleExternalTrigger()``.

    .. property:: inputs
        :type: List[alazar.Input]

        List of input configurations for each channel to acquire.
        Default is empty.

    .. property:: options
        :type: List[AuxIOTriggerOut | AuxIOClockOut | AuxIOPacerOut | OCTIgnoreBadClock]

        List of acquisition options.
        Default is empty.

    .. property:: resampling
        :type: numpy.ndarray[int]

        Zero-based index of samples to keep in each record.
        All other samples are removed from the record.
        This can be used to perform resampling with nearest-neighbor interpolation.
        Number of samples to keep must match the number of samples per record.
        Set to an empty array (``[]``) to disable resampling.
        Disabled by default.

    .. property:: acquire_timeout
        :type: datetime.timedelta

        Timeout for the acquisition of each block.
        Defaults to ``timedelta(seconds=1)``.

    .. property:: stop_on_error
        :type: bool

        Automatically stop the acquisition when an error occurs.
        Default is ``True``.

    .. property:: channel_mask
        :type: int

        Bitmask of channels configured for acquisition.
        Read-only.

    .. property:: samples_per_second
        :type: int

        Number of samples per second for internally-clocked acquisitions.
        Read-only.

        :raises RuntimeError:
            if :data:`samples_per_second_is_known` is ``False``.

    .. property:: samples_per_second_is_known
        :type: bool

        ``True`` if :data:`samples_per_second` is specified in the configuration (e.g., :class:`vortex.alazar.InternalClock`) and ``False`` otherwise.
        Read-only.

    .. property:: recommended_minimum_records_per_block
        :type: int

        Minimum recommended records per block for the configured Alazar digitizer.
        Read-only.

    .. property:: bytes_per_multisample
        :type: int

    .. method:: copy()

        Create a copy of this configuration.

        :return AlazarConfig:
            The copy.

Host with On-board FFT
^^^^^^^^^^^^^^^^^^^^^^

.. class:: vortex.acquire.AlazarFFTAcquisition

    Base: :class:`~vortex.acquire.AlazarAcquisition`

    Acquire data using an AlazarTech digitizer with the on-board FPGA configured for FFT computation.

    This class may be used to simultaneously acquire the raw and FFT data using the :data:`~vortex.acquire.AlazarFFTConfig.include_time_domain` option.
    In this case, both the raw and FFT data are combined into a single record.

    .. note::

        This component supports preloading with the engine.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        The Alazar card is fully configured when this method returns.

        :param AlazarFFTConfig config:
            New configuration to apply.

    .. property:: config
        :type: AlazarFFTConfig

        Copy of the active configuration.

.. class:: vortex.acquire.AlazarFFTConfig

    Base: :class:`~vortex.acquire.AlazarConfig`

    Configuration object for :class:`~vortex.acquire.AlazarFFTAcquisition`.

    .. property:: fft_length
        :type: int

        Length of on-board FFT to perform.
        Records are zero-padded to reach this length.
        Must be larger than samples per record and must be a power of 2.

    .. property:: spectral_filter
        :type: numpy.ndarray[numpy.complex64]

        Spectral filter to apply before the FFT.
        Must have the same length as FFT.
        Set to empty array (``[]``) to disable.
        Disabled by default.

    .. property:: background
        :type: numpy.ndarray[numpy.uint16]

        Background record to subtract.
        Must have the same length as a record.
        Set to empty array (``[]``) to disable.
        Disabled by default.

    .. property:: include_time_domain
        :type: bool

        Append time domain data to the output FFT record.
        Requires a pointer cast to access since different data types are combined into single record.
        Defaults to ``False``.

    .. property:: samples_per_ascan
        :type: int

        Number of samples per the output A-scan, which may differ from samples per record depending on FFT settings.
        Read-only.

    .. property:: ascans_per_block
        :type: int

        Number of A-scans per block which is identical to number of records per blocks.
        Provided for API consistency only.

    .. property:: buffer_bytes_per_record
        :type: int

        The number of bytes required for each record buffer.
        Read-only.

    .. method:: copy()

        Create a copy of this configuration.

        :return AlazarFFTConfig:
            The copy.

GPU
^^^

.. class:: vortex.acquire.AlazarGPUAcquisition

    Base: :class:`~vortex.acquire.AlazarAcquisition`

    Acquire data using an AlazarTech digitizer with extensions to deliver data directly to a CUDA-capable GPU.

    .. note::

        This component supports preloading with the engine.

    .. note::

        *vortex* will perform its own GPU data transfers as needed.
        :class:`~vortex.acquire.AlazarAcquisition` with *vortex*-managed GPU transfer gives comparable performance to :class:`~vortex.acquire.AlazarGPUAcquisition`.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        The Alazar card is fully configured when this method returns.

        :param AlazarGPUConfig config:
            New configuration to apply.

    .. method:: next(buffer, id=0)

        Acquire the buffer and return the number of acquired records.

        :param cupy.ndarray[cupy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>` on GPU device :data:`self.config.gpu_device_index <vortex.acquire.AlazarGPUConfig.gpu_device_index>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            If the number acquired is less than the number requested, the acquisition is complete.

        :raises RuntimeError:
            If the acquisition fails.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param cupy.ndarray[cupy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>` on GPU device :data:`self.config.gpu_device_index <vortex.acquire.AlazarGPUConfig.gpu_device_index>`.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            If the number of records acquired is less than the number requested, the acquisition is complete.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: AlazarGPUConfig

        Copy of the active configuration.

.. class:: vortex.acquire.AlazarGPUConfig

    Base: :class:`~vortex.acquire.AlazarConfig`

    Configuration object for :class:`~vortex.acquire.AlazarGPUAcquisition`.

    .. property:: gpu_device_index
        :type: int

        Index of CUDA device for delivery of data.
        Defaults to index ``0``.

    .. method:: copy()

        Create a copy of this configuration.

        :return AlazarGPUConfig:
            The copy.

Configuration
^^^^^^^^^^^^^

An Alazar configuration is specified using a series of classes that encode configuration options.
These options are shared by all Alazar acquisition classes.

.. seealso::
    Consult the `ATS-SDK <https://docs.alazartech.com/ats-sdk-user-guide/latest/>`_ documentation for fine details regarding this configuration.
    Most configuration classes below map to one or two ATS-SDK API calls.

.. class:: vortex.acquire.AlazarDevice

    Representation of an Alazar device identifier.

    .. method:: __init__(system_index=1, board_index=1)

        Create a new object.

        :param int system_index:
            Value for :data:`system_index`.
        :param int board_index:
            Value for :data:`board_index`.


    .. property:: system_index
        :type: int

        Index of Alazar system.
        The first system has index ``1``.

    .. property:: board_index
        :type: int

        Index of board within Alazar system
        The first board has index ``1``.

.. class:: vortex.acquire.alazar.Coupling

    Enumeration of coupling modes.

    .. attribute:: DC
    .. attribute:: AC

.. class:: vortex.acquire.alazar.ClockEdge

    Enumeration of clock edges.

    .. attribute:: Rising
    .. attribute:: Falling

.. class:: vortex.acquire.alazar.TriggerSlope

    Enumeration of trigger slopes.

    .. attribute:: Positive
    .. attribute:: Negative

.. data:: vortex.acquire.alazar.InfiniteAcquisition
    :type: int

    Value to indicate infinite acquisition.

.. data:: vortex.acquire.alazar.TriggerRangeTTL
    :type: int

    Trigger range value that indicates TTL.

Clock
+++++

.. class:: vortex.alazar.InternalClock

    Configure an internal clock source.

    .. method:: __init__(samples_per_second=800_000_000)

        Create a new object.

        :param int samples_per_second:
            Value for :data:`samples_per_second`.

    .. property:: samples_per_second
        :type: int

        Number of samples per second to configure for the internal clock

.. class:: vortex.alazar.ExternalClock

    Configure an external clock source.

    .. method:: __init__(level_ratio=0.5, coupling=Coupling.AC, edge=ClockEdge.Rising, dual=False)

        Create a new object.

        :param float level_ratio:
            Value for :data:`level_ratio`.
        :param ~alazar.Coupling coupling:
            Value for :data:`coupling`.
        :param ~alazar.ClockEdge edge:
            Value for :data:`edge`.
        :param bool dual:
            Value for :data:`dual`.

    .. property:: level_ratio
        :type: float

        Signal level threshold in range ``[0, 1]`` at which a clock edge is detected.

    .. property:: coupling
        :type: ~alazar.Coupling

        Coupling for clock input.

    .. property:: edge
        :type: ~alazar.ClockEdge

        Edge on which to trigger the clock.

    .. property:: dual
        :type: bool

        Trigger on both rising and falling clock edges if ``True``.
        Otherwise, trigger only on edge set by :data:`edge`.

Trigger
+++++++

.. class:: vortex.alazar.SingleExternalTrigger

    Configure a single external trigger.

    .. method:: __init__(range_millivolts=2500, level_ratio=0.09, delay_samples=80, slope=TriggerSlope.Positive, coupling=DC)

        Create a new object.

        :param int range_millivolts:
            Value for :data:`range_millivolts`.
        :param float level_ratio:
            Value for :data:`level_ratio`.
        :param int delay_samples:
            Value for :data:`delay_samples`.
        :param ~alazar.TriggerSlope slope:
            Value for :data:`slope`.
        :param ~alazar.Coupling coupling:
            Value for :data:`coupling`.

    .. property:: range_millivolts
        :type: int

        Trigger input range, specified in millivolts.
        Value of ``0`` indicates TLL trigger input.

    .. property:: level_ratio
        :type: float

        Signal level threshold in range ``[0, 1]`` at which a trigger event is detected.

    .. property:: delay_samples
        :type: int

        Number of samples to skip following a trigger event before acquiring a record.
        Non-negative.

    .. property:: slope
        :type: ~alazar.TriggerSlope

        Polarity of trigger signal.

    .. property:: coupling
        :type: ~alazar.Coupling

        Coupling for trigger input.

.. class:: vortex.alazar.DualExternalTrigger

    Configure a dual external trigger.

    .. method:: __init__(range_millivolts=2500, level_ratios=[0.09, 0.09], delay_samples=80, initial_slope=Positive, coupling=DC)

        Create a new object.

        :param int range_millivolts:
            Value for :data:`range_millivolts`.
        :param List[float[2]] level_ratios:
            Value for :data:`level_ratios`.
        :param int delay_samples:
            Value for :data:`delay_samples`.
        :param ~alazar.TriggerSlope initial_slope:
            Value for :data:`initial_slope`.
        :param ~alazar.Coupling coupling:
            Value for :data:`coupling`.

    .. property:: range_millivolts
        :type: int

        Trigger input range, specified in millivolts.

    .. property:: level_ratios
        :type: List[float[2]]

        A pair of signal level thresholds in range ``[0, 1]`` at which trigger events are detected.

    .. property:: delay_samples
        :type: int

        Number of samples to skip following a trigger event before acquiring a record.
        Non-negative.

    .. property:: initial_slope
        :type: ~alazar.TriggerSlope

        Polarity of the initial trigger.

    .. property:: coupling
        :type: ~alazar.Coupling

        Coupling for trigger input.

Input
+++++

.. class:: vortex.alazar.Input

    Configure an input channel.

    .. method:: __init__(channel=Channel.B, range_millivolts=400, impedance_ohms=50, coupling=Coupling.DC)

        Create a new object.

        :param ~alazar.Channel channel:
            Value for :data:`channel`.
        :param int range_millivolts:
            Value for :data:`range_millivolts`.
        :param int impedance_ohms:
            Value for :data:`impedance_ohms`.
        :param ~alazar.Coupling coupling:
            Value for :data:`coupling`.

    .. property:: channel
        :type: ~alazar.Channel

        Channel to receive this configuration.

    .. property:: range_millivolts
        :type: int

        Channel input range, specified in millivolts.

    .. property:: impedance_ohms
        :type: int

        Channel input impedance, specified in Ohms.

    .. property:: coupling
        :type: ~alazar.Coupling

        Coupling for channel input.

    .. property:: bytes_per_sample
        :type: int

        Number of bytes per sample acquired for this input.
        Read-only.

    .. method:: copy()

        Copy this object.

        :return alazar.Input:
            The copy.

.. class:: vortex.acquire.alazar.Channel

    Enumeration of input channels.

    .. attribute:: A
    .. attribute:: B
    .. attribute:: C
    .. attribute:: D
    .. attribute:: E
    .. attribute:: F
    .. attribute:: G
    .. attribute:: H
    .. attribute:: I
    .. attribute:: J
    .. attribute:: K
    .. attribute:: L
    .. attribute:: M
    .. attribute:: N
    .. attribute:: O
    .. attribute:: P

Options
+++++++

Although the Alazar configuration will accept multiple auxiliary I/O options, only the last one will have effect.

.. class:: vortex.alazar.AuxIOTriggerOut

    Pass the trigger through the auxillary I/O port.

.. class:: vortex.alazar.AuxIOClockOut

    Pass the clock through the auxillary I/O port.

.. class:: vortex.alazar.AuxIOPacerOut

    Pass the clock through the auxillary I/O port after a divider.

    .. method:: __init__(divider=2)

        Create a new object.

        :param int divider:
            Value for :data:`divider`.

    .. property:: divider
        :type: int

        Divider for clock, with minimum value of ``2``.
        The clock frequency is divided by this value to determine the pacer frequency.

.. class:: vortex.alazar.OCTIgnoreBadClock

    Activate the OCT ignore bad clock feature.

    .. method:: __init__(good_seconds=4.95e-6, bad_seconds=4.95e-6)

        Create a new object

        :param float good_seconds:
            Value for :data:`good_seconds`.
        :param float bad_seconds:
            Value for :data:`bad_seconds`.

    .. seealso::
        See the `AlazarOCTIgnoreBadClock <https://docs.alazartech.com/ats-sdk-user-guide/latest/reference/AlazarOCTIgnoreBadClock.html>`_ documentation for full details.

    .. property:: good_seconds
        :type: float

        Good clock duration, specified in seconds.

    .. property:: bad_seconds
        :type: float

        Bad clock duration, specified in seconds.

Board
+++++

.. class:: vortex.alazar.Board

    Represents an Alazar digitizer board.

    .. method:: __init__(system_index=None, board_index=None)

        Create an object for the board identified by the system and board index.
        Pass no arguments to create an empty object.

        :param int system_index:
            System index.
        :param int board_index:
            Board index.

    .. property:: handle
        :type: int

        Handle to the underlying board for use with Alazar API.

    .. property:: valid
        :type: bool

        ``True`` is this object is non-empty and ``False`` otherwise.

    .. property:: info
        :type: ~alazar.Info

        Configuration, parameters, and data regarding the corresponding digitizer.

.. TODO: finish board documentation

Teledyne SP Devices
-------------------

ADQ
^^^

.. class:: vortex.acquire.TeledyneAcquisition

    Acquire data using a Teledyne SP Devices digitizer using the ADQAPI.

    Once :func:`initialize()` is called, the acquisition may be started and stopped as many times as necessary.

    .. note::

        This component does not support preloading with the engine.

    .. attention::

        This component uses zero-copy memory operations that require memory reserved for direct memory access (DMA).
        This memory is frequently in short supply, especially as memory fragmentation worsens with system uptime.
        Engine pipelines using this component may require smaller and/or fewer blocks.
        Restarting the computer to reduce memory fragmentation may improve performance.

    .. admonition:: Linux

        Linux by default limits DMA buffers to 4 MiB in size.
        When configuring the engine, block sizing parameters must satisfy ``ascans_per_block * samples_per_ascan * 2 < 4194304``.
        Otherwise, the Teledyne configuration will fail.
        In practice, since A-scan length is a hardware property, ``ascans_per_block`` is limited to a few hundred A-scans.
        Consider using hugepages with :attr:`~vortex.acquire.TeledyneConfig.enable_hugepages` to avoid this issue.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        The Teledyne card is fully configured when this method returns.

        :param TeledyneConfig config:
            New configuration to apply.

    .. method:: prepare()

        This method has no effect for Teledyne cards.

    .. method:: start()

        Start the acquisition.

    .. method:: stop()

        Stop the acquisition.

        .. caution::
            Asynchronously acquired buffers that completed before the acquisition was stopped may continue to result after this method returns.

    .. method:: next(buffer, id=0)

        Acquire the buffer and return the number of acquired records.

        .. attention::

            Although the buffer data type is 16-bit unsigned integer for compatibility, the data stored within the buffer is actually signed.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            If the number acquired is less than the number requested, the acquisition is complete.

        :raises RuntimeError:
            If the acquisition fails.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        .. attention::

            Although the buffer data type is 16-bit unsigned integer for compatibility, the data stored within buffers is actually signed.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            If the number of records acquired is less than the number requested, the acquisition is complete.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: TeledyneConfig

        Copy of the active configuration.

    .. property:: board_handle
        :type: int

        Handle to the underlying board for use with AQDAPI.


.. class:: vortex.acquire.TeledyneConfig

    Base: :class:`~vortex.acquire.NullAcquisitionConfig`

    Configuration object for :class:`~vortex.acquire.TeledyneAcquisition`.

    .. property:: device
        :type: int

        Index of the card to use.
        Default to ``0``.

    .. property:: clock
        :type: Clock

        Clock configuration.
        Defaults to ``Clock()``.

    .. property:: inputs
        :type: List[teledyne.Input]

        List of input configurations for each channel to acquire.
        Default is empty.

    .. property:: trigger_source
        :type: TriggerSource

        Source for the acquisition trigger.
        Defaults to ``PortTrig``.

    .. property:: trigger_skip_factor
        :type: int

        Downsample triggers by the given factor.
        A value of ``1`` skips no triggers.
        Defaults to ``1``.

    .. property:: trigger_offset_samples
        :type: int

        Number of samples to skip after a trigger before acquiring a record.
        Defaults to ``0``.

    .. property:: trigger_sync_passthrough
        :type: bool

        Emit a pulse on the sync port on each trigger.
        This is useful for synchronizing external hardware, such as I/O cards.
        Defaults to ``True``.

        .. important::

            Configure to ``False`` to use the test pattern generator.

    .. property:: sample_skip_factor
        :type: int

        Downsample samples within a record by the given factor.
        A value of ``1`` skips no samples.
        Defaults to ``1``.


    .. property:: enable_fwoct
        :type: bool

        Activate features that require FWOCT_ firmware when ``True`` and disable them when ``False``.
        If ``False`` and FWOCT_ is detected, FWOCT_ is configured in passthrough mode.
        Default is ``False``.

        .. NOTE::

            FWOCT_ version 8 or higher is required for full feature support.

    .. property:: resampling_factor
        :type: float

        Scale factor for k-clock resampling using the on-board FPGA.
        That is, a value of ``2`` will upsample each record by a factor of ``2``.
        Non-integer resamplings are allowed.
        Set to ``0`` to disable resampling.
        Defaults to ``0``.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: clock_delay_samples
        :type: float

        Number of samples, positive or negative, to delay the k-clock signal relative to the OCT signal.
        Non-integer numbers of samples are allowed.
        Set to ``0`` to disable.
        Defaults to ``0``.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: clock_edges
        :type: vortex.acquire.teledyne.ClockEdges

        Which k-clock edges to consider when resampling.
        Defaults to ``Rising``.

        .. tip::

            When :attr:`resampling_factor` of more than 2 is required, it is recommended to use ``Both`` for :attr:`clock_delay_samples` and reduce the resampling factor by a factor of 2.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: spectral_filter
        :type: numpy.ndarray[numpy.complex64]

        Spectral filter to apply before the FFT.
        Must have the same length as a record.
        Set to empty array (``[]``) to disable.
        Disabled by default.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: background
        :type: numpy.ndarray[numpy.uint16]

        Background record to subtract.
        Must have the same length as a record.
        Set to empty array (``[]``) to disable.
        Disabled by default.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: fft_mode
        :type: vortex.acquire.teledyne.FFTMode

        Output mode for the on-board FFT.
        Default is ``Disabled``.

        .. important::

            Using an OCT processor is still recommended when enabling the FFT.
            Use the :attr:`~vortex.process.CPUProcessorConfig.enable_ifft`, :attr:`~vortex.process.CPUProcessorConfig.enable_square`, :attr:`~vortex.process.CPUProcessorConfig.enable_log10`, and :attr:`~vortex.process.CPUProcessorConfig.enable_magnitude` to avoid repeat processing.

        .. note::

            This feature requires the FWOCT_ firmware.

    .. property:: enable_hugepages
        :type: bool

        On Linux, allocate transfer buggers using hugepages when enabled.
        This can greatly increase the available memory for buffers.
        Default is ``False``.

        .. attention::

            This option is experimental.
            Memory allocated through hugepages may not be freed when the Python interpreter exits.

        .. warning::

            On Windows, this option will raise an exception when enabled.

    .. property:: acquire_timeout
        :type: datetime.timedelta

        Timeout for the acquisition of each block.
        Defaults to ``timedelta(seconds=1)``.

    .. property:: stop_on_error
        :type: bool

        Automatically stop the acquisition when an error occurs.
        Default is ``True``.

    .. property:: periodic_trigger_frequency
        :type: float

        Trigger frequency for when ``Periodic`` is the trigger source.
        Defaults to ``10000``.

    .. property:: test_pattern_signal
        :type: bool

        Use the test pattern signal as the acquisition source.
        Defaults to ``False``.

.. _FWOCT: https://www.spdevices.com/what-we-do/products/firmware/fwoct

Configuration
^^^^^^^^^^^^^

.. seealso::
    Consult the `ADQAPI <https://www.spdevices.com/what-we-do/products/software>`_ documentation for fine details regarding this configuration.
    A large number of configuration options exists, not all of which *vortex* implements.

.. class:: vortex.acquire.teledyne.Clock

    Collection of clock settings.

    .. property:: sampling_frequency
        :type: int

        Sampling frequency for a record.
        Defaults to ``2_500_000_000``.

    .. property:: reference_frequency
        :type: int

        Sets the clock reference frequency.
        Defaults to ``10_000_000``.

    .. property:: clock_generator
        :type: ClockGenerator

        Source for clock.
        Defaults to ``InternalPLL``.

    .. property:: reference_source
        :type: ClockReferenceSource

        Source for clock reference.
        Defaults to ``Internal``.

    .. property:: delay_adjustment
        :type: int

        Sets the clock delay adjustment value.
        Enables clock delay adjustment if the value is non-zero.
        Defaults to ``0``.

    .. property:: low_jitter_mode_enabled
        :type: bool

        Enables low jitter mode when set to ``True``.
        Defaults to ``True``.

.. class:: vortex.acquire.teledyne.Input

    Representation of input channel.

    .. property:: channel
        :type: int

        Index of channel.
        Defaults to ``0``.

.. class:: vortex.acquire.teledyne.ClockGenerator

    Enumeration of clock generators.

    .. attribute:: InternalPLL
    .. attribute:: ExternalClock

.. class:: vortex.acquire.teledyne.ClockReferenceSource

    Enumerate of clock reference sources.

    .. attribute:: Internal
    .. attribute:: Port_CLK
    .. attribute:: PXIE_10M

.. class:: vortex.acquire.teledyne.TriggerSource

    Enumeration of trigger sources.

    .. attribute:: PortTrig
    .. attribute:: PortSync
    .. attribute:: PortGPIO
    .. attribute:: Periodic

.. class:: vortex.acquire.teledyne.ClockEdges

    Enumeration of clock edges.

    .. attribute:: Rising
    .. attribute:: Falling
    .. attribute:: Both

.. class:: vortex.acquire.teledyne.FFTMode

    Enumeration of FFT modes.

    .. attribute:: Disabled
    .. attribute:: Complex
    .. attribute:: Magnitude
    .. attribute:: LogMagnitude

NI Vision
---------

IMAQ
^^^^

.. class:: vortex.acquire.ImaqAcquisition

    Acquire data using a NI IMAQ-compatible line-scan or area-scan camera.
    Each line or row of the acquire corresponds to a record.
    Each element of the line or row corresponds to a sample.

    .. caution::

        :class:`~vortex.acquire.ImaqAcquisition` does not support preloading.
        When registering :class:`~vortex.acquire.ImaqAcquisition` with an :class:`~vortex.engine.Engine`, set ``preload=False``.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the acquisition using the supplied configuration.
        The Alazar card is fully configured when this method returns.

        :param ImaqAcquisitionConfig config:
            New configuration to apply.

    .. method:: prepare()

        Prepare to imminently start the acquisition.
        Present for API uniformity, but calling is unnecessary.

    .. method:: start()

        Start the acquisition.

    .. method:: stop()

        Stop the acquisition.

        .. caution::
            Asynchronously acquired buffers that completed before the acquisition was stopped may continue to result after this method returns.

    .. method:: next(buffer, id=0)

        Acquire the buffer and return the number of acquired records.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

        :returns int:
            The number of records acquired.
            If the number acquired is less than the number requested, the acquisition is complete.

        :raises RuntimeError:
            If the acquisition fails.

    .. method:: next_async(buffer, callback, id=0)

        Acquire the buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param numpy.ndarray[numpy.uint16] buffer:
            The buffer to acquire, with shape that matches :data:`self.config.shape <vortex.acquire.NullAcquisitionConfig.shape>`.
        :param Callable[[int, Exception], None] callback:
            Callback to execute when buffer is filled.
            The callback receives two arguments, the number of records acquired and any exception which occurred during the acquisition.
            If the number of records acquired is less than the number requested, the acquisition is complete.

        :param int id: Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: ImaqAcquisitionConfig

        Copy of the active configuration.

.. class:: vortex.acquire.ImaqAcquisitionConfig

    Base: :class:`~vortex.acquire.NullAcquisitionConfig`

    Configuration object for :class:`~vortex.acquire.ImaqAcquisition`.

    .. property:: device_name
        :type: str

        Name of IMAQ device to access.
        Defaults to ``"img0"``.

    .. property:: offset
        :type: List[int[2]]

        Region of interest offset within acquisition window in ``(X, Y)`` format.
        The shape of the region of interest is set with :data:`~NullAcquisitionConfig.records_per_block` and :data:`~NullAcquisitionConfig.samples_per_record`.
        Prefer to use :data:`sample_offset` and :data:`record_offset` instead.

    .. property:: sample_offset
        :type: int

        Number of samples to skip in the acquisition window.
        Defines the "left" offset of the region of interest.
        Defaults to ``0``.

    .. property:: record_offset
        :type: int

        Number of records to skip in the acquisition window.
        Defines the "top" offset of the region of interest.
        Defaults to ``0``.

    .. property:: line_trigger
        :type: Optional[LineTrigger]

        Configure a line input trigger.
        Set to ``None`` to disable.
        Disabled by default.

    .. property:: frame_trigger
        :type: Optional[FrameTrigger]

        Configure a frame input trigger.
        Set to ``None`` to disable.
        Disabled by default.

    .. property:: trigger_output
        :type: List[TriggerOutput]

        Configure multiple trigger outputs.
        Defaults to ``[TriggerOutput()]``.

    .. property:: ring_size
        :type: int

        Number of buffers to allocate for the internal ring buffer.
        Recommended minimum is ``10``.
        Adjust as needed to avoid frame overruns.
        Defaults to ``10``.

    .. property:: acquire_timeout
        :type: datetime.timedelta

        Timeout for the acquisition of a frame.
        Defaults to ``timedelta(seconds=1)``.

    .. property:: stop_on_error
        :type: bool

        Automatically stop the acquisition on an error.
        Deafults to ``True``.

    .. property:: bypass_region_check
        :type: bool

        Do not check that the region of interest fits within the acquisition window.
        If the region of interest does not fit but this check is disabled, the acquired frame will be zero-padded to the requested size.
        Enabled by default.

    .. method:: copy()

        Create a copy of this configuration.

        :return ImaqAcquisitionConfig:
            The copy.

.. TODO: documentation for Imaq object

Configuration
^^^^^^^^^^^^^

.. seealso::
    Consult the `IMAQ <https://documentation.help/NI-IMAQ/>`_ documentation for fine details regarding this configuration.
    Most configuration classes below map to one or two IMAQ API calls.

.. class:: vortex.acquire.imaq.Signal

    Enumeration of signal types.

    .. attribute:: NoSignal
    .. attribute:: External
    .. attribute:: RTSI
    .. attribute:: IsoIn
    .. attribute:: IsoOut
    .. attribute:: Status
    .. attribute:: ScaledEncoder
    .. attribute:: SoftwareTrigger

.. class:: vortex.acquire.imaq.Polarity

    Enumeration of polarities.

    .. attribute:: Low
    .. attribute:: High

.. class:: vortex.acquire.imaq.Source

    Enumeration of signal sources.

    .. attribute:: Disabled
    .. attribute:: AcquisitionInProgress
    .. attribute:: AcquisitionDone
    .. attribute:: PixelClock
    .. attribute:: Unasserted
    .. attribute:: Asserted
    .. attribute:: Hsync
    .. attribute:: Vsync
    .. attribute:: FrameStart
    .. attribute:: FrameDone
    .. attribute:: ScaledEncoder

.. class:: vortex.acquire.imaq.RegionOfInterest

    Representation of a rectangular region of interest within an image.

    .. method:: __init__(top, left, height, width, pixels_per_row=0)

        Create a new object.

        :param int top:
            Value for :data:`top`.
        :param int left:
            Value for :data:`left`.
        :param int height:
            Value for :data:`height`.
        :param int width:
            Value for :data:`width`.
        :param int pixels_per_row:
            Value for :data:`pixels_per_row`.

    .. property:: top
        :type: int
    .. property:: left
        :type: int
    .. property:: height
        :type: int
    .. property:: width
        :type: int

    .. property:: pixels_per_row
        :type: int

        A value of ``0`` indicates that there are :data:`width` pixels per row.

Trigger Input
+++++++++++++

.. class:: vortex.acquire.LineTrigger

    Configure a line trigger.

    .. method:: __init__(line=0, skip=0, polarity=Polarity.High, signal=Signal.External)

        Create a new object.

        :param int line:
            Value for :data:`line`.
        :param int skip:
            Value for :data:`skip`.
        :param ~imaq.Polarity polarity:
            Value for :data:`polarity`.
        :param ~imaq.Signal signal:
            Value for :data:`signal`.

    .. property:: line
        :type: int

        Index of input for the trigger.

    .. property:: skip
        :type: int

        Number of samples to skip after the line trigger is received.

    .. property:: polarity
        :type: ~imaq.Polarity

        Polarity of the trigger.

    .. property:: signal
        :type: ~imaq.Signal

        Signal type of the trigger.


.. class:: vortex.acquire.FrameTrigger

    Configure a frame trigger.

    .. method:: __init__(line=0, polarity=Polarity.High, signal=Signal.External)

        Create a new object.

        :param int line:
            Value for :data:`line`.
        :param ~imaq.Polarity polarity:
            Value for :data:`polarity`.
        :param ~imaq.Signal signal:
            Value for :data:`signal`.

    .. property:: line
        :type: int

        Index of input for the trigger.

    .. property:: polarity
        :type: vortex.acquire.imaq.Polarity

        Polarity of the trigger.

    .. property:: signal
        :type: vortex.acquire.imaq.Signal

        Signal type of the trigger.

Trigger Output
++++++++++++++

.. class:: vortex.acquire.TriggerOutput

    Configure a trigger output.

    .. method:: __init__(line=0, source=Source.Hsync, polarity=Polarity.High, signal=Signal.External)

        Create a new object.

        :param int line:
            Value for :data:`line`.
        :param ~imaq.Source source:
            Value for :data:`source`.
        :param ~imaq.Polarity polarity:
            Value for :data:`polarity`.
        :param ~imaq.Signal signal:
            Value for :data:`signal`.

    .. property:: line
        :type: int

        Index of the output for the trigger.

    .. property:: source
        :type: ~imaq.Source

        Source of the trigger.

    .. property:: polarity
        :type: ~imaq.Polarity

        Polarity of the trigger.

    .. property:: signal
        :type: ~imaq.Signal

        Signal type of the trigger.
