.. _module/process:

Process
#######

Processing components perform transformations of data in *vortex*, such as computing the FFT.
Each component provides a similar API.

-   :class:`~vortex.process.NullProcessor`
-   :class:`~vortex.process.CopyProcessor`
-   :class:`~vortex.process.CPUProcessor`
-   :class:`~vortex.process.CUDAProcessor`

Overview
========

A processing component is first initialized with a call to :func:`~vortex.process.CUDAProcessor.initialize()`, supplying a matching configuration object.
Initialization brings the component into a state where processing can begin and frequently involves the allocation of memory buffers.
An unlimited amount of data can then be transformed through the processor.
Processors allow changes to their configuration using the :func:`~vortex.process.CUDAProcessor.change()` method which accepts the new configuration.
Not all aspects of the configuration can be changed, however.
Take care when changes of configuration require reallocation of internal buffers.

Data is transformed synchronously with :func:`~vortex.process.CUDAProcessor.next()` or asynchronously with :func:`~vortex.process.CUDAProcessor.next_async()`.
Both methods accept an ``id`` argument, which is provided only for user bookkeeping and logging.
In the documentation below, the input is referred to as :math:`x[i,j,k]` (records) and the output is referred to as :math:`y[i,j,k]` (A-scans).
The indices :math:`i`, :math:`j`, and :math:`k` correspond to the record/A-scan, sample, and channel dimensions, respectively.
All buffers must have a single channel per sample (i.e., :math:`k \in \{ 1 \}`).

Components
==========

Null
----

.. class:: vortex.process.NullProcessor

    Perform no processing.

    This class is provided as an engine placeholder for testing or mocking.
    No operation is performed on the input or output buffers.

    .. property:: config
        :type: NullProcessorConfig

        Copy of the active configuration.

.. class:: vortex.process.NullProcessorConfig

    Configuration object for :class:`~vortex.process.NullProcessor`.

    .. property:: input_shape
        :type: List[int[3]]

        Required shape of input buffers.
        Returns list of [ :data:`records_per_block`, :data:`samples_per_record`, ``1`` ].
        Read-only.

    .. property:: samples_per_record
        :type: int

        Number of samples per record.

    .. property:: records_per_block
        :type: int

        Number of records in each input buffer or block.
        Identical to :data:`ascans_per_block`.

    .. property:: output_shape
        :type: List[int[3]]

        Required shape of output buffers.
        Returns list of [ :data:`ascans_per_block`, :data:`samples_per_ascan`, ``1``].
        Read-only.

    .. property:: samples_per_ascan
        :type: int

        Number of samples per A-scan.
        Returns either :data:`samples_per_record` or the length of :data:`~vortex.process.CUDAProcessorConfig.resampling_samples`, if the latter is available.
        Read-only.

    .. property:: ascans_per_block
        :type: int

        Number of A-scans in each acquired buffer or block.
        Identical to :data:`records_per_block`.

    .. method:: validate()

        Check the configuration for errors.

        :raises RuntimeError:
            If the configuration is invalid.

    .. method:: copy()

        Create a copy of this configuration.

        :return NullProcessorConfig:
            The copy.

Copy
----

.. class:: vortex.process.CopyProcessor

    Copy data from input to output buffers with optional slicing (:math:`s[n]`) and linear transformation (:math:`a` and :math:`b`).

    .. math::

        y[i,j,k] = a + b x[i,s[j],k]

    OCT processing is not performed.
    This processor is intended primarily for acquisitions that provide processed OCT data, such as :class:`~vortex.acquire.AlazarFFTAcquisition`.
    Computation is performed on the CPU.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the processor using the supplied configuration.
        All necessary internal buffers are allocated when this method returns.

        :param CopyProcessorConfig config:
            New configuration to apply.

    .. method:: change(config)

        Change the processor configuration.
        All configuration options may be changed, but changes to :data:`CopyProcessorConfig.slots` will not have an effect.

        .. danger::
            It is not safe to call this method while a block is currently processing.

        :param CopyProcessorConfig config:
            New configuration to apply.

    .. method:: next(input_buffer, output_buffer, id=0, append_history=True)

        Process the next buffer.

        :param numpy.ndarray[numpy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param numpy.ndarray[numpy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. method:: next_async(input_buffer, output_buffer, callback, id=0, append_history=True)

        Process the next buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param numpy.ndarray[numpy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param numpy.ndarray[numpy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param Callable[[Exception], None] callback:
            Callback to execute when buffer is processed.
            The callback receives as an argument any exception which occurred during processing.
        :param int id:
            Number to associate with the buffer for logging purposes.

    .. property:: config
        :type: CopyProcessorConfig

        Copy of the active configuration.

.. class:: vortex.process.CopyProcessorConfig

    Base: :class:`~vortex.process.NullProcessorConfig`

    Configuration object for :class:`~vortex.process.CopyProcessor`.

    .. property:: sample_slice
        :type: [~vortex.format.NullSlice | ~vortex.format.SimpleSlice]

        Optional slicing operation :math:`s[n]` to apply along each record/A-scan.
        Defaults to ``NullSlice()``.

    .. property:: sample_transform
        :type: [~vortex.format.NullTransform | ~vortex.format.LinearTransform ]

        Optional linear transformation :math:`a` and :math:`b` applied to each sample during copy.
        Defaults to ``NullTransform()``.

    .. property:: slots
        :type: int

        Number of parallel processing pipelines.
        Adjust to achieve the desired CPU utilization for machines with high hardware concurrency.
        Recommended minimum is ``2`` slots to facilitate pipelining of successive blocks.
        The copy itself parallelized across all CPU cores; this field only affects pipeline-level parallelism.
        Defaults to ``2``.

    .. method:: copy()

        Create a copy of this configuration.

        :return CopyProcessorConfig:
            The copy.

OCT
---

These OCT processors perform averaging, resampling, filtering, and FFT operations to transform raw spectra into A-scans.
These operations are applied in the order shown below.

#.  Rolling average with window length :math:`M`.

    .. math::
        \hat{x}[i,j,k] = x[i,j,k] - \frac{1}{M} \sum_{l=0}^{M-1}{ x[i - l,j,k] } ,

#.  Resampling with linear interpolation (:math:`r[j]`).

    .. math::
        z[i,j,k] = \big(\lceil r[j] \rceil - r[j] \big) \hat{x}\big[i, \lfloor r[j] \rfloor, k \big] + \big(r[j] - \lfloor r[j] \rfloor\big) \hat{x}\big[i, \lceil r[j] \rceil, k \big] ,

#.  Frequency-domain filtering (:math:`h[i,j,k]`) and inverse FFT with normalization.

    .. math::
        y[i,j,k] = \log_{10} \left| \frac{1}{N} \mathcal{F}^{-1} \big\{ h[:,j,:] z[i,j,k] \big\} \right|^2

All computation is performed in floating-point until the cast to the output datatype.
Each operation can be enabled, disabled, and/or customized via the processor configuration options.

CPU
+++

.. class:: vortex.process.CPUProcessor

    Perform OCT processing with averaging, resampling by linear interpolation, spectral filtering, and FFT on the CPU.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the processor using the supplied configuration.
        All necessary internal buffers are allocated when this method returns.

        :param CPUProcessorConfig config:
            New configuration to apply.

    .. method:: change(config)

        Change the processor configuration.
        If the change requires buffer reallocation, the pipeline is stalled and record history for the rolling average may be lost.
        May be called while a block is currently processing.

        :param CPUProcessorConfig config:
            New configuration to apply.

    .. method:: next(input_buffer, output_buffer, id=0, append_history=True)

        Process the next buffer.

        :param numpy.ndarray[numpy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param numpy.ndarray[numpy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.
        :param bool append_history:
            Include the raw spectra from this input buffer in the rolling average.

    .. method:: next_async(input_buffer, output_buffer, callback, id=0, append_history=True)

        Process the next buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param numpy.ndarray[numpy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param numpy.ndarray[numpy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param Callable[[Exception], None] callback:
            Callback to execute when buffer is processed.
            The callback receives as an argument any exception which occurred during processing.
        :param int id:
            Number to associate with the buffer for logging purposes.
        :param bool append_history:
            Include the raw spectra from this input buffer in the rolling average.

    .. property:: config
        :type: CPUProcessorConfig

        Copy of the active configuration.

.. class:: vortex.process.CPUProcessorConfig

    Base: :class:`~vortex.process.NullProcessorConfig`

    Configuration object for :class:`~vortex.process.CPUProcessor`.

    .. property:: average_window
        :type: int

        Length :math:`M` of rolling average window in records/A-scans for background subtraction.
        Disable background subtraction by setting to ``0``.
        Disabled by default.

        .. note::
            Record/A-scan history includes both active and inactive records.

        .. warning::
            Record/A-scan history is stored internally within the processor to allow :data:`average_window` to exceed :data:`records_per_block`.
            Create a new processor to discard this history.

    .. property:: resampling_samples
        :type: numpy.ndarray[numpy.float32]

        Optional positions at which to resample the raw spectra prior FFT.
        The number of resampled positions determines :data:`samples_per_record`.
        Valid positions are in the range [``0``, :data:`samples_per_record`].
        Set to an empty array ([]) to disable resampling.
        Disabled by default.

    .. property:: spectral_filter
        :type: numpy.ndarray[numpy.float32 | numpy.complex64]

        Optional spectral filter to multiply with the spectra after resampling but before the FFT.
        Must have shape compatible with [ :data:`samples_per_ascan` ]; that is, the spectral filter should match the length of the output A-scan, not the input record.
        Set to an empty array ([]) to disable spectral filtering.
        Disabled by default.

    .. property:: enable_ifft
        :type: bool

        Enable the FFT for OCT processing with length :math:`N` determined by :data:`samples_per_ascan`.
        When enabled, the FFT is multiplied by the normalization factor of :math:`1 / N`.
        When disabled, the complex magnitude is performed instead and no normalization factor is applied.
        Enabled by default.

        .. note::
            If FFT normalization is not desired, scale the spectral filter by :math:`N` to cancel out the normalization factor.

    .. property:: enable_log10
        :type: bool

        Enable the application of :math:`log_{10}(...)` to the complex magnitude.
        Enabled by default.

    .. property:: enable_square
        :type: bool

        Enabling squaring of the complex magnitude after the FFT.
        When enabled, the processor output is power.
        When disabled, the processor output is amplitude.
        Enabled by default.

    .. property:: enable_magnitude
        :type: bool

        Enable returning the complex magnitude after square and log operations.
        When enabled, the processor output is complex magnitude.
        When disabled, the processor output is the real part only.
        Enabled by default.

    .. property:: levels
        :type: Optional[~vortex.Range]

        When set, rescale values within [``levels.min``, ``levels.max``] to the full range of the output data type.
        This is helpful for quantization when the output data type has a small dynamic range.
        When unset, cast values to the output data type directly.
        Unset by default.

    .. property:: channel
        :type: int

        Index of channel to select for processing.

    .. property:: slots
        :type: int

        Number of parallel processing pipelines.
        Adjust to achieve the desired CPU utilization for machines with high hardware concurrency.
        Recommended minimum is ``2-4`` slots to facilitate pipelining of successive blocks but higher numbers will likely yield better throughput.
        Each processing step is individually parallelized across all CPU cores; this field only affects pipeline-level parallelism.
        Defaults to ``2``.

    .. method:: copy()

        Create a copy of this configuration.

        :return CPUProcessorConfig:
            The copy.

GPU
+++

.. class:: vortex.process.CUDAProcessor

    Perform OCT processing with averaging, resampling by linear interpolation, spectral filtering, and FFT on a CUDA-capable GPU.

    .. method:: __init__(logger=None)

        Create a new object with optional logging.

        :param vortex.Logger logger:
            Logger to receive status messages.
            Logging is disabled if not provided.

    .. method:: initialize(config)

        Initialize the processor using the supplied configuration.
        All necessary internal buffers are allocated when this method returns.

        :param CUDAProcessorConfig config:
            New configuration to apply.

    .. method:: change(config)

        Change the processor configuration.
        May be called while a block is currently processing.
        If the change requires buffer reallocation, the pipeline is stalled and record history for the rolling average may be lost.
        Changes that require no buffer reallocation and take effect immediately are

        - disabling any processing step (e.g., resampling or FFT),
        - reducing the window of the rolling average, and
        - altering a non-empty spectral filter or resampling vector without increasing its length.

        All other changes will likely require a buffer reallocation.

        :param CUDAProcessorConfig config:
            New configuration to apply.

    .. method:: next(input_buffer, output_buffer, id=0, append_history=True)

        Process the next buffer.

        :param cupy.ndarray[cupy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param cupy.ndarray[cupy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param int id:
            Number to associate with the buffer for logging purposes.
        :param bool append_history:
            Include the raw spectra from this input buffer in the rolling average.

    .. method:: next_async(input_buffer, output_buffer, callback, id=0, append_history=True)

        Process the next buffer asynchronously and execute the callback when complete.

        .. caution::
            The callback may be executed in the calling thread before this method returns if an error occurs while queueing the background acquisition.

        :param cupy.ndarray[cupy.uint16] input_buffer:
            The input buffer, with shape that matches :data:`self.config.input_shape <vortex.process.NullProcessorConfig.input_shape>`.
        :param cupy.ndarray[cupy.int8] output_buffer:
            The output buffer, with shape that matches :data:`self.config.output_shape <vortex.process.NullProcessorConfig.output_shape>`.
        :param Callable[[Exception], None] callback:
            Callback to execute when buffer is processed.
            The callback receives as an argument any exception which occurred during processing.
        :param int id:
            Number to associate with the buffer for logging purposes.
        :param bool append_history:
            Include the raw spectra from this input buffer in the rolling average.

    .. property:: config
        :type: CUDAProcessorConfig

        Copy of the current configuration.

.. class:: vortex.process.CUDAProcessorConfig

    Base: :class:`~vortex.process.CPUProcessorConfig`

    Configuration object for :class:`~vortex.process.CUDAProcessor`.

    .. property:: clock_channel
        :type: Optional[int]

        Index of k-clock channel for dynamic resampling.
        Enables k-clock resampling on a per A-scan basis with set to a non-``None`` value.
        Defaults to ``None``.

        .. note::

            Dynamic resampling is only supported when the ``cuda_dynamic_resampling`` feature is present.
            This requires CUDA 11 or higher.

    .. property:: interpret_as_signed
        :type: bool

        Interpret the input data as signed data instead of unsigned data.
        Relevant only for Python where data types are pre-specified.
        Defaults to ``False``.

    .. property:: slots
        :type: int

        Number of parallel CUDA streams to use for processing.
        Recommended minimum is ``2`` slots to facilitate pipelining of successive blocks.
        For GPUs with sufficient compute resources, increasing the number of slots could enable parallel computation.
        Defaults to ``2``.

    .. property:: device
        :type: int

        Index of CUDA device to use for processing.
        Defaults to index ``0``.

    .. method:: copy()

        Create a copy of this configuration.

        :return CUDAProcessorConfig:
            The copy.
