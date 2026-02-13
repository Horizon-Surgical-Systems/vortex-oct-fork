Troubleshoot
============

This document is a compilation of troubleshooting steps for commonly encountered issues.

Buffering
---------

Underflow
+++++++++

    | digital write failed: (-200290) buffer underflow
    | analog write failed: (-200290) buffer underflow

*vortex* generates I/O waveforms on the fly, rather than relying on regeneration.
If computational overhead or timing glitches cause *vortex* to miss an I/O deadline, buffer underflow errors occur, such as those above from NI DAQmx hardware.
Similarly, *vortex* allocates all of its memory buffers in advance of the acquisition.
If data processing or slow storage fail to recycle buffers to *vortex* in a timely fashion, the above buffer underflow errors occur.
You can attempt the following steps in increasing order of difficulty.

#.  For resistance to timing glitches, increase the :data:`EngineConfig.preload_count <vortex.engine.EngineConfig.preload_count>` setting for the :class:`~vortex.engine.Engine`.
    This will give *vortex* more tolerance in meeting the I/O deadline for signal generation.

#.  For tolerance to buffering, increase the :data:`EngineConfig.blocks_to_allocate <vortex.engine.EngineConfig.preload_count>` setting for the :class:`~vortex.engine.Engine`.
    This will give *vortex* more space to buffer in memory before underflowing.

#.  Reduce the computational or storage requirements of the acquisition by shortening or :attr:`slicing <vortex.format.StackFormatExecutorConfig.sample_slice>` A-scans.

#.  If building *vortex* from source, check that code optimization is enabled.
    Optimization is already enabled for the Python binary wheels.

#.  Reduce the overall computational load on the computer.

#.  Consider increasing the priority level of the *vortex* process.
    This may need to be done every time the *vortex* process is launched.

#.  Migrate complex or time-consuming processing out of *vortex*'s callback threads and into a background task.

#.  Upgrade the computer's computational or storage capabilities.
    In general, solid-state storage is necessary for time-unlimited streaming to disk at typical A-scan rates.
    Otherwise, the acquisition duration is limited to the available system memory.

#.  If using Python, transition to C++ use of *vortex*.

#.  If using Windows, transition to Linux for better real-time performance.

Insufficient Resources
++++++++++++++++++++++

    | (533) ApiInsufficientResources
    | There were not enough system resources to complete this operation. The most common reason of this return code is using too many DMA buffers, or using DMA buffers that are too big. Please try reducing the number of buffers posted to the board at any time, and/or try reducing the DMA buffer sizes.

This error is specific to AlazarTech digitizers.
The problem is that :attr:`EngineConfig.preload_count <vortex.engine.EngineConfig.preload_count>` or :attr:`EngineConfig.records_per_block <vortex.engine.EngineConfig.records_per_block>` is too large.
Try the acquisition with either or both reduced.

Trigger
-------

Timeouts
++++++++

    | (579) ApiWaitTimeout
    | The operation did not finish during the timeout interval. Try the operation again, or abort the acquisition.

This suggests that the trigger is not connected or configured correctly for the acquisition device.

#.  Check that the trigger is wired correctly.
#.  Check that the acquisition trigger configuration specifies the correct trigger input, coupling, and level.

.. note::

    Some Alazar cards require a TTL trigger input, which is not the default in :attr:`SingleExternalTrigger.range_millivolts <vortex.acquire.SingleExternalTrigger.range_millivolts>`.


Analog/Digital Output Errors
++++++++++++++++++++++++++++

    | analog write failed: (-200292)
    | Some or all of the samples to write could not be written to the buffer yet. More space will free up as samples currently in the buffer are generated. To wait for more space to become available, use a longer write timeout. To make the space available sooner, increase the sample rate.

This usually occurs when *vortex* is generating scan waveforms faster than the DAQmx card is clocking them out, which suggests a trigger problem.

#.  Check that your analog out trigger is wired correctly.
#.  Check that you have configured the correct trigger input.
    For example, set ``ioc_out.clock.source = 'pfi12'`` instead of the the default of ``'pfi0'``, such as in `demo/_common/engine.py <https://gitlab.com/vortex-oct/vortex/-/blob/develop/demo/_common/engine.py>`_.

    | error during I/O for block 0: digital write failed: (-201434)
    | Configuration failed because the task tried to change the direction of a line while the watchdog timer is expired. Clear the expiration of the watchdog timer task before trying to change the direction of any line, even if the line is not watched by the watchdog timer task.

This suggests that a prior program configured the watchdog but did not clear it upon shutdown.
Fix the condition by resetting the affected card in NI MAX.

Instability
+++++++++++

    Each time an acquisition is started, the scan appears shifted.
    The amount of the shift varies between acquisitions but remains constant for a given acquisition.

This suggests that there is incorrect hardware synchronization when the acquisition starts.
Usually, the acquisition components control or generate triggers for synchronized IO components.
*vortex* will start the acquisition or IO components marked as the master trigger source last when setting up an acquisition to ensure hardware synchronization.
Both incorrect configuration and electrical problems can cause loss of synchronization.

    When the acquisition is running for a long time, the scan starts to shift slowly.

This suggests that the IO hardware is slowly losing synchronization with the acquisition hardware.
This is frequently an electrical problem that produces dropped triggers.

In both cases, the troubleshooting steps are the same.

#.  Check that a single acquisition component has ``master=True`` when added to the engine configuration in :meth:`EngineConfig.add_acquisition() <vortex.engine.EngineConfig.add_acquisition>`.
    This is the default if not specified.
#.  Check that IO components have ``master=False`` when added to the engine configuration in :meth:`EngineConfig.add_io() <vortex.engine.EngineConfig.add_io>`.
    This is the default if not specified.
#.  Check that there is a single electrical trigger source and that it is wired correctly.
#.  Check that the trigger electrical connections are intact and secure.
#.  Check that the configured trigger levels (if any) are set appropriately in *vortex* or external software.
#.  Probe the trigger line(s) with an oscilloscope to ensure that triggers are not reaching the IO hardware prior to the acquisition starting and that triggers are not dropping.
    Configure *vortex* to output a square wave at half the trigger rate to determine when IO hardware is receiving triggers.
    The configuration necessary to do this is present in most demos, such as the :ref:`demo/live-view` demo.

.. seealso::

    See :attr:`EngineConfig.strobes <vortex.engine.EngineConfig.strobes>` and :ref:`strobe-signal-missing` for help setting up the square wave output.

Scan Pattern
------------

Limit Violations
++++++++++++++++

    | axis 1 violated velocity limit during segment 0 at sample 159: 10050.251256281583 > 8000

This message appears when the scan velocity through a segment (B-scan) exceeds the limit configured for the scanners. You have several options.

#.  Reduce the scan velocity by reducing the segment length or increasing the number of samples per segment.
#.  Increase the scanner limits (e.g., increase the velocity of :attr:`RasterScanConfig.limits`).
#.  Disable limit checks (:attr:`RasterScanConfig.bypass_limits_check`).

If the configured scan velocity limit does in fact reflect the maximum capabilities of your scanner, the second two options will result in scan distortion.

Strobes
-------

.. _strobe-signal-missing:

Strobe Signal Missing
+++++++++++++++++++++

The default strobe configuration (in C++) is below.

.. code-block:: c++

    {
        strobe::sample(0, 2),
        strobe::sample(1, 1000),
        strobe::sample(2, 1000, strobe::polarity_t::low),
        strobe::segment(3),
        strobe::volume(4)
    };

This produces the following signals.

-   **Line 0**: Square wave with period of half the A-scan rate.
-   **Line 1**: Pulse high every 1000 A-scans.
-   **Line 2**: Pulse low every 1000 A-scans.
-   **Line 3**: Pulse high at the start of every segment.
-   **Line 4**: Pulses high at the start of every volume.

Perform the following troubleshooting steps to identify the cause of a missing strobe output.

#.  Check that the strobe IO object is getting added to the engine.
    Make sure you have the line ``ec.add_io(self._strobe)`` included in your script.
#.  If you are using :class:`~vortex.scan.FreeformScan`, you may want to try a :class:`~vortex.scan.RasterScan` for debugging scan-derived strobes.
    This would exclude any issues with scan pattern generation.
#.  If you are troubleshooting with an oscilloscope, check that the segment strobe is not too brief to trigger capture.
    For NI DAQmx hardware, you may want to try connecting to ``DevX/portX.0``, which should have a square wave at half the A-scan rate.

Storage
-------

Missing Data
++++++++++++

Storage objects, such as :class:`~vortex.storage.SimpleStackInt8`, allow unlimited streaming to disk and therefore cannot finish writing the file header until the acquisition has finished.
In C++, all *vortex* storage objects automatically update the file header in their destructors.
In Python, however, an explicit call to :func:`~vortex.storage.SimpleStackInt8.close()` is required because Python may skip destructors during interpreter shutdown.

A classic symptom of this issue is that a large file is written to disk but reports that it contains zero volumes when loaded.

.. code-block:: python

    >>> os.path.getsize('data.npy')
    51200256
    >>> data = np.load('data.npy')
    >>> data
    array([], shape=(0, 100, 500, 1024), dtype=int8)

The solution is to keep track of all storage objects and explicitly call :func:`~vortex.storage.SimpleStackInt8.close()` when the acquisition finishes.
This is readily accomplished by appending storage objects to a list as they are created.

Recovery
^^^^^^^^

.. caution::

    Data recovery as described below is neither guaranteed nor supported.

If you need to recover the saved data, you can attempt to do so as follows for NumPy files.

.. code-block:: python

    # determine the volume shape from the unfinalized header
    data = np.load('data.npy')
    volume_shape = data.shape[1:]
    volume_dtype = data.dtype

    # read in all data as a flat array, skipping the 256 byte header
    # NOTE: the header size is specific to vortex (see src/vortex/storage/detail/header.hpp)
    data = np.fromfile('data.npy', volume_dtype, offset=256)

    # determine volume count
    volume_size = np.prod(volume_shape)
    volume_count = len(data) // np.prod(volume_shape)

    # complete a partial final volume with zeros
    missing = len(data) - volume_count * volume_size
    if missing:
        data = np.concatenate((raw_data, np.zeros((missing,), volume_dtype)))

    # transform into volumes
    data = data.reshape((-1, *volume_shape))

For files without a header, you can follow a similar procedure except that you must manually specify ``volume_shape`` and ``volume_dtype`` instead of reading them from the file.
