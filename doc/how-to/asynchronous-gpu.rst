Asynchronous GPU Operation
==========================

*vortex* performs all GPU operations asynchronously to maximize throughput and exposes its CUDA resources to Python through :mod:`cupy`.
A Python application that has registered endpoint callbacks can safely

.. caution::

    *vortex* protects CUDA buffers with locks to prevent simultaneous modification, especially resizing.
    Entering a context manager to access a CUDA buffer in Python transparently acquires the associated lock.
    Avoid modifying a *vortex* endpoint with its underling buffer locked as this may produce a deadlock.

Chain Operations
----------------

Chaining additional processing on to a *vortex* endpoint is as simple as accessing its CUDA buffer and issuing :mod:`cupy` calls to the proper stream.
This example below shows how to generate a maximum intensity projection (MIP) from within a :data:`~vortex.engine.StackDeviceTensorEndpointInt8.volume_callback`.

.. code-block:: python

    def handler(*args):
        # access and lock the buffer
        with endpoint.tensor as volume:
            # dispatch to stream
            with endpoint.stream as stream:
                # create the MIP
                mip: numpy.ndarray = cupy.max(volume, axis=2).get()
                # (optional) create an event to monitor completion
                done: cupy.cuda.Event = stream.record()

The MIP is created asynchronously so it has not necessarily been created once ``handler()`` returns.
Any subsequent access to ``mip`` that requires its computation to have completed should call :meth:`done.synchronize() <cupy.cuda.Event.synchronize>`.

User-Managed Buffering
----------------------

For applications where you would like to buffer volumes before decided their disposition (i.e., save or discard), you can asynchronously chain copy operations to efficiently add volumes to your own buffer.
The workflow is as follows:

#.  Configure a :class:`~vortex.engine.StackDeviceTensorEndpointInt8` for the engine.

#.  Attach a handler for the endpoint's :data:`~vortex.engine.StackDeviceTensorEndpointInt8.volume_callback` which copies that data to your own buffer.
    This will be very fast operation if you maintain your buffers on the GPU.

    .. code-block:: python

        my_endpoint = StackDeviceTensorEndpoint(...)
        my_buffers = []

        def handler(sample, scan_idx, volume_idx):
            with my_endpoint.tensor as volume:
                with my_endpoint.stream:
                    my_buffers.append(cupy.copy(volume))
        my_endpoint.volume_callback = handler

#.  Perform the acquisition.

#.  Whenever the acquisition is completed, you can then inspect ``my_buffers`` to determine further action.
    If you wish to save volumes at this point, you can use standard APIs (e.g., :func:`numpy.save()`) or you can use the Python bindings for the *vortex* storage objects (e.g., :func:`~vortex.storage.SimpleStackInt8.write_volume()`).

.. note::

    *vortex* executes callbacks once the formatting operations have been dispatched to the GPU but not necessarily after they have completed.
    This is why the :func:`cupy.copy()` call above is performed in the endpoint's CUDA stream, which chains the copy to the queued operations.
    If you need the data immediately, call :meth:`my_endpoint.stream.synchronize() <cupy.cuda.Stream.synchronize>` to wait until the GPU has caught up.
