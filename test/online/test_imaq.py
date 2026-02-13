from time import sleep

import pytest

import numpy

try:
    from vortex import get_console_logger as gcl
    from vortex.acquire import ImaqAcquisition
except ImportError:
    pytest.skip('NI IMAQ support is not available', allow_module_level=True)

background = None

def test_acquire():
    a = ImaqAcquisition(gcl('acquire', 0))
    ac = a.config

    ac.offset = (10, 20)
    ac.records_per_block = 200
    ac.samples_per_record = 768

    a.initialize(ac)
    a.prepare()

    buffers = [numpy.zeros(ac.shape, dtype=numpy.uint16) for _ in range(10)]

    count = None
    exc = None
    def cb(n, e):
        nonlocal count, exc
        count = n
        exc = e

    a.start()
    for (i, b) in enumerate(buffers):
        a.next_async(b, cb, id=i)
    sleep(2)
    a.stop()

    if exc:
        raise exc
    assert count == buffers[0].shape[0]
    assert (buffers[0] != buffers[1]).any()

    from matplotlib import pyplot
    pyplot.imshow(numpy.row_stack(buffers))
    pyplot.show()

if __name__ == '__main__':
    pytest.main()
