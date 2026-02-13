import pytest
import numpy
from time import sleep

try:
    import vortex
    from vortex import get_console_logger as gcl
    from vortex.acquire import TeledyneAcquisition, teledyne
except ImportError:
    pytest.skip('Teledyne support is not available', allow_module_level=True)

def test_acquire():

    devices = teledyne.enumerate()
    print("available devices:", devices)
    assert len(devices) == 1

    a = TeledyneAcquisition(gcl('acquire', 0))
    ac = a.config
    ac.inputs.append(teledyne.Input(0))
    ac.inputs.append(teledyne.Input(1))

    print("clock:", ac.clock)
    print("inputs:", ac.inputs)

    ac.records_per_block = 1
    ac.samples_per_record = 7680

    a.initialize(ac)
    a.prepare()

    buffers = [numpy.zeros(ac.shape, dtype=numpy.uint16) for _ in range(1)]

    count = None
    exc = None
    def cb(n, e):
        nonlocal count, exc
        count = n
        exc = e

    for (i, b) in enumerate(buffers):
        a.next_async(b, cb, id=i)

    a.start()
    sleep(1)
    a.stop()

    if exc:
        raise exc
    assert count == buffers[0].shape[0]
    assert (buffers[0] != buffers[1]).any()

    #from matplotlib import pyplot
    #for i in range(5):
    #    pyplot.plot(buffers[0][i])
    #pyplot.show()

if __name__ == '__main__':
    test_acquire()
#    pytest.main()
