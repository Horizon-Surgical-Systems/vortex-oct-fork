from ast import Import
from time import sleep

import pytest

import numpy

try:
    from vortex import get_console_logger as gcl
    from vortex.acquire import AlazarAcquisition, AlazarFFTAcquisition, alazar
    from vortex.engine import dispersion_phasor
    from vortex.memory import CudaHostTensorUInt16
except ImportError:
    pytest.skip('Alazar support is not available')

background = None

def test_acquire():
    a = AlazarAcquisition(gcl('acquire', 0))
    ac = a.config

    ac.records_per_block = 1000
    ac.samples_per_record = 768

    ac.clock = alazar.ExternalClock()
    ac.inputs.append(alazar.Input(alazar.Channel.A))
    ac.options.clear()

    a.initialize(ac)
    a.prepare()

    buffers = [numpy.zeros(ac.shape, dtype=numpy.uint16) for _ in range(10)]

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

    global background
    background = numpy.mean(buffers[0], axis=0).astype(numpy.int16)[:, 0] + 2**15

    from matplotlib import pyplot
    for i in range(5):
        pyplot.plot(buffers[0][i])
    pyplot.show()

def test_acquire_fft():
    a = AlazarFFTAcquisition(gcl('acquire', 0))
    ac = a.config

    ac.records_per_block = 1000
    ac.samples_per_record = 768
    ac.fft_length = 1024
    ac.include_time_domain = True

    global background
    # TODO: acquire background within this test
    ac.background = numpy.concatenate((background, numpy.zeros((ac.fft_length - ac.samples_per_record,))))
    dispersion = (-7e-5, 0)
    window = numpy.hamming(ac.samples_per_record)
    phasor = numpy.conj(dispersion_phasor(len(window), dispersion))
    ac.spectral_filter = numpy.concatenate((window * phasor, numpy.zeros((ac.fft_length - ac.samples_per_record,))))

    ac.clock = alazar.ExternalClock()
    ac.inputs.append(alazar.Input(alazar.Channel.A))
    ac.options.clear()

    a.initialize(ac)
    a.prepare()

    buffers = [CudaHostTensorUInt16() for _ in range(30)]
    for b in buffers:
        b.resize(ac.shape)

    count = None
    exc = None
    def cb(n, e):
        nonlocal count, exc
        count = n
        exc = e

    for (i, b) in enumerate(buffers):
        with b as data:
            a.next_async(data, cb, id=i)

    a.start()
    sleep(1)
    a.stop()

    if exc:
        raise exc
    assert count == buffers[0].shape[0]
    with buffers[0] as d0:
        with buffers[1] as d1:
            assert (d0 != d1).any()

    from matplotlib import pyplot
    for i in range(len(buffers)):
        with buffers[i] as data:
            pyplot.plot(data[0] + i * 2**16)
    pyplot.show()

if __name__ == '__main__':
    pytest.main()
