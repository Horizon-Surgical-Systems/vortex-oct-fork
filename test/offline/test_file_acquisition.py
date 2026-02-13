from math import ceil
import os
from tempfile import mkstemp
from numpy import random

import pytest

import numpy
from numpy.testing import assert_allclose

from vortex.acquire import FileAcquisition

@pytest.fixture(params=[True, False])
def loop(request):
    return request.param

@pytest.fixture(params=[(100, 200, 1), (100, 200, 2)])
def shape(request):
    return request.param

@pytest.fixture(params=[23, 100*200, 48043])
def random_data(request):
    rng = numpy.random.default_rng(0)

    data = rng.integers(2**16 - 1, size=request.param, dtype=numpy.uint16)

    return data

@pytest.fixture
def random_data_path(random_data):
    (fd, path) = mkstemp()
    open(path, 'wb').write(random_data.tobytes())

    yield path

    os.close(fd)
    os.remove(path)

@pytest.fixture
def acquire(loop, shape, random_data_path):
    acquire = FileAcquisition()
    ac = acquire.config

    ac.records_per_block = shape[0]
    ac.samples_per_record = shape[1]
    ac.channels_per_sample = shape[2]
    ac.loop = loop
    ac.path = random_data_path

    acquire.initialize(ac)
    acquire.start()

    yield acquire

    acquire.stop()

def test_file_acquire(loop, shape, random_data, acquire):
    blocks = []
    block = numpy.zeros(shape, dtype=numpy.uint16)
    for i in range(int(random_data.size / block.size) * 2 + 2):
        if loop:
            n = acquire.next(block)
            assert n == len(block)

        else:
            if (i + 1) * block.size > len(random_data):
                if len(random_data) % (block.shape[1] * block.shape[2]) != 0:
                    with pytest.raises(RuntimeError):
                        n = acquire.next(block)

                    break

                else:
                    n = acquire.next(block)
                    assert n < len(block)

            else:
                n = acquire.next(block)
                assert n == len(block)

        blocks.append(block[:n].copy())
        if n < len(block):
            break

    if blocks:
        load_data = numpy.concatenate([b.flatten() for b in blocks])
        ref_data = numpy.concatenate([random_data] * ceil(len(load_data) / len(random_data)))[:len(load_data)]

        assert_allclose(load_data, ref_data)

if __name__ == '__main__':
    pytest.main()
