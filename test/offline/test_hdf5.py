import os
from tempfile import mkstemp

import pytest

import numpy
from numpy.testing import assert_allclose

import vortex

@pytest.fixture(params=[(31, 17, 7, 1), (89, 43, 12, 3)])
def shape(request):
    return request.param

@pytest.fixture
def random_data(shape):
    rng = numpy.random.default_rng(0)

    data = rng.integers(-128, 127, size=shape, dtype=numpy.int8)

    return data

@pytest.fixture
def random_path(random_data):
    (fd, path) = mkstemp()

    yield path

    os.close(fd)
    os.remove(path)

def test_hdf5_storage_matlab(random_data, random_path):
    try:
        from vortex.storage import HDF5StackInt8, HDF5StackHeader
    except ImportError:
        pytest.skip('vortex is missing HDF5 support')

    h5py = pytest.importorskip('h5py')

    storage = HDF5StackInt8()
    cfg = storage.config

    cfg.path = random_path
    cfg.shape = random_data.shape
    cfg.header = HDF5StackHeader.MATLAB

    # write data
    storage.open(cfg)
    storage.write_volume(random_data)
    storage.close()

    # read data
    with h5py.File(random_path, 'r') as f:
        output_data = numpy.array(f.get('data'))

    assert_allclose(output_data, random_data[numpy.newaxis, ...])
