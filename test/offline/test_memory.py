import pytest

import numpy as np

import vortex

from util import get_cupy_or_skip, get_compute_capability

@pytest.fixture(params=['CudaDevice', 'CudaHost', 'Cpu'])
def device(request):
    if 'Cuda' in request.param:
        # check for CUDA support
        get_compute_capability(0)

    return request.param

SHAPES = [[100], [10, 20], [10, 20, 30], [10, 20, 30, 40]]
@pytest.fixture(params=SHAPES, ids=['x'.join([str(n) for n in s]) for s in SHAPES])
def shape(request):
    return request.param

@pytest.fixture(params=['Int8', 'UInt16', 'UInt64', 'Float32', 'Float64'])
def dtype_name(request):
    return request.param

@pytest.fixture
def dtype(dtype_name):
    return getattr(np, dtype_name.lower())

@pytest.fixture
def tensor(device, dtype_name):
    return getattr(vortex.memory, f'{device}Tensor{dtype_name}')()

def test_properties(tensor, shape):
    assert not tensor.valid
    assert tensor.shape == []

    tensor.resize(shape)
    assert tensor.valid
    assert tensor.shape == shape
    assert tensor.dimension == len(shape)
    assert tensor.count == np.prod(shape)

    tensor.clear()
    assert not tensor.valid

def test_readback(tensor, shape, dtype):
    # generate test data with appropriate range
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        ref = np.random.randint(info.min, info.max, shape, dtype)
    else:
        ref = np.random.random(shape).astype(dtype)

    tensor.resize(shape)
    with tensor as volume:
        data = ref.copy()

        if hasattr(tensor, 'device'):
            cupy = get_cupy_or_skip()
            data = cupy.asarray(data)

        volume[:] = data

        if hasattr(tensor, 'device'):
            volume = volume.get()
        np.testing.assert_allclose(volume, ref)
