import shutil
from tempfile import mkdtemp
from pathlib import Path

import numpy as np

from vortex.storage import SimpleStackHeader, SimpleStackConfig, SimpleStackUInt16, SimpleStackInt8, SimpleStackFloat64

import pytest

@pytest.fixture(params=[(3, 11, 21, 31, 1), (7, 123, 53, 82, 2)])
def shape(request):
    return request.param

@pytest.fixture(params=[(np.uint16, SimpleStackUInt16), (np.int8, SimpleStackInt8)]) #, (np.float64, SimpleStackFloat64)])
def dtype_storage_class_pair(request):
    return request.param

@pytest.fixture()
def dtype(dtype_storage_class_pair):
    return dtype_storage_class_pair[0]

@pytest.fixture()
def storage_class(dtype_storage_class_pair):
    return dtype_storage_class_pair[1]

@pytest.fixture
def volumes(shape, dtype):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return data

@pytest.fixture
def random_directory():
    path = mkdtemp()

    yield Path(path)

    shutil.rmtree(path, ignore_errors=True)

def test_nrrd(volumes, random_directory, storage_class):
    nrrd = pytest.importorskip('nrrd')

    cfg = SimpleStackConfig()
    cfg.shape = volumes[0].shape
    cfg.header = SimpleStackHeader.NRRD
    cfg.path = (random_directory / 'test.nrrd').as_posix()

    write(volumes, cfg, storage_class())

    volumes_nrrd, _ = nrrd.read(cfg.path, index_order='C')

    np.testing.assert_equal(volumes, volumes_nrrd)

def test_nifti(volumes, random_directory, storage_class):
    nib = pytest.importorskip('nibabel')

    cfg = SimpleStackConfig()
    cfg.shape = volumes[0].shape
    cfg.header = SimpleStackHeader.NIfTI
    cfg.path = (random_directory / 'test.nii').as_posix()

    write(volumes, cfg, storage_class())

    nib.arrayproxy.ArrayProxy._default_order = 'C'
    volumes_nifti = np.asanyarray(nib.load(cfg.path).dataobj)

    np.testing.assert_equal(volumes, volumes_nifti)

def write(volumes, cfg, storage):
    storage.open(cfg)
    for volume in volumes:
        storage.write_volume(volume)
        storage.advance_volume()
    storage.close()
