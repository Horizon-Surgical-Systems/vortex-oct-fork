import pytest
from math import pi

import numpy as np

import imageio

from vortex import Range
from vortex.format import RadialFormatExecutor, PositionFormatExecutor

from util import get_compute_capability

def test_radial_formatter():
    cupy = get_compute_capability(35)

    formatter = RadialFormatExecutor()

    cfg = formatter.config
    cfg.volume_xy_extent = (Range(-1, 1), Range(-1, 1))
    cfg.segment_rt_extent = (Range(-1, 1), Range(0, pi))
    cfg.radial_segments_per_volume = 8
    cfg.radial_records_per_segment = 50

    formatter.initialize(cfg)

    volume = cupy.zeros((100, 100, 10), dtype=cupy.int8)
    segments = [cupy.zeros((cfg.radial_records_per_segment, volume.shape[-1]), dtype=cupy.int8) + int(v) for v in np.linspace(20, 100, cfg.radial_segments_per_volume)]

    for (idx, segment) in enumerate(segments):
        formatter.format(volume, segment, idx)

    imageio.imwrite('test-radial.png', cupy.asnumpy(volume[:, :, 0]).astype(np.uint8))

def test_position_formatter():
    cupy = get_compute_capability(35)

    formatter = PositionFormatExecutor()

    cfg = formatter.config
    cfg.set(offset=(0, 50), angle=-pi/2)

    formatter.initialize(cfg)

    volume = cupy.zeros((100, 100, 10), dtype=cupy.int8)

    segment = cupy.empty((50, volume.shape[2]), dtype=cupy.int8)
    segment[:] = cupy.linspace(20, 100, 50)[:, np.newaxis]

    position = cupy.empty((segment.shape[0], 2), dtype=cupy.float64)
    position[:] = cupy.arange(len(segment))[:, np.newaxis]

    formatter.format(volume, position, position, segment, 0)

    imageio.imwrite('test-position.png', cupy.asnumpy(volume[:, :, 0]).astype(np.uint8))

if __name__ == '__main__':
    pytest.main()
