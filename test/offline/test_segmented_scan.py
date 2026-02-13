import pytest
import numpy
from numpy.testing import assert_allclose

from vortex.marker import MarkerList
from vortex.scan import RasterScan, RepeatedRasterScan, RadialScan, RepeatedRadialScan

@pytest.fixture(params=[RasterScan, RepeatedRasterScan, RadialScan, RepeatedRadialScan])
def scan(request):
    obj = request.param()
    cfg = obj.config

    cfg.loop = True
    obj.initialize(cfg)

    return obj

def _drain_scan(scan, count):
    markers = MarkerList()
    waveforms = numpy.empty((count, 2))

    n = scan.next(markers, waveforms)
    waveforms = waveforms[:n]

    return (waveforms, markers)

def test_restart(scan):
    scan.restart(0, (0, 0), (0, 0), True)

    # trigger buffering
    n = len(scan.scan_buffer())

    # obtain refernece data
    (waveforms1, markers1) = _drain_scan(scan, n // 2)

    # trigger the bug by restarting after fully buferring
    scan.restart(0, (0, 0), (0, 0), True)

    # read fewer samples than the buffered size to trigger the bug
    (waveforms2, markers2) = _drain_scan(scan, n // 2)

    assert_allclose(waveforms1, waveforms2)
    assert markers1 == markers2

if __name__ == '__main__':
    pytest.main()
