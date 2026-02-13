import pytest
import numpy

from vortex.marker import MarkerList
from vortex.scan import RasterScan, RepeatedRasterScan, RadialScan, RepeatedRadialScan
from vortex_tools.scan import partition_segments_by_activity

@pytest.fixture(params=[RasterScan, RepeatedRasterScan, RadialScan, RepeatedRadialScan])
def scan(request):
    obj = request.param()
    cfg = obj.config

    cfg.loop = True
    obj.initialize(cfg)

    return obj

@pytest.fixture
def waveforms(scan):
    return scan.scan_buffer()

@pytest.fixture
def markers(scan):
    return scan.scan_markers()

def test_segments(scan, waveforms, markers):
    cfg = scan.config
    (active, _) = partition_segments_by_activity(markers, waveforms)

    segments_per_volume = cfg.segments_per_volume
    try:
        segments_per_volume *= cfg.repeat_count
    except AttributeError:
        pass
    try:
        if cfg.bidirectional_volumes:
            segments_per_volume *= 2
    except AttributeError:
        pass

    assert len(active) == segments_per_volume
    for s in active:
        assert len(s) == cfg.samples_per_segment

def assert_limits_satisfied(dt, q, limits):
    qd = numpy.diff(q, 1, axis=0) / dt
    qdd = numpy.diff(q, 2, axis=0) / dt**2

    for i in range(q.shape[1]):
        assert q[:, i].min() >= limits[i].position.min
        assert q[:, i].max() <= limits[i].position.max

        assert numpy.abs(qd[:, i]).max() <= limits[i].velocity
        assert numpy.abs(qdd[:, i]).max() <= limits[i].acceleration

def test_limits(scan, waveforms):
    cfg = scan.config
    assert_limits_satisfied(cfg.sampling_interval, waveforms, cfg.limits)

def test_change_limits(scan, waveforms):
    cfg = scan.config

    n = len(waveforms) // 2
    markers = MarkerList()
    buffer = numpy.empty((2 * n, cfg.channels_per_sample))

    samples = scan.next(markers, buffer[:n])
    assert samples == n

    cfg.angle = 5
    scan.change(cfg)

    samples = scan.next(markers, buffer[n:])
    assert samples == n

    assert_limits_satisfied(cfg.sampling_interval, buffer, cfg.limits)

if __name__ == '__main__':
    pytest.main()
