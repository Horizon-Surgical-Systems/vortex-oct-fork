import pytest
import numpy

from vortex.marker import MarkerList
from vortex.scan import RasterScan, RadialScan
from vortex.engine import ScanQueue

@pytest.fixture
def raster_scan():
    scan = RasterScan()

    cfg = scan.config
    cfg.loop = False

    scan.initialize(cfg)

    return scan

@pytest.fixture
def radial_scan():
    scan = RadialScan()

    cfg = scan.config
    cfg.loop = False

    scan.initialize(cfg)

    return scan

@pytest.fixture
def scan_queue():
    return ScanQueue()

@pytest.fixture
def blended_waveform(scan_queue, raster_scan, radial_scan):
    n = len(raster_scan.scan_buffer()) // 4
    markers = MarkerList()
    buffer = numpy.empty((2 * n, raster_scan.config.channels_per_sample))

    # start with one scan
    scan_queue.append(raster_scan)
    (samples, generated) = scan_queue.generate(markers, buffer[:n])
    assert samples == n
    assert generated == n

    # switch to another
    scan_queue.interrupt(radial_scan)
    (samples, generated) = scan_queue.generate(markers, buffer[n:])
    assert samples == n
    assert generated == n

    return buffer

def test_scan_blend_repetition(blended_waveform):
    diff = numpy.diff(blended_waveform, axis=0)

    assert not numpy.isclose(diff, 0).all(axis=1).any()

def test_scan_blend_limits(raster_scan, blended_waveform):
    cfg = raster_scan.config
    dt = cfg.sampling_interval

    q = blended_waveform
    qd = numpy.diff(q, 1, axis=0) / dt
    qdd = numpy.diff(q, 2, axis=0) / dt**2

    for i in range(q.shape[1]):
        assert numpy.abs(qd[:, i]).max() <= cfg.limits[i].velocity
        assert numpy.abs(qdd[:, i]).max() <= cfg.limits[i].acceleration

def test_callbacks(scan_queue, raster_scan):

    # start scan queue at start of scan
    segments = raster_scan.scan_segments()
    scan_queue.reset(0, segments[0].position[0], segments[0].entry_velocity(raster_scan.config.samples_per_second))

    # set up event handlers
    events = []
    def event_handler(code, event):
        nonlocal events
        events.append((code, event))

    empty = 0
    def empty_handler(osq):
        nonlocal empty
        if empty == 0:
            osq.append(raster_scan, callback=lambda _, e: event_handler(7, e))
        empty += 1
    scan_queue.empty_callback = empty_handler

    # generate a buffer that will drain the scan queue
    n = len(raster_scan.scan_buffer())
    markers = MarkerList()
    buffer = numpy.empty((3 * n, raster_scan.config.channels_per_sample))

    # only append a single scan
    scan_queue.append(raster_scan, callback=lambda _, e: event_handler(3, e))
    (samples, generated) = scan_queue.generate(markers, buffer)

    assert samples < generated
    assert generated == buffer.shape[0]

    assert empty == 2
    assert events == [
        (3, ScanQueue.Event.Start),
        (3, ScanQueue.Event.Finish),
        (7, ScanQueue.Event.Start),
        (7, ScanQueue.Event.Finish)
    ]

if __name__ == '__main__':
    pytest.main()
