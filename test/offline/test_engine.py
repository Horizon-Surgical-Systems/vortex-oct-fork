import os
from tempfile import mkstemp
from threading import Lock

import pytest

import numpy
from numpy.testing import assert_allclose

from vortex.acquire import NullAcquisition
from vortex.engine import Engine, StreamDumpStorage, NullEndpoint, Block
from vortex.format import FormatPlanner
from vortex.io import NullIO
from vortex.process import NullProcessor
from vortex.scan import RasterScan
from vortex.storage import StreamDump, StreamDumpConfig

from util import get_compute_capability

@pytest.fixture
def options():
    class cfg:
        bscans_per_volume = 100
        ascans_per_bscan = 100
        ascans_per_block = 325
        samples_per_ascan = 867
        blocks_to_acquire = 100
        blocks_to_allocate = 4
        preload_count = 0

    return cfg

@pytest.fixture
def scan(options):
    scan = RasterScan()
    sc = scan.config

    sc.bscans_per_volume = options.bscans_per_volume
    sc.ascans_per_bscan = options.ascans_per_bscan
    sc.loop = True

    scan.initialize(sc)
    return scan

@pytest.fixture
def acquire(options):
    acquire = NullAcquisition()
    ac = acquire.config

    ac.records_per_block = options.ascans_per_block
    ac.samples_per_record = options.samples_per_ascan
    ac.channels_per_sample = 1

    acquire.initialize(ac)
    return acquire

@pytest.fixture
def process(acquire):
    process = NullProcessor()
    pc = process.config

    pc.samples_per_record = acquire.config.samples_per_record
    pc.ascans_per_block = acquire.config.records_per_block

    process.initialize(pc)
    return process

@pytest.fixture
def format(options):
    format = FormatPlanner()
    fc = format.config

    fc.segments_per_volume = options.bscans_per_volume
    fc.records_per_segment = options.ascans_per_bscan

    format.initialize(fc)
    return format

def test_events_completion(options, scan, acquire, process, format):
    # check for CUDA support
    get_compute_capability(0)

    lock = Lock()

    engine = Engine()
    ec = engine.config

    ec.add_acquisition(acquire, [process])
    ec.add_processor(process, [format])
    ec.add_formatter(format, [NullEndpoint()])

    ec.preload_count = options.preload_count
    ec.records_per_block = options.ascans_per_block
    ec.blocks_to_allocate = options.blocks_to_allocate
    ec.blocks_to_acquire = options.blocks_to_acquire

    engine.initialize(ec)
    engine.prepare()

    engine.scan_queue.append(scan)

    events = []
    def event_handler(evt, exc):
        nonlocal events
        with lock:
            events.append((evt, exc))

    engine.event_callback = event_handler

    engine.start()
    engine.wait()
    engine.stop()

    with lock:
        assert events == [
            (Engine.Event.Launch, None),
            (Engine.Event.Run, None),
            (Engine.Event.Start, None),
            (Engine.Event.Complete, None),
            (Engine.Event.Stop, None),
            (Engine.Event.Exit, None),
        ]

@pytest.fixture(params=[0.5, 1.5])
def galvo_delay_samples(request, options):
    return int(request.param * options.ascans_per_block)

def test_galvo_delay(options, scan, acquire, process, format, galvo_delay_samples):
    # check for CUDA support
    get_compute_capability(0)

    delays = [0, galvo_delay_samples // 3, galvo_delay_samples]
    streams = [Block.StreamIndex.GalvoTarget, Block.StreamIndex.SampleTarget]
    temps = []

    sdss = []
    for delay in delays:
        for stream in streams:
            temps.append(mkstemp())

            sdc = StreamDumpConfig()
            sdc.path = temps[-1][1]
            sdc.stream = stream

            sds = StreamDump()
            sds.open(sdc)
            sdss.append((delay, sds))

    engine = Engine()
    ec = engine.config

    ec.add_acquisition(acquire, [process])
    ec.add_processor(process, [format])
    ec.add_formatter(format, [StreamDumpStorage(sds, delay) for (delay, sds) in sdss])
    for delay in delays:
        ec.add_io(NullIO(), lead_samples=delay)

    ec.preload_count = options.preload_count
    ec.records_per_block = options.ascans_per_block
    ec.blocks_to_allocate = options.blocks_to_allocate
    ec.blocks_to_acquire = options.blocks_to_acquire

    engine.initialize(ec)
    engine.prepare()

    engine.scan_queue.append(scan)

    engine.start()
    engine.wait()
    engine.stop()

    # flush everything to disk
    for (_, sds) in sdss:
        sds.close()

    data = [numpy.fromfile(path, dtype=numpy.float64).reshape((-1, ec.galvo_output_channels)) for (_, path) in temps]
    galvos = data[0::2]
    samples = data[1::2]

    # clean up
    for (fd, path) in temps:
        os.close(fd)
        os.remove(path)

    # check timeshifted signals against each other
    n = max(delays)
    ref = galvos[0]
    for (delay, galvo, sample) in zip(delays, galvos, samples):
        assert_allclose(ref, sample)
        assert_allclose(ref[n:], galvo[n - delay:-delay or None])

def test_profiler(options, scan, acquire, process, format):
    # check for CUDA support
    get_compute_capability(0)

    # configure profiler
    (fd, path) = mkstemp()
    os.close(fd)
    os.environ['VORTEX_PROFILER_LOG'] = path

    engine = Engine()
    ec = engine.config

    ec.add_acquisition(acquire, [process])
    ec.add_processor(process, [format])
    ec.add_formatter(format, [NullEndpoint()])

    ec.preload_count = options.preload_count
    ec.records_per_block = options.ascans_per_block
    ec.blocks_to_allocate = options.blocks_to_allocate
    ec.blocks_to_acquire = options.blocks_to_acquire

    engine.initialize(ec)
    engine.prepare()

    engine.scan_queue.append(scan)
    engine.start()
    engine.wait()
    engine.stop()

    # check profiler
    profiler_log_size = os.path.getsize(path)
    os.remove(path)

    assert profiler_log_size > 0

if __name__ == '__main__':
    pytest.main()
