from itertools import combinations

import pytest
import numpy as np
from numpy.testing import assert_allclose

from vortex import get_console_logger as get_logger
from vortex.format import FormatPlanner, StackFormatExecutor
from vortex.engine import source, find_rising_edges, compute_resampling, dispersion_phasor, StackHostTensorEndpointInt8, Engine, EngineConfig
from vortex.scan import RasterScan

try:
    from vortex.acquire import AlazarConfig, AlazarAcquisition, alazar
    from vortex.engine import acquire_alazar_clock
except ImportError:
    pytest.skip('Alazar support is not avilable')

import vortex.process
PROCESSOR_CLASSES = {}
for name in ['CPU', 'CUDA']:
    try:
        PROCESSOR_CLASSES[name] = getattr(vortex.process, f'{name}Processor')
    except AttributeError:
        pass
if not PROCESSOR_CLASSES:
    raise pytest.skip('no OCT processors available', allow_module_level=True)

@pytest.fixture(scope='module')
def names():
    return list(PROCESSOR_CLASSES.keys())

@pytest.fixture(scope='module')
def options():
    class cfg:
        internal_clock = True
        clock_samples_per_second = 800_000_000

        input_channel = alazar.Channel.A
        clock_channel = alazar.Channel.B

        ascans_per_block = 1000
        blocks_to_allocate = 128
        preload_count = 32

        swept_source = source.Axsun100k
        dispersion = (28e-6, 0)

        samples_per_ascan = 1024
        ascans_per_bscan = 500
        bscans_per_volume = 20

        process_slots = 2

        log_level = 1

    return cfg

@pytest.fixture(scope='module')
def scan(options):
    scan = RasterScan()
    sc = scan.config

    sc.bscans_per_volume = options.bscans_per_volume
    sc.ascans_per_bscan = options.ascans_per_bscan
    sc.loop = False

    scan.initialize(sc)
    return scan

@pytest.fixture(scope='module')
def acquire_cfg(options):
    ac = AlazarConfig()
    if options.internal_clock:
        ac.clock = alazar.InternalClock(options.clock_samples_per_second)
    else:
        ac.clock = alazar.ExternalClock()

    ac.inputs.append(alazar.Input(options.input_channel))

    ac.records_per_block = options.ascans_per_block

    return ac

@pytest.fixture(scope='module')
def board(acquire_cfg):
    return alazar.Board(acquire_cfg.device.system_index, acquire_cfg.device.board_index)

@pytest.fixture(scope='module')
def resampling(options, acquire_cfg, board):
    if options.internal_clock:
        (clock_samples_per_second, clock) = acquire_alazar_clock(options.swept_source, acquire_cfg, options.clock_channel, get_logger('acquire', options.log_level))
        options.swept_source.clock_edges_seconds = find_rising_edges(clock, clock_samples_per_second, len(options.swept_source.clock_edges_seconds))
        resampling = compute_resampling(options.swept_source, acquire_cfg.samples_per_second, options.samples_per_ascan)

        # acquire enough samples to obtain the required ones
        acquire_cfg.samples_per_record = board.info.smallest_aligned_samples_per_record(resampling.max())
    else:
        resampling = []
        acquire_cfg.samples_per_record = board.info.smallest_aligned_samples_per_record(options.swept_source.clock_rising_edges_per_trigger)

    return resampling

@pytest.fixture(scope='module')
def acquire(options, acquire_cfg, board, resampling):
    if options.internal_clock:
        acquire_cfg.samples_per_record = board.info.smallest_aligned_samples_per_record(resampling.max())
    else:
        acquire_cfg.samples_per_record = board.info.smallest_aligned_samples_per_record(options.swept_source.clock_rising_edges_per_trigger)

    acquire = AlazarAcquisition(get_logger('acquire', options.log_level))
    acquire.initialize(acquire_cfg)

    return acquire

@pytest.fixture(scope='module')
def processors(options, acquire_cfg, resampling):
    processors = []
    for (name, Processor) in PROCESSOR_CLASSES.items():
        process = Processor(get_logger(f'process-{name}', options.log_level))
        pc = process.config

        # match acquisition settings
        pc.samples_per_record = acquire_cfg.samples_per_record
        pc.ascans_per_block = acquire_cfg.records_per_block

        pc.slots = options.process_slots

        # reasmpling
        pc.resampling_samples = resampling

        # spectral filter with dispersion correction
        window = np.hanning(pc.samples_per_ascan)
        phasor = dispersion_phasor(len(window), options.dispersion)
        pc.spectral_filter = window * phasor

        # DC subtraction per block
        pc.average_window = 2 * pc.ascans_per_block

        process.initialize(pc)
        processors.append(process)

    return processors

@pytest.fixture(scope='module')
def formats(options, names):
    formats = []
    for name in names:
        format = FormatPlanner(get_logger(f'format-{name}', options.log_level))
        fc = format.config

        fc.segments_per_volume = options.bscans_per_volume
        fc.records_per_segment = options.ascans_per_bscan

        format.initialize(fc)
        formats.append(format)

    return formats

@pytest.fixture(scope='module')
def endpoints(options, names, processors):
    endpoints = []
    for (name, processor) in zip(names, processors):
        endpoints.append(StackHostTensorEndpointInt8(StackFormatExecutor(), (options.bscans_per_volume, options.ascans_per_bscan, processor.config.samples_per_ascan), log=get_logger(f'endpoint-{name}', options.log_level)))

    return endpoints

@pytest.fixture(scope='module')
def volumes(options, scan, acquire, processors, formats, endpoints):
    ec = EngineConfig()
    ec.add_acquisition(acquire, processors)
    for (process, format) in zip(processors, formats):
        ec.add_processor(process, [format])
    for (format, endpoint) in zip(formats, endpoints):
        ec.add_formatter(format, [endpoint])

    ec.preload_count = options.preload_count
    ec.records_per_block = options.ascans_per_block
    ec.blocks_to_allocate = options.blocks_to_allocate
    ec.blocks_to_acquire = 0

    engine = Engine(get_logger('engine', options.log_level))

    engine.initialize(ec)
    engine.prepare()

    engine.scan_queue.append(scan)

    engine.start()
    engine.wait()
    engine.stop()

    volumes = []
    for endpoint in endpoints:
        with endpoint.tensor as volume:
            try:
                endpoint.stream.synchronize()
            except AttributeError:
                pass
            volumes.append(volume.copy())

    return volumes

def test_acquire(volumes):
    pass

@pytest.mark.parametrize(['nameA', 'nameB'], combinations(PROCESSOR_CLASSES.keys(), 2))
def test_volumes(request, nameA, nameB, names, volumes):
    volumeA = volumes[names.index(nameA)]
    volumeB = volumes[names.index(nameB)]

    if request.config.option.plot:
        from matplotlib import pyplot as plt, cm

        for i in range(volumes[0].shape[0]):
            fig, axs = plt.subplots(1, 3)

            for (name, volume, ax) in zip([nameA, nameB], [volumeA, volumeB], axs):
                ax.imshow(volume[i].T, cmap=cm.turbo, interpolation='nearest', aspect='auto', vmin=0, vmax=40)
                ax.set_title(name)

            diff = volumeA[i] - volumeB[i]
            dp = axs[2].imshow(diff.T, cmap=cm.turbo, interpolation='nearest', aspect='auto')
            axs[2].set_title(f'{nameA} vs {nameB}: {abs(diff).max()} / {np.count_nonzero(diff) / diff.size:.1g}')
            plt.colorbar(dp, ax=axs[2])

        plt.show()

    assert_allclose(volumeA, volumeB, atol=1)

if __name__ == '__main__':
    pytest.main()
