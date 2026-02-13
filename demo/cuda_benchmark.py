from time import time, sleep

import numpy
import cupy

from vortex import Range, get_console_logger as get_logger
from vortex.scan import RasterScan, warp
from vortex.engine import EngineConfig, Engine, dispersion_phasor, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint

from vortex.acquire import NullAcquisition
from vortex.process import CUDAProcessor
from vortex.format import FormatPlanner, StackFormatExecutor, StackFormatExecutorConfig, SimpleSlice

class cfg:
    samples_per_ascan = 1024
    ascans_per_bscan = 512
    bscans_per_volume = 512

    ascans_per_block = 512      # number of A-scans in a memory buffer
    blocks_to_allocate = 32     # number of memory buffers to allocate
    blocks_to_acquire = 100000  # number of memory buffers to acquire before exiting
                                # (0 means infinite)

    preload_count = 8           # number of memory buffers to post to Alazar and NI cards before
                                # the acquistion begins (determines scan change latency)

    process_slots = 2

    log_level = 1               # log verbosity (higher number means less verbose)

class OCTEngine:
    def __init__(self, cfg):
        #
        # scan
        #

        self._raster_scan = RasterScan()
        rsc = self._raster_scan.config
        rsc.bscans_per_volume = cfg.bscans_per_volume
        rsc.ascans_per_bscan = cfg.ascans_per_bscan

        # rsc.offset = (1, 1)               # units of volts
        # rsc.angle = 0                     # units of radians
        rsc.bscan_extent = Range(-0, 0)     # units of volts
        rsc.volume_extent = Range(-0, 0)    # units of volts

        rsc.bidirectional_segments = False
        rsc.bidirectional_volumes = False

        rsc.samples_per_second = 100000
        rsc.loop = True

        rsc.warp = warp.Angular()
        rsc.warp.factor = 2

        self._raster_scan.initialize(rsc)
        self._raster_scan.prepare()

        #
        # acquisition
        #

        self._acquire = NullAcquisition()
        ac = self._acquire.config

        ac.records_per_block = cfg.ascans_per_block
        ac.samples_per_record = cfg.samples_per_ascan
        ac.channels_per_sample = 1

        self._acquire.initialize(ac)

        #
        # OCT processing setup
        #

        self._process = CUDAProcessor(get_logger('process', cfg.log_level))
        pc = self._process.config

        # match acquisition settings
        pc.samples_per_record = ac.samples_per_record
        pc.ascans_per_block = ac.records_per_block

        pc.slots = cfg.process_slots

        # reasmpling
        pc.resampling_samples = numpy.arange(0, pc.samples_per_record)

        # spectral filter
        window = numpy.hamming(pc.samples_per_ascan)
        phasor = dispersion_phasor(len(window), (2.8e-5, 0))
        pc.spectral_filter = window * phasor

        # DC subtraction per block
        pc.average_window = 2 * pc.ascans_per_block

        self._process.initialize(pc)

        #
        # output setup
        #

        # format planners
        self._stack_format = FormatPlanner(get_logger('format', cfg.log_level))

        fc = self._stack_format.config
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        self._stack_format.initialize(fc)

        # format executors
        samples_to_save = pc.samples_per_ascan // 2

        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(samples_to_save)

        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        stack_tensor_endpoint = StackDeviceTensorEndpoint(cfe, (cfg.bscans_per_volume, cfg.ascans_per_bscan, samples_to_save), get_logger('cube', cfg.log_level))
        self._stack_tensor_endpoint = stack_tensor_endpoint

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [self._stack_format])
        ec.add_formatter(self._stack_format, [stack_tensor_endpoint])

        ec.preload_count = cfg.preload_count
        ec.records_per_block = cfg.ascans_per_block
        ec.blocks_to_allocate = cfg.blocks_to_allocate
        ec.blocks_to_acquire = cfg.blocks_to_acquire

        self._engine = Engine(get_logger('engine', cfg.log_level))

        self._engine.initialize(ec)
        self._engine.prepare()

    def run(self):
        self._engine.scan_queue.append(self._raster_scan)

        t1 = time()
        self._engine.start()
        self._engine.wait()
        t2 = time()

        self._engine.stop()

        sleep(1)
        dt = t2 - t1
        total_ascans = cfg.blocks_to_acquire * cfg.ascans_per_block

        with cupy.cuda.Device(self._process.config.device) as device:
            print(f'using GPU {device.pci_bus_id} ({device.id})')
        print(f'ascan length = {cfg.samples_per_ascan}, block size = {cfg.ascans_per_block}, process slots = {cfg.process_slots}')
        print(f'input: {self._process.config.input_shape}, output: {self._process.config.output_shape}')
        print(f'time = {dt} s   ascans = {total_ascans}   ascans/sec = {total_ascans / dt:.0f}')

if __name__ == '__main__':
    engine = OCTEngine(cfg)
    engine.run()
