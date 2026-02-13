from datetime import timedelta


from vortex import Range, get_console_logger as get_logger
from vortex.scan import RasterScan, warp
from vortex.engine import EngineConfig, Engine, Block, SpectraStackEndpoint, SpectraHDF5StackEndpoint, BufferStrategy

from vortex.acquire import AlazarAcquisition, alazar
from vortex.process import NullProcessor
from vortex.io import DAQmxIO, AnalogVoltageOutput
from vortex.format import FormatPlanner
from vortex.storage import SimpleStackUInt16, SimpleStackHeader, HDF5StackUInt16, HDF5StackHeader

class cfg:
    samples_per_ascan = 128 * 4000
    ascans_per_bscan = 1000
    bscans_per_volume = 10

    ascans_per_block = 8        # number of A-scans in a memory buffer
    blocks_to_allocate = 2500   # number of memory buffers to allocate
    blocks_to_acquire = 0       # number of memory buffers to acquire before exiting
                                # (0 means infinite)
    volumes_to_acquire = 1      # number of volumes to acquire (0 means infinite)

    preload_count = 64          # number of memory buffers to post to Alazar and NI cards before
                                # the acquistion begins (determines scan change latency)

    clock_samples_per_second = 1.8e9
    input_channels = [alazar.Channel.A, alazar.Channel.B]

    galvo_delay_seconds = 0     # number of seconds to delay the start of active lines to
                                # compensate for galvo lag (only when putting data in a cube)

    triggers_per_second = 15e3

    save_path = 'acquire_to_disk.v73.mat'

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

        rsc.samples_per_second = round(cfg.triggers_per_second)
        rsc.loop = (cfg.volumes_to_acquire == 0)

        rsc.warp = warp.Angular()
        rsc.warp.factor = 2

        self._raster_scan.initialize(rsc)

        #
        # acquisition
        #

        self._acquire = AlazarAcquisition(get_logger('acquire', cfg.log_level))
        ac = self._acquire.config
        ac.clock = alazar.InternalClock(int(cfg.clock_samples_per_second))

        for channel in cfg.input_channels:
            ac.inputs.append(alazar.Input(channel))
        ac.options.append(alazar.AuxIOTriggerOut())

        ac.records_per_block = cfg.ascans_per_block
        ac.samples_per_record = cfg.samples_per_ascan

        ac.acquire_timeout = timedelta(seconds=10)

        self._acquire.initialize(ac)

        #
        # OCT processing setup
        #

        self._process = NullProcessor() #get_logger('process', cfg.log_level))
        pc = self._process.config

        # match acquisition settings
        pc.samples_per_record = ac.samples_per_record
        pc.ascans_per_block = ac.records_per_block

        self._process.initialize(pc)

        #
        # galvo control
        #

        self._io_out = DAQmxIO(get_logger('output', cfg.log_level))

        # output
        ioc_out = self._io_out.config
        ioc_out.samples_per_block = ac.records_per_block
        ioc_out.samples_per_second = round(cfg.triggers_per_second)
        ioc_out.blocks_to_buffer = cfg.preload_count
        ioc_out.name = 'output'

        stream = Block.StreamIndex.GalvoTarget
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao0', 15 / 10, stream, 0))
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao1', 15 / 10, stream, 1))

        self._io_out.initialize(ioc_out)

        #
        # output setup
        #

        # format planners
        self._stack_format = FormatPlanner(get_logger('format', cfg.log_level))

        fc = self._stack_format.config
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = False

        self._stack_format.initialize(fc)

        if 0:
            # format executors
            bs = SimpleStackUInt16(get_logger('save', cfg.log_level))

            bsc = bs.config
            bsc.path = cfg.save_path
            bsc.shape = (fc.segments_per_volume, fc.records_per_segment, ac.samples_per_record, ac.channels_per_sample)
            bsc.buffering = False
            bsc.header = SimpleStackHeader.NumPy

            bs.open(bsc)

            # spectra_stack_endpoint = SpectraStackEndpoint(StackFormatExecutor(), bs, BufferStrategy.Block, get_logger('endpoint', cfg.log_level))
            spectra_stack_endpoint = SpectraStackEndpoint(bs, get_logger('endpoint', cfg.log_level))

        if 1:
            # format executors
            bs = HDF5StackUInt16(get_logger('save', cfg.log_level))

            bsc = bs.config
            bsc.path = cfg.save_path
            bsc.shape = (fc.segments_per_volume, fc.records_per_segment, ac.samples_per_record, ac.channels_per_sample)
            bsc.header = HDF5StackHeader.MATLAB

            bs.open(bsc)

            # spectra_stack_endpoint = SpectraHDF5StackEndpoint(StackFormatExecutor(), bs, BufferStrategy.Block, get_logger('endpoint', cfg.log_level))
            spectra_stack_endpoint = SpectraHDF5StackEndpoint(bs, get_logger('endpoint', cfg.log_level))

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [self._stack_format])
        ec.add_formatter(self._stack_format, [spectra_stack_endpoint])
        ec.add_io(self._io_out, lead_samples=round(cfg.galvo_delay_seconds * ioc_out.samples_per_second))

        ec.preload_count = cfg.preload_count
        ec.records_per_block = cfg.ascans_per_block
        ec.blocks_to_allocate = cfg.blocks_to_allocate
        ec.blocks_to_acquire = cfg.blocks_to_acquire

        ec.galvo_input_channels = 0
        ec.galvo_output_channels = len(self._io_out.config.channels)

        self._engine = Engine(get_logger('engine', cfg.log_level))

        self._engine.initialize(ec)
        self._engine.prepare()

    def run(self):
        for _ in range(max([1, cfg.volumes_to_acquire])):
            self._engine.scan_queue.append(self._raster_scan)

        self._engine.start()

        try:
            while True:
                if self._engine.wait_for(0.5):
                    print()
                    break
                status = self._engine.status()
                print(f'dispatch = {status.dispatched_blocks:5d}  utilization = {status.block_utilization*100:3.1f}%', end='\r')

        except KeyboardInterrupt:
            pass

        self._engine.stop()

if __name__ == '__main__':
    engine = OCTEngine(cfg)
    engine.run()
