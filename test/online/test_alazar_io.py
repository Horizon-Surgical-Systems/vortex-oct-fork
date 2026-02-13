from vortex import get_console_logger as gcl, Range
from vortex.acquire import AlazarAcquisition, AlazarConfig, alazar
from vortex.io import AlazarIO, AlazarIOConfig, DAQmxIO, DAQmxConfig, daqmx
from vortex.process import NullProcessor, NullProcessorConfig
from vortex.format import FormatPlanner, FormatPlannerConfig
from vortex.engine import Engine, EngineConfig, Block, NullEndpoint
from vortex.scan import RasterScan, RasterScanConfig

ASCANS_PER_BSCAN = 500
BSCANS_PER_VOLUME = 120
SAMPLES_PER_ASCAN = 512
ASCANS_PER_BLOCK = 2000
INPUT_CHANNEL = alazar.Channel.A
LOG_LEVEL = 1
PRELOAD_COUNT = 32
TRIGGERS_PER_SECOND = 200_000

# create a raster scan
rsc = RasterScanConfig()
rsc.bscans_per_volume = BSCANS_PER_VOLUME
rsc.ascans_per_bscan = ASCANS_PER_BSCAN
rsc.bscan_extent = Range(-3, 3)
rsc.volume_extent = Range(-1, 1)
rsc.samples_per_second = TRIGGERS_PER_SECOND
rsc.loop = True

scan = RasterScan()
scan.initialize(rsc)

# configure external clocking from an Alazar card
ac = AlazarConfig()
ac.clock = alazar.InternalClock(1_000_000_000)
ac.inputs.append(alazar.Input(INPUT_CHANNEL))
ac.records_per_block = ASCANS_PER_BLOCK
ac.samples_per_record = SAMPLES_PER_ASCAN
ac.trigger.delay_samples = 0
ac.trigger.range_millivolts = 0

acquire = AlazarAcquisition(gcl('acquire', LOG_LEVEL))
acquire.initialize(ac)

# configure no processing
pc = NullProcessorConfig()
pc.samples_per_record = acquire.config.samples_per_record
pc.ascans_per_block = acquire.config.records_per_block

process = NullProcessor()
process.initialize(pc)

# configure Alazar galvo output
aioc = AlazarIOConfig()
aioc.samples_per_block = ac.records_per_block
aioc.blocks_to_buffer = PRELOAD_COUNT

for (i, c) in enumerate(aioc.analog_output_channels):
    c.logical_units_per_physical_unit = 15 / 10
    c.stream = Block.StreamIndex.GalvoTarget
    c.channel = i

aio = AlazarIO(gcl('dac', LOG_LEVEL))
aio.initialize(aioc)

# configure NI galvo output for reference
nioc = DAQmxConfig()
nioc.name = 'ni'
nioc.samples_per_block = ac.records_per_block
nioc.samples_per_second = TRIGGERS_PER_SECOND
nioc.blocks_to_buffer = PRELOAD_COUNT

for i in range(2):
    nioc.channels.append(daqmx.AnalogVoltageOutput(f'Dev1/ao{i}', 15 / 10, Block.StreamIndex.GalvoTarget, i))

nio = DAQmxIO(gcl(nioc.name, LOG_LEVEL))
nio.initialize(nioc)

# configure standard formatting (not relevant)
fc = FormatPlannerConfig()
fc.segments_per_volume = BSCANS_PER_VOLUME
fc.records_per_segment = ASCANS_PER_BSCAN
fc.adapt_shape = False

format = FormatPlanner(gcl('format', LOG_LEVEL))
format.initialize(fc)

# configure the engine
ec = EngineConfig()

ec.add_acquisition(acquire, [process])
ec.add_processor(process, [format])
ec.add_formatter(format, [NullEndpoint()])
ec.add_io(aio)
ec.add_io(nio)

# reasonable default parameters
ec.preload_count = PRELOAD_COUNT
ec.records_per_block = ASCANS_PER_BLOCK
ec.blocks_to_allocate = ec.preload_count * 2
ec.blocks_to_acquire = 0 # inifinite acquisition

engine = Engine(gcl('engine', LOG_LEVEL))
engine.initialize(ec)
engine.prepare()

# load the scan
engine.scan_queue.append(scan)

# start the engine and wait for the scan to complete
engine.start()
try:
    while True:
        engine.wait_for(0.1)
except KeyboardInterrupt:
    pass
finally:
    engine.stop()
