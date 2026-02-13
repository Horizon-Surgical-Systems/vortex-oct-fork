from vortex import get_console_logger as gcl, Range
from vortex.process import NullProcessor, NullProcessorConfig
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutor, StackFormatExecutorConfig
from vortex.engine import Engine, EngineConfig, SpectraStackHostTensorEndpointUInt16
from vortex.scan import RasterScan, RasterScanConfig

# SOURCE = 'null'
SOURCE = 'file'
# SOURCE = 'alazar'
# SOURCE = 'teledyne'

ASCANS_PER_BSCAN = 500
BSCANS_PER_VOLUME = 1
SAMPLES_PER_ASCAN = 1024
ASCANS_PER_BLOCK = 100
INPUT_CHANNEL = 0
LOG_LEVEL = 1

# create a repeated A-scan
rsc = RasterScanConfig()
rsc.bscans_per_volume = BSCANS_PER_VOLUME
rsc.ascans_per_bscan = ASCANS_PER_BSCAN
rsc.bscan_extent = Range(0, 0)
rsc.volume_extent = Range(0, 0)
# complete only a single volume
rsc.loop = False

scan = RasterScan()
scan.initialize(rsc)

if SOURCE == 'null':

    from vortex.acquire import NullAcquisitionConfig, NullAcquisition

    # produce infinitely blocks of uninitialized memory
    ac = NullAcquisitionConfig()
    ac.records_per_block = ASCANS_PER_BLOCK
    ac.samples_per_record = SAMPLES_PER_ASCAN

    acquire = NullAcquisition()
    acquire.initialize(ac)

elif SOURCE == 'file':

    # create a temporary file with a pure sinusoid for tutorial purposes only
    import numpy as np
    spectrum = 2**15 + 2**14 * np.sin(2*np.pi * (SAMPLES_PER_ASCAN / 11) * np.linspace(0, 1, SAMPLES_PER_ASCAN))
    spectra = np.repeat(spectrum[None, ...], ASCANS_PER_BLOCK, axis=0)

    import os
    from tempfile import mkstemp
    (fd, test_file_path) = mkstemp()
    # NOTE: the Python bindings are restricted to the uint16 data type
    open(test_file_path, 'wb').write(spectra.astype(np.uint16).tobytes())
    os.close(fd)

    # NOTE: intentionally leave the file to keep the tutorial simple
    # os.remove(test_file_path)

    from vortex.acquire import FileAcquisitionConfig, FileAcquisition

    # produce blocks ready from a file
    ac = FileAcquisitionConfig()
    ac.path = test_file_path
    ac.records_per_block = ASCANS_PER_BLOCK
    ac.samples_per_record = SAMPLES_PER_ASCAN
    ac.loop = True # repeat the file indefinitely

    acquire = FileAcquisition(gcl('acquire', LOG_LEVEL))
    acquire.initialize(ac)

elif SOURCE == 'alazar':

    from vortex.acquire import AlazarAcquisition, AlazarConfig, alazar

    # map channel index to enumerated value
    # NOTE: only to keep the tutorial simple
    channel = list(alazar.Channel.__members__.values())[INPUT_CHANNEL]

    # configure external clocking from an Alazar card
    ac = AlazarConfig()
    ac.clock = alazar.ExternalClock()
    ac.inputs.append(alazar.Input(channel))
    ac.records_per_block = ASCANS_PER_BLOCK
    ac.samples_per_record = SAMPLES_PER_ASCAN

    acquire = AlazarAcquisition(gcl('acquire', LOG_LEVEL))
    acquire.initialize(ac)

elif SOURCE == 'teledyne':

    from vortex.acquire import TeledyneAcquisition, TeledyneConfig, teledyne

    # configure external clocking from an Teledyne card
    ac = TeledyneConfig()
    ac.inputs.append(teledyne.Input(INPUT_CHANNEL))
    ac.records_per_block = ASCANS_PER_BLOCK
    ac.samples_per_record = SAMPLES_PER_ASCAN
    ac.acquire_timeout = 1.0
    ac.trigger_source = teledyne.TriggerSource.Periodic
    ac.trigger_sync_passthrough = False

    acquire = TeledyneAcquisition(gcl('acquire', LOG_LEVEL))
    acquire.initialize(ac)

else:

    raise RuntimeError('invalid source configured')

# configure no processing
pc = NullProcessorConfig()
pc.samples_per_record = acquire.config.samples_per_record
pc.ascans_per_block = acquire.config.records_per_block

process = NullProcessor()
process.initialize(pc)

# configure standard formatting
fc = FormatPlannerConfig()
fc.segments_per_volume = BSCANS_PER_VOLUME
fc.records_per_segment = ASCANS_PER_BSCAN
fc.adapt_shape = False

format = FormatPlanner(gcl('format', LOG_LEVEL))
format.initialize(fc)

# store raw spectra in a volume
sfec = StackFormatExecutorConfig()
sfe  = StackFormatExecutor()
sfe.initialize(sfec)

endpoint = SpectraStackHostTensorEndpointUInt16(sfe, [BSCANS_PER_VOLUME, ASCANS_PER_BSCAN, SAMPLES_PER_ASCAN], gcl('endpoint', LOG_LEVEL))

# configure the engine
ec = EngineConfig()

ec.add_acquisition(acquire, [process], False)
ec.add_processor(process, [format])
ec.add_formatter(format, [endpoint])

# reasonable default parameters
ec.preload_count = 32
ec.records_per_block = ASCANS_PER_BLOCK
ec.blocks_to_allocate = ec.preload_count * 2
ec.blocks_to_acquire = 0 # inifinite acquisition

engine = Engine(gcl('engine', LOG_LEVEL))
engine.initialize(ec)
engine.prepare()

# load the scan
engine.scan_queue.append(scan)

# start the engine and wait for the scan to complete
# NOTE: since loop is false above, only one scan is executed
engine.start()
try:
    engine.wait()
finally:
    engine.stop()

# retrieve the collected data
# data is ordered by B-scan (segment), A-scan, and sample
with endpoint.tensor as volume:
    # combine all the B-scans (if there are multiple) and average all the spectra together
    average_spectrum = volume.reshape((-1, SAMPLES_PER_ASCAN)).mean(axis=0)

# show the average spectrum
from matplotlib import pyplot as plt
plt.plot(average_spectrum)

plt.xlabel('sample number')
plt.ylabel('intensity (unscaled)')
plt.title('Average Spectrum')
plt.show()
