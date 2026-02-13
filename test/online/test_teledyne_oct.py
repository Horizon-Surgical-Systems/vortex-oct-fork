import vortex
from vortex import get_console_logger as gcl, Range
from vortex.acquire import TeledyneAcquisition
from vortex.process import NullProcessor, NullProcessorConfig
from vortex.process import CUDAProcessor, CUDAProcessorConfig
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutor, StackFormatExecutorConfig
from vortex.engine import Engine, EngineConfig, SpectraStackHostTensorEndpointUInt16, StackDeviceTensorEndpointInt8 as AscanStackTensorEndpointInt8
from vortex.scan import RasterScan, RasterScanConfig
import vortex.process
import numpy as np

ASCANS_PER_BSCAN = 500
BSCANS_PER_VOLUME = 1
SAMPLES_PER_ASCAN = 1024
ASCANS_PER_BLOCK = 1000
input_channel = 0
clock_channel = 1
log_level = 0
enable_processing = True

rsc = RasterScanConfig()
rsc.bscans_per_volume = BSCANS_PER_VOLUME
rsc.ascans_per_bscan = ASCANS_PER_BSCAN
rsc.bscan_extent = Range(0, 0)
rsc.volume_extent = Range(0, 0)
# complete only a single volume
rsc.loop = False

scan = RasterScan()
scan.initialize(rsc)

# configure external clocking from an Teledyne card
acquire = TeledyneAcquisition(gcl('acquire', log_level))
ac = acquire.config
ac.inputs.append(vortex.acquire.teledyne.Input(input_channel))
ac.records_per_block = ASCANS_PER_BLOCK
ac.samples_per_record = SAMPLES_PER_ASCAN
ac.acquire_timeout = 1.0
ac.trigger_source = vortex.acquire.teledyne.TriggerSource.Periodic
acquire.initialize(ac)

if enable_processing:
    # configure signal processing
    pc = CUDAProcessorConfig()
    pc.samples_per_record = acquire.config.samples_per_record
    pc.ascans_per_block = acquire.config.records_per_block

    # apply windowing
    pc.spectral_filter = np.hanning(pc.samples_per_ascan)

    process = CUDAProcessor(gcl('process', log_level))
    process.initialize(pc)
else:
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

format = FormatPlanner(gcl('format', log_level))
format.initialize(fc)

# store raw spectra in a volume
sfec = StackFormatExecutorConfig()
sfe  = StackFormatExecutor()
sfe.initialize(sfec)

# store raw spectra in a volume
spectrum_endpoint = SpectraStackHostTensorEndpointUInt16(sfe, [BSCANS_PER_VOLUME, ASCANS_PER_BSCAN, SAMPLES_PER_ASCAN], gcl('spectra', log_level))
# store processed A-scans in a volume
ascan_endpoint = AscanStackTensorEndpointInt8(sfe, [BSCANS_PER_VOLUME, ASCANS_PER_BSCAN, SAMPLES_PER_ASCAN], gcl('ascans', log_level))

# configure the engine
ec = EngineConfig()

ec.add_acquisition(acquire, [process], preload=False)
ec.add_processor(process, [format])
ec.add_formatter(format, [spectrum_endpoint, ascan_endpoint])

# reasonable default parameters
ec.preload_count = 32
ec.records_per_block = 1000
ec.blocks_to_allocate = ec.preload_count * 2
ec.blocks_to_acquire = 0 # infinite acquisition

engine = Engine(gcl('engine', log_level))
engine.initialize(ec)
engine.prepare()

# load the scan
engine.scan_queue.append(scan)

# start the engine and wait for the scan to complete
# NOTE: since loop is false above, only one scan is executed
engine.start()
engine.wait()
engine.stop()

# retrieve the collected data
# data is ordered by B-scan (segment), spectrum, and sample
with spectrum_endpoint.tensor as volume:
    # combine all the B-scans (if there are multiple) and average all the spectra together
    average_spectrum = volume.reshape((-1, SAMPLES_PER_ASCAN)).mean(axis=0)

# data is ordered by B-scan (segment), A-scan, and sample
with ascan_endpoint.tensor as volume:
    # combine all the B-scans (if there are multiple) and average all the A-scans together
    # NOTE: copy data from GPU to CPU with .get()
    ascan_data = volume.reshape((-1, SAMPLES_PER_ASCAN)).mean(axis=0).get()

print(average_spectrum)
print(ascan_data)

# show the average spectrum
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, axs = plt.subplots(2, 1)
axs[0].set_title('Spectrum')
axs[0].set_ylabel('intensity (unscaled)')
axs[0].set_xlabel('sample number')
axs[1].set_title('A-scan')
axs[1].set_ylabel('power (dB)')
axs[1].set_xlabel('depth')

for ax in axs:
    ax.set_xlim(0, SAMPLES_PER_ASCAN - 1)

spectrum_plot, = axs[0].plot(average_spectrum)
ascan_plot, = axs[1].plot(ascan_data)

plt.xlabel('sample number')
plt.ylabel('intensity (unscaled)')
plt.title('Average Spectrum')
plt.tight_layout()
plt.savefig('output.png', bbox_inches='tight')
