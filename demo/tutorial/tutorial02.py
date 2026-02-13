import numpy as np

from matplotlib import pyplot as plt

from vortex import get_console_logger as gcl, Range
from vortex.process import CUDAProcessor, CUDAProcessorConfig
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutor, StackFormatExecutorConfig
from vortex.engine import Engine, EngineConfig, SpectraStackHostTensorEndpointUInt16, StackDeviceTensorEndpointInt8 as AscanStackTensorEndpointInt8
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
# run scan forever
rsc.loop = True

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

    # create a temporary file with a varying sinusoid for tutorial purposes only
    import numpy as np
    frequency = np.random.default_rng(1234).normal(SAMPLES_PER_ASCAN / 11, SAMPLES_PER_ASCAN / 200, 10 * ASCANS_PER_BLOCK)
    spectra = 2**15 + 2**14 * np.sin(2*np.pi * frequency[:, None] * np.linspace(0, 1, SAMPLES_PER_ASCAN)[None, :])

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

# configure FFT processing on GPU
pc = CUDAProcessorConfig()
pc.samples_per_record = acquire.config.samples_per_record
pc.ascans_per_block = acquire.config.records_per_block

# apply windowing
pc.spectral_filter = np.hanning(pc.samples_per_ascan)

process = CUDAProcessor(gcl('process', LOG_LEVEL))
process.initialize(pc)

# configure standard formatting
fc = FormatPlannerConfig()
fc.segments_per_volume = BSCANS_PER_VOLUME
fc.records_per_segment = ASCANS_PER_BSCAN
fc.adapt_shape = False

format = FormatPlanner()
format.initialize(fc)

# store raw spectra in a volume
sfec = StackFormatExecutorConfig()
sfe  = StackFormatExecutor()
sfe.initialize(sfec)

# store raw spectra in a volume
spectrum_endpoint = SpectraStackHostTensorEndpointUInt16(sfe, [BSCANS_PER_VOLUME, ASCANS_PER_BSCAN, SAMPLES_PER_ASCAN], gcl('spectra', LOG_LEVEL))
# store processed A-scans in a volume
ascan_endpoint = AscanStackTensorEndpointInt8(sfe, [BSCANS_PER_VOLUME, ASCANS_PER_BSCAN, SAMPLES_PER_ASCAN], gcl('ascans', LOG_LEVEL))

# configure the engine
ec = EngineConfig()

ec.add_acquisition(acquire, [process], False)
ec.add_processor(process, [format])
ec.add_formatter(format, [spectrum_endpoint, ascan_endpoint])

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

# create a simple UI
fig, axs = plt.subplots(2, 1)
axs[0].set_title('Spectrum')
axs[0].set_ylabel('intensity (unscaled)')
axs[0].set_xlabel('sample number')
axs[1].set_title('A-scan')
axs[1].set_ylabel('power (dB)')
axs[1].set_xlabel('depth')

for ax in axs:
    ax.set_xlim(0, SAMPLES_PER_ASCAN - 1)

# initialize the UI with dummy data
spectrum_plot, = axs[0].plot(np.zeros((SAMPLES_PER_ASCAN,)))
ascan_plot, = axs[1].plot(np.zeros((SAMPLES_PER_ASCAN,)))

# connect endpoints to data display
def _update_spectrum(sample_idx, scan_idx, volume_idx):
    # retrieve the collected data
    # data is ordered by B-scan (segment), spectrum, and sample
    with spectrum_endpoint.tensor as volume:
        # combine all the B-scans (if there are multiple) and average all the spectra together
        spectrum_plot.set_ydata(volume.reshape((-1, SAMPLES_PER_ASCAN)).mean(axis=0))
spectrum_endpoint.volume_callback = _update_spectrum

def _update_ascan(sample_idx, scan_idx, volume_idx):
    # retrieve the collected data
    # data is ordered by B-scan (segment), A-scan, and sample
    with ascan_endpoint.tensor as volume:
        # combine all the B-scans (if there are multiple) and average all the A-scans together
        # NOTE: copy data from GPU to CPU with .get()
        ascan_plot.set_ydata(volume.reshape((-1, SAMPLES_PER_ASCAN)).mean(axis=0).get())
ascan_endpoint.volume_callback = _update_ascan

# start the engine and wait for the scan to complete
# NOTE: since loop is false above, only one scan is executed
engine.start()

plt.show(block=False)
plt.tight_layout()

try:
    while plt.get_fignums():
        if engine.wait_for(0.01):
            break

        for ax in axs:
            ax.relim()
            ax.autoscale_view(True, False, True)
        plt.draw()
        plt.pause(0.1)

except KeyboardInterrupt:
    pass
finally:
    engine.stop()
