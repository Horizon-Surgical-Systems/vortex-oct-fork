import sys
from time import time
from math import pi

import cupy
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

# configure the root logger to accept all records
import logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.NOTSET)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(name)s] %(filename)s:%(lineno)d\t%(levelname)s:\t%(message)s')

# set up colored logging to console
from rainbow_logging_handler import RainbowLoggingHandler
console_handler = RainbowLoggingHandler(sys.stderr)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

from vortex import Range, get_python_logger
from vortex.marker import Flags
from vortex.scan import RasterScanConfig, RasterScan, RadialScanConfig, RadialScan
from vortex.engine import source
from vortex.simple import SimpleEngine, SimpleEngineConfig

swept_source = source.Axsun100k

raster_sc = RasterScanConfig()
raster_sc.bscans_per_volume = 500
raster_sc.ascans_per_bscan = 500
#raster_sc.offset = (1, 1)
raster_sc.bscan_extent = Range(-5, 5)
raster_sc.volume_extent = Range(-5, 5)
raster_sc.bidirectional_segments = True
raster_sc.bidirectional_volumes = True
raster_sc.samples_per_second = swept_source.triggers_per_second
raster_sc.loop = True
raster_sc.flags = Flags(0x1)

raster_s = RasterScan()
raster_s.initialize(raster_sc)


radial_sc = RadialScanConfig()
radial_sc.bscans_per_volume = 500
radial_sc.ascans_per_bscan = 500
#radial_sc.offset = (1, 1)
radial_sc.bscan_extent = Range(-5, 5)
#radial_sc.volume_extent = Range(-5, 5)
radial_sc.set_half_evenly_spaced(radial_sc.bscans_per_volume)
radial_sc.bidirectional_segments = True
radial_sc.bidirectional_volumes = True
radial_sc.samples_per_second = swept_source.triggers_per_second
radial_sc.loop = True
radial_sc.flags = Flags(0x2)

radial_s = RadialScan()
radial_s.initialize(radial_sc)


sec = SimpleEngineConfig()
sec.internal_clock = True
sec.ascans_per_bscan = radial_sc.ascans_per_bscan
sec.bscans_per_volume = radial_sc.bscans_per_volume
sec.blocks_to_acquire = 0
sec.preload_count = 128
sec.blocks_to_allocate = 512
sec.swept_source = swept_source
sec.dispersion = (-7e-5, 0)
sec.galvo_delay = 95e-6
# uncomment this line to save data to disk
# sec.save_path = r'scan.broct'

get_python_logger('acquire')
get_python_logger('process')
get_python_logger('input')
get_python_logger('output')
get_python_logger('format')
get_python_logger('storage')
get_python_logger('dump')
get_python_logger('engine')

engine = SimpleEngine('vortex.')
engine.initialize(sec)

engine.append_scan(raster_s)

done = False
def handle_close(e):
    global done
    done = True

def handle_keypress(e):
    if e.key == 'l':
        print('rotate left')
        raster_sc.angle += pi / 6
        raster_s.change(raster_sc)
        radial_sc.angle += pi / 6
        radial_s.change(radial_sc)

    if e.key == 'r':
        print('rotate right')
        raster_sc.angle -= pi / 6
        raster_s.change(raster_sc)
        radial_sc.angle -= pi / 6
        radial_s.change(radial_sc)

    if e.key == '1':
        print('swtich to raster scan')
        engine.interrupt_scan(raster_s)

    if e.key == '2':
        print('switch to radial scan')
        engine.interrupt_scan(radial_s)

dirty_bscan_idxs = []
def update(radial, bscans):
    dirty_bscan_idxs.extend(bscans)

mip = numpy.zeros(engine.volume().shape[:2], dtype=numpy.int8)
bscan = numpy.zeros(engine.volume().shape[1:], dtype=numpy.int8)

fig, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
fig.canvas.mpl_connect('key_press_event', handle_keypress)
fig.canvas.mpl_connect('close_event', handle_close)

a = radial_sc.volume_extent.length / radial_sc.bscans_per_volume / (radial_sc.bscan_extent.length / radial_sc.ascans_per_bscan)
mip_im = ax1.imshow(mip, interpolation='nearest', vmin=0, vmax=60) #, aspect=a)
bscan_im = ax2.imshow(bscan.T, interpolation='nearest', vmin=0, vmax=60)

ax1.set_title('MIP')
ax2.set_title('B-scan')
fig.suptitle('Keys: 1 = raster, 2 = radial, L/R = rotate scan')

engine.start(update)

try:
    mark = time()
    n = 0

    while not done:
        if engine.wait_for(0.01):
            break

        if dirty_bscan_idxs:
            fresh_bscan_idx = dirty_bscan_idxs[-1]

            update_bscan_idxs = list(set(dirty_bscan_idxs))
            update_bscan_idxs.sort()

            dirty_bscan_idxs.clear()
            with engine.volume() as vol:
                mip = cupy.asnumpy(cupy.max(vol, axis=2))
                bscan = cupy.asnumpy(vol[fresh_bscan_idx, ...])

            mip_im.set_array(mip)
            bscan_im.set_array(bscan.T)
            n += 1

        pyplot.pause(0.01)
except KeyboardInterrupt:
    pass
finally:
    engine.stop()

print('FPS:', n / (time() - mark))
