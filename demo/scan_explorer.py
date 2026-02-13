import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

from vortex import Range
from vortex.marker import Flags
from vortex.scan import RasterScan, RepeatedRasterScan, RadialScan, RepeatedRadialScan, FreeformScan, SequentialPattern, RasterScanConfig, RadialScanConfig

from vortex_tools.scan import plot_annotated_waveforms_time, plot_annotated_waveforms_space

def freeform():
    t = numpy.linspace(0, 1, 500)
    t2 = numpy.linspace(0, 5, 2500)
    waypoints = [
        numpy.column_stack((t+0.5, 0.3*t**2 + 0.5)),
        numpy.column_stack((0.01*numpy.cos(50*t), t/2)),
        numpy.column_stack((t/3+0.2, -t/2)),
        numpy.column_stack((0.1*numpy.sin(2*t2 + 0.2) - 0.5, 0.2*numpy.cos(3*t2) - 0.5))
    ]

    pattern = SequentialPattern().to_pattern(waypoints)

    scan = FreeformScan()
    cfg = scan.config
    cfg.pattern = pattern
    cfg.loop = True
    scan.initialize(cfg)

    show(scan, 'Freeform')

def standard(scan, name=None):
    name = name or type(scan)

    cfg = scan.config
    cfg.loop = True
    scan.initialize(cfg)

    show(scan, name)

def radial_nonnegative():
    scan = RadialScan()

    cfg = scan.config
    cfg.bscan_extent = Range(2, 5)
    cfg.volume_extent = Range(1, 2)
    cfg.loop = True
    scan.initialize(cfg)

    show(scan, 'Radial - Non-negative')

def raster_aim(name=None):
    raster = RasterScanConfig()
    raster.flags = Flags(0x1) # optional

    aiming = RadialScanConfig()
    aiming.set_aiming()
    aiming.offset = (5, 0)
    aiming.flags = Flags(0x2) # optional

    n = raster.segments_per_volume
    pattern = raster.to_segments()[:n//2] + aiming.to_segments() + raster.to_segments()[n//2:]

    scan = FreeformScan()
    cfg = scan.config
    cfg.pattern = pattern
    cfg.loop = True
    scan.initialize(cfg)

    show(scan, 'Raster + Aim')

def show(scan, name):
    cfg = scan.config

    fig, _ = plot_annotated_waveforms_time(cfg.sampling_interval, scan.scan_buffer(), scan.scan_markers())
    fig.suptitle(name)
    fig, _ = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers())
    fig.suptitle(name)

if __name__ == '__main__':
    standard(RasterScan(), 'Raster')
    standard(RepeatedRasterScan(), 'Repeated Raster')
    standard(RadialScan(), 'Radial')
    standard(RepeatedRadialScan(), 'Repeated Radial')
    freeform()
    radial_nonnegative()
    raster_aim()

    pyplot.show()
