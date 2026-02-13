from math import pi, sin, cos

import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

from vortex.scan import RadialScanConfig, FreeformScan

from vortex_tools.scan import plot_annotated_waveforms_time, plot_annotated_waveforms_space

def run():
    radial = RadialScanConfig()
    pattern = radial.to_segments()

    # append extra channels to radial scan
    thetas = numpy.linspace(0, 2*pi, len(pattern))
    for (theta, segment) in zip(thetas, pattern):
        external = [cos(theta), sin(theta)] + numpy.ones_like(segment.position)
        segment.position = numpy.column_stack((segment.position, external))

    scan = FreeformScan()
    cfg = scan.config

    # increase channel and limits count
    cfg.channels_per_sample = pattern[0].position.shape[1]
    cfg.limits = [cfg.limits[0]] * cfg.channels_per_sample

    cfg.pattern = pattern
    cfg.loop = True

    scan.initialize(cfg)

    fig, _ = plot_annotated_waveforms_time(cfg.sampling_interval, scan.scan_buffer(), scan.scan_markers())
    fig.suptitle('External Channel Demo')

    fig, (ax0, ax1) = pyplot.subplots(1, 2, constrained_layout=True)
    plot_annotated_waveforms_space(scan.scan_buffer()[:, 0:2], scan.scan_markers(), inactive_marker='k', axes=ax0)
    ax0.set_title('Galvo Scan')
    plot_annotated_waveforms_space(scan.scan_buffer()[:, 2:4], scan.scan_markers(), inactive_marker='k', axes=ax1)
    ax1.set_title('External Scan')

    fig.suptitle('External Channel Demo')

if __name__ == '__main__':
    run()

    pyplot.show()
