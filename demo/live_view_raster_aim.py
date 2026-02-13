import sys
import os
from math import pi

import numpy

from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as get_logger
from vortex.marker import Flags, VolumeBoundary, ScanBoundary
from vortex.scan import RasterScanConfig, RadialScanConfig, FreeformScanConfig, FreeformScan
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint

from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget, CrossSectionImageWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

class OCTEngine(BaseEngine):
    def __init__(self, cfg: StandardEngineParams):
        super().__init__(cfg)

        #
        # scan
        #

        rsc = RasterScanConfig()
        rsc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        rsc.volume_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        rsc.bscans_per_volume = cfg.bscans_per_volume
        rsc.ascans_per_bscan = cfg.ascans_per_bscan
        rsc.bidirectional_segments = cfg.bidirectional
        rsc.bidirectional_volumes = cfg.bidirectional
        rsc.flags = Flags(0x1)

        raster_segments = rsc.to_segments()

        asc = RadialScanConfig()
        asc.ascans_per_bscan = cfg.ascans_per_bscan
        asc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        asc.set_aiming()
        asc.flags = Flags(0x2)

        aiming_segments = asc.to_segments()

        pattern = []
        idx = numpy.linspace(0, len(raster_segments), 10, dtype=int)
        for (i, (a, b)) in enumerate(zip(idx[:-1], idx[1:])):
            if i > 0:
                markers = raster_segments[a].markers
                markers.insert(0, VolumeBoundary(0, 0, False))
                markers.insert(0, ScanBoundary(0, 0))

            pattern += raster_segments[a:b]
            pattern += aiming_segments

        ffsc = FreeformScanConfig()
        ffsc.pattern = pattern
        ffsc.loop = True

        scan = FreeformScan()
        scan.initialize(ffsc)
        self._scan = scan

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.adapt_shape = False

        fc.mask = rsc.flags
        fc.segments_per_volume = rsc.bscans_per_volume
        fc.records_per_segment = rsc.ascans_per_bscan
        raster_format = FormatPlanner(get_logger('format-raster', cfg.log_level))
        raster_format.initialize(fc)
        self._raster_format = raster_format

        fc.mask = asc.flags
        fc.segments_per_volume = asc.bscans_per_volume
        fc.records_per_segment = asc.ascans_per_bscan
        aiming_format = FormatPlanner(get_logger('format-aiming', cfg.log_level))
        aiming_format.initialize(fc)
        self._aiming_format = aiming_format

        # format executors
        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(self._process.config.samples_per_ascan // 2)
        samples_to_save = cfec.sample_slice.count()

        sfe = StackFormatExecutor()
        sfe.initialize(cfec)

        raster_tensor_endpoint = StackDeviceTensorEndpoint(sfe, (rsc.bscans_per_volume, rsc.ascans_per_bscan, samples_to_save), get_logger('endpoint-raster', cfg.log_level))
        self._raster_tensor_endpoint = raster_tensor_endpoint

        aiming_tensor_endpoint = StackDeviceTensorEndpoint(sfe, (asc.bscans_per_volume, asc.ascans_per_bscan, samples_to_save), get_logger('endpoint-aiming', cfg.log_level))
        self._aiming_tensor_endpoint = aiming_tensor_endpoint

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [raster_format, aiming_format])
        ec.add_formatter(raster_format, [raster_tensor_endpoint])
        ec.add_formatter(aiming_format, [aiming_tensor_endpoint])
        ec.add_io(self._io_out, lead_samples=round(cfg.galvo_delay * self._io_out.config.samples_per_second))
        ec.add_io(self._strobe)

        ec.preload_count = cfg.preload_count
        ec.records_per_block = cfg.ascans_per_block
        ec.blocks_to_allocate = cfg.blocks_to_allocate
        ec.blocks_to_acquire = cfg.blocks_to_acquire

        ec.galvo_output_channels = len(self._io_out.config.channels)

        engine = Engine(get_logger('engine', cfg.log_level))
        self._engine = engine

        engine.initialize(ec)
        engine.prepare()

    def run(self):
        app = QApplication(sys.argv)

        import traceback
        def handler(cls, ex, trace):
            traceback.print_exception(cls, ex, trace)
            app.closeAllWindows()
        sys.excepthook = handler

        self._engine.scan_queue.append(self._scan)

        raster_widget = RasterEnFaceWidget(self._raster_tensor_endpoint)
        aiming_widgets = [CrossSectionImageWidget(self._aiming_tensor_endpoint, fixed=i, title=f'Aiming Cross-Section {i}') for i in range(2)]

        for w in aiming_widgets + [raster_widget]:
            w.show()

        self._raster_tensor_endpoint.aggregate_segment_callback = raster_widget.notify_segments

        def cb(v):
            for aw in aiming_widgets:
                aw.notify_segments(v)
        self._aiming_tensor_endpoint.aggregate_segment_callback = cb

        self._engine.start()

        try:
            while raster_widget.isVisible() and all([aw.isVisible() for aw in aiming_widgets]):
                if self._engine.wait_for(0.01):
                    break

                for w in aiming_widgets + [raster_widget]:
                    w.update()

                app.processEvents(QEventLoop.AllEvents, 10)

        except KeyboardInterrupt:
            pass
        finally:
            self._engine.stop()

if __name__ == '__main__':
    setup_logging()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS)
    engine.run()
