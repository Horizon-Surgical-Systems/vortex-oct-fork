import sys
import os
from math import pi

from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as get_logger
from vortex.marker import Flags
from vortex.scan import RasterScanConfig, RasterScan, RadialScanConfig, RadialScan, SpiralScanConfig, SpiralScan, limits
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint, RadialDeviceTensorEndpointInt8 as RadialDeviceTensorEndpoint, AscanSpiralDeviceTensorEndpointInt8 as AscanSpiralDeviceTensorEndpoint

from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, RadialFormatExecutorConfig, RadialFormatExecutor, SpiralFormatExecutorConfig, SpiralFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget, RadialEnFaceWidget, SpiralEnFaceWidget, CrossSectionImageWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

class OCTEngine(BaseEngine):
    def __init__(self, cfg: StandardEngineParams):
        super().__init__(cfg)

        #
        # scan
        #

        raster_sc = RasterScanConfig()
        raster_sc.bscans_per_volume = cfg.bscans_per_volume
        raster_sc.ascans_per_bscan = cfg.ascans_per_bscan
        raster_sc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.volume_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.bidirectional_segments = cfg.bidirectional
        raster_sc.bidirectional_volumes = cfg.bidirectional
        raster_sc.samples_per_second = cfg.swept_source.triggers_per_second
        raster_sc.loop = True
        raster_sc.flags = Flags(0x1)

        raster_scan = RasterScan()
        raster_scan.initialize(raster_sc)
        self._raster_scan = raster_scan

        radial_sc = RadialScanConfig()
        radial_sc.bscans_per_volume = cfg.bscans_per_volume
        radial_sc.ascans_per_bscan = cfg.ascans_per_bscan
        radial_sc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        # set half_evenly_spaced sets the volume extent to [0, pi)
        radial_sc.set_half_evenly_spaced(radial_sc.bscans_per_volume)
        # adjust angle to that radial scan is same orientation as raster scan
        radial_sc.angle = pi / 2
        radial_sc.bidirectional_segments = cfg.bidirectional
        radial_sc.bidirectional_volumes = cfg.bidirectional
        radial_sc.samples_per_second = cfg.swept_source.triggers_per_second
        radial_sc.loop = True
        radial_sc.flags = Flags(0x2)

        radial_scan = RadialScan()
        radial_scan.initialize(radial_sc)
        self._radial_scan = radial_scan

        spiral_sc = SpiralScanConfig()
        spiral_sc.volume_extent = Range(0, cfg.scan_dimension)
        spiral_sc.rings_per_spiral = cfg.bscans_per_volume
        spiral_sc.ascans_per_bscan = cfg.bscans_per_volume * cfg.ascans_per_bscan * 2
        spiral_sc.linear_velocity = 0
        spiral_sc.angular_velocity = 10
        spiral_sc.angle = pi / 2
        spiral_sc.samples_per_second = cfg.swept_source.triggers_per_second
        spiral_sc.loop = True
        spiral_sc.flags = Flags(0x4)

        spiral_scan = SpiralScan()
        spiral_scan.initialize(spiral_sc)
        self._spiral_scan = spiral_scan

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = False

        fc.mask = raster_sc.flags
        stack_format = FormatPlanner(get_logger('raster format', cfg.log_level))
        stack_format.initialize(fc)
        self._stack_format = stack_format

        fc.mask = radial_sc.flags
        radial_format = FormatPlanner(get_logger('radial format', cfg.log_level))
        radial_format.initialize(fc)
        self._radial_format = radial_format

        fc.mask = spiral_sc.flags
        fc.segments_per_volume = 1
        fc.records_per_segment = spiral_sc.samples_per_segment
        spiral_format = FormatPlanner(get_logger('spiral format', cfg.log_level))
        spiral_format.initialize(fc)
        self._spiral_format = spiral_format

        # format executors
        cfec = StackFormatExecutorConfig()
        # only keep half of the spectrum
        cfec.sample_slice = SimpleSlice(self._process.config.samples_per_ascan // 2)
        samples_to_save = cfec.sample_slice.count()


        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        stack_tensor_endpoint = StackDeviceTensorEndpoint(cfe, (raster_sc.bscans_per_volume, raster_sc.ascans_per_bscan, samples_to_save), get_logger('stack', cfg.log_level))
        self._stack_tensor_endpoint = stack_tensor_endpoint

        rfec = RadialFormatExecutorConfig()
        rfec.sample_slice = cfec.sample_slice
        rfec.volume_xy_extent = [Range(cfg.scan_dimension, -cfg.scan_dimension), Range(-cfg.scan_dimension, cfg.scan_dimension)]
        rfec.segment_rt_extent = (radial_sc.bscan_extent, radial_sc.volume_extent)
        rfec.radial_segments_per_volume = radial_sc.bscans_per_volume
        rfec.radial_records_per_segment = radial_sc.ascans_per_bscan

        rfe = RadialFormatExecutor()
        rfe.initialize(rfec)
        radial_tensor_endpoint = RadialDeviceTensorEndpoint(rfe, (raster_sc.bscans_per_volume, raster_sc.ascans_per_bscan, samples_to_save), get_logger('radial', cfg.log_level))
        self._radial_tensor_endpoint = radial_tensor_endpoint

        sfec = SpiralFormatExecutorConfig()
        sfec.sample_slice = cfec.sample_slice
        sfec.volume_xy_extent = [Range(cfg.scan_dimension, -cfg.scan_dimension), Range(-cfg.scan_dimension, cfg.scan_dimension)]
        sfec.radial_extent = spiral_sc.volume_extent
        sfec.rings_per_spiral = spiral_sc.rings_per_spiral
        sfec.samples_per_spiral = spiral_sc.samples_per_segment
        sfec.spiral_velocity = spiral_sc.angular_velocity

        sfe = SpiralFormatExecutor()
        sfe.initialize(sfec)
        spiral_tensor_endpoint = AscanSpiralDeviceTensorEndpoint(sfe, (raster_sc.bscans_per_volume, raster_sc.ascans_per_bscan, samples_to_save), get_logger('spiral', cfg.log_level))
        self._spiral_tensor_endpoint = spiral_tensor_endpoint

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [stack_format, radial_format, spiral_format])
        ec.add_formatter(stack_format, [stack_tensor_endpoint])
        ec.add_formatter(radial_format, [radial_tensor_endpoint])
        ec.add_formatter(spiral_format, [spiral_tensor_endpoint])
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

    def _handle_keypress(self, e):
        if e.key() == Qt.Key.Key_1:
            print('switch to raster scan')

            # clear volume
            with self._stack_tensor_endpoint.tensor as volume:
                volume[:] = 0
            # invalidate all B-scans
            self._stack_widget.notify_segments(range(self._raster_scan.config.bscans_per_volume))

            self._engine.scan_queue.interrupt(self._raster_scan)

        elif e.key() == Qt.Key.Key_2:
            print('switch to radial scan')

            # clear volume
            with self._radial_tensor_endpoint.tensor as volume:
                volume[:] = 0
            # invalidate all B-scans
            self._radial_widget.notify_segments([0])

            self._engine.scan_queue.interrupt(self._radial_scan)

        elif e.key() == Qt.Key.Key_3:
            print('switch to spiral scan')

            # clear volume
            with self._spiral_tensor_endpoint.tensor as volume:
                volume[:] = 0
            # invalidate all B-scans
            self._spiral_widget.notify_segments([0])

            self._engine.scan_queue.interrupt(self._spiral_scan)

        elif e.key() == Qt.Key.Key_Q:
            self._engine.stop()

    def run(self):
        app = QApplication(sys.argv)

        import traceback
        def handler(cls, ex, trace):
            traceback.print_exception(cls, ex, trace)
            app.closeAllWindows()
        sys.excepthook = handler

        self._engine.scan_queue.append(self._raster_scan)

        self._stack_widget = RasterEnFaceWidget(self._stack_tensor_endpoint)
        self._radial_widget = RadialEnFaceWidget(self._radial_tensor_endpoint)
        self._spiral_widget = SpiralEnFaceWidget(self._spiral_tensor_endpoint)
        self._cross_widget = CrossSectionImageWidget(self._stack_tensor_endpoint)

        self._stack_tensor_endpoint.aggregate_segment_callback = self._stack_widget.notify_segments
        self._radial_tensor_endpoint.aggregate_segment_callback = self._radial_widget.notify_segments
        self._spiral_tensor_endpoint.update_callback = lambda: self._spiral_widget.notify_segments([0])

        def cb(v):
            self._stack_widget.notify_segments(v)
            self._cross_widget.notify_segments(v)
        self._stack_tensor_endpoint.aggregate_segment_callback = cb
        def cb(v):
            self._radial_widget.notify_segments(v)
        self._radial_tensor_endpoint.aggregate_segment_callback = cb
        def cb(v):
            self._spiral_widget.notify_segments(v)
        self._spiral_tensor_endpoint.aggregate_segment_callback = cb

        self._stack_widget.keyPressEvent = self._handle_keypress
        self._radial_widget.keyPressEvent = self._handle_keypress
        self._spiral_widget.keyPressEvent = self._handle_keypress
        self._cross_widget.keyPressEvent = self._handle_keypress

        self._stack_widget.show()
        self._radial_widget.show()
        self._spiral_widget.show()
        self._cross_widget.show()

        self._stack_widget.setWindowTitle('Raster Scan (Press 1)')
        self._radial_widget.setWindowTitle('Radial Scan (Press 2)')
        self._spiral_widget.setWindowTitle('Spiral Scan (Press 3)')

        self._engine.start()

        try:
            while (self._stack_widget.isVisible() or self._radial_widget.isVisible() or self._spiral_widget.isVisible()) and self._cross_widget.isVisible():
                if self._engine.wait_for(0.01):
                    break

                self._stack_widget.update()
                self._radial_widget.update()
                self._spiral_widget.update()
                self._cross_widget.update()

                app.processEvents(QEventLoop.AllEvents, 10)

        except KeyboardInterrupt:
            pass
        finally:
            self._engine.stop()

if __name__ == '__main__':
    setup_logging()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS)
    engine.run()
