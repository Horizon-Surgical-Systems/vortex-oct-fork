import sys
import os

from time import time
from math import pi

import numpy

from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import QApplication


from vortex import Range, get_console_logger as get_logger
from vortex.marker import Flags
from vortex.scan import RasterScanConfig, RasterScan, RadialScanConfig, RadialScan
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint, RadialDeviceTensorEndpointInt8 as RadialDeviceTensorEndpoint

from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, RadialFormatExecutorConfig, RadialFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget, RadialEnFaceWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

class OCTEngine(BaseEngine):
    def __init__(self, cfg: StandardEngineParams):
        cfg.preload_count *= 4

        super().__init__(cfg)
        #
        # scan
        #

        raster_sc = RasterScanConfig()
        raster_sc.bscans_per_volume = cfg.bscans_per_volume
        raster_sc.ascans_per_bscan = cfg.ascans_per_bscan
        raster_sc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.volume_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.bidirectional_segments = True
        raster_sc.bidirectional_volumes = True
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
        radial_sc.set_half_evenly_spaced(radial_sc.bscans_per_volume)
        radial_sc.bidirectional_segments = True
        radial_sc.bidirectional_volumes = True
        radial_sc.samples_per_second = cfg.swept_source.triggers_per_second
        radial_sc.loop = True
        radial_sc.flags = Flags(0x2)

        radial_scan = RadialScan()
        radial_scan.initialize(radial_sc)
        self._radial_scan = radial_scan

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = True

        fc.mask = Flags(0x01)
        stack_format = FormatPlanner(get_logger('format', cfg.log_level + 1))
        stack_format.initialize(fc)
        self._stack_format = stack_format

        fc.mask = Flags(0x02)
        radial_format = FormatPlanner(get_logger('format', cfg.log_level + 1))
        radial_format.initialize(fc)
        self._radial_format = radial_format

        # format executors
        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(self._process.config.samples_per_ascan // 2)
        samples_to_save = cfec.sample_slice.count()


        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        stack_tensor_endpoint = StackDeviceTensorEndpoint(cfe, (raster_sc.bscans_per_volume, raster_sc.ascans_per_bscan, samples_to_save), get_logger('cube', cfg.log_level))
        self._stack_tensor_endpoint = stack_tensor_endpoint

        rfec = RadialFormatExecutorConfig()
        rfec.sample_slice = cfec.sample_slice
        rfec.volume_xy_extent = (Range(-5, 5), Range(-5, 5))
        rfec.segment_rt_extent = (Range(-5, 5), Range(0, pi))
        rfec.radial_segments_per_volume = radial_sc.bscans_per_volume
        rfec.radial_records_per_segment = radial_sc.ascans_per_bscan

        rfe = RadialFormatExecutor()
        rfe.initialize(rfec)
        radial_tensor_endpoint = RadialDeviceTensorEndpoint(rfe, (1000, 1000, samples_to_save), get_logger('radial', cfg.log_level))
        self._radial_tensor_endpoint = radial_tensor_endpoint

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [stack_format, radial_format])
        ec.add_formatter(stack_format, [stack_tensor_endpoint])
        ec.add_formatter(radial_format, [radial_tensor_endpoint])
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

        self._engine.scan_queue.append(self._raster_scan)

        stack_widget = RasterEnFaceWidget(self._stack_tensor_endpoint)
        radial_widget = RadialEnFaceWidget(self._radial_tensor_endpoint)

        stack_widget.show()
        radial_widget.show()

        self._stack_tensor_endpoint.aggregate_segment_callback = stack_widget.notify_segments
        self._radial_tensor_endpoint.aggregate_segment_callback = radial_widget.notify_segments

        self._next_handler_id = 1
        self._handlers = {}

        def handle_event(sample, eid):
            handler = self._handlers.pop(eid, None)
            if handler:
                handler(eid)

        self._stack_tensor_endpoint.event_callback = handle_event

        def handle_keypress(e):

            mapping = {
                Qt.Key_Right: ( 0, -1),
                Qt.Key_Left:  ( 0,  1),
                Qt.Key_Up:    ( 1,  0),
                Qt.Key_Down:  (-1,  0),
            }

            if delta := mapping.get(e.key()):
                config = self._raster_scan.config
                config.offset += delta
                self._raster_scan.change(config, False)

                config = self._radial_scan.config
                config.offset += delta
                self._radial_scan.change(config, False)

            if e.key() == Qt.Key_I:
                config = self._raster_scan.config
                config.bscans_per_volume += 50

                def h(eid):
                    tensor = self._stack_tensor_endpoint.tensor
                    with tensor:
                        self._stack_tensor_endpoint.stream.synchronize()
                        orig_shape = tensor.shape
                        shape = orig_shape[:]
                        shape[0] = config.bscans_per_volume
                        tensor.resize(shape)
                        print(eid, tensor.shape)

                eid = self._next_handler_id
                self._next_handler_id += 1
                # print(eid, config.bscans_per_volume)

                self._handlers[eid] = h
                self._raster_scan.change(config, False, eid)

            elif e.key() == Qt.Key_F:
                config = self._raster_scan.config
                config.bscans_per_volume = max([ config.bscans_per_volume - 50, 50 ])

                def h(eid):
                    tensor = self._stack_tensor_endpoint.tensor
                    with tensor:
                        self._stack_tensor_endpoint.stream.synchronize()
                        orig_shape = tensor.shape
                        shape = orig_shape[:]
                        shape[0] = config.bscans_per_volume
                        tensor.resize(shape)
                        print(eid, tensor.shape)

                eid = self._next_handler_id
                self._next_handler_id += 1
                # print(eid, config.bscans_per_volume)

                self._handlers[eid] = h
                self._raster_scan.change(config, False, eid)

            # if e.key() == Qt.Key_U:
            #     config = self._radial_scan.config
            #     config.ascans_per_bscan += 100
            #     self._radial_scan.change(config, False, 2)
            # elif e.key() == Qt.Key_D:
            #     config = self._radial_scan.config
            #     config.ascans_per_bscan = max([ config.ascans_per_bscan - 100, 100 ])
            #     self._radial_scan.change(config, False, 2)

            if e.key() == Qt.Key_Q:
                app.closeAllWindows()

            if e.key() == Qt.Key_1:
                print('raster')
                self._engine.scan_queue.interrupt(self._raster_scan)
            if e.key() == Qt.Key_2:
                print('radial')
                self._engine.scan_queue.interrupt(self._radial_scan)

        stack_widget.keyPressEvent = handle_keypress
        radial_widget.keyPressEvent = handle_keypress

        self._engine.start()

        try:
            while stack_widget.isVisible() and radial_widget.isVisible():
                if self._engine.wait_for(0.01):
                    break

                app.processEvents(QEventLoop.AllEvents, 10)

        except KeyboardInterrupt:
            pass
        finally:
            self._engine.stop()

if __name__ == '__main__':
    setup_logging()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS)
    engine.run()
