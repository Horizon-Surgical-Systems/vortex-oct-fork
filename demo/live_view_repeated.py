import sys
import os

from time import time

import numpy
import cupy

from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as get_logger
from vortex.scan import RepeatedRasterScanConfig, RepeatedRasterScan
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget, EnFaceImageWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

class RepeatedRasterEnFaceWidget(EnFaceImageWidget):
    def __init__(self, *args, **kwargs):
        self._repeat_count = kwargs.pop('repeat_count')

        super().__init__(*args, **kwargs)

        self.setWindowTitle(kwargs.pop('title', 'Repeated Raster En Face'))

    def _image_shape(self):
        return (self._endpoint.tensor.shape[0] // self._repeat_count, self._endpoint.tensor.shape[1])

    def _update_image(self, endpoint, raw_bscan_idxs):
        with endpoint.tensor as volume:
            endpoint.stream.synchronize()

            updates = {}
            for idx in raw_bscan_idxs:
                # which repeated bscan was updated
                bscan_idx = idx // self._repeat_count
                # which repetition number
                repeat_idx = idx % self._repeat_count

                # accumulate
                updates.setdefault(bscan_idx, []).append(repeat_idx)

            # update appropriate repeated bscans
            for (bscan_idx, repeat_idxs) in updates.items():
                # update based on the most recent repeat
                repeat_idx = max(repeat_idxs)

                # compute index into volume
                start = bscan_idx * self._repeat_count
                end = start + repeat_idx + 1

                # check validity
                if bscan_idx * self._repeat_count + repeat_idx > volume.shape[0]:
                    continue

                # compute variance SVP
                self.data[bscan_idx, ...] = cupy.asnumpy(cupy.max(cupy.std(volume[start:end, ...], axis=0, keepdims=True), axis=2))


class OCTEngine(BaseEngine):
    def __init__(self, cfg: StandardEngineParams):
        super().__init__(cfg)

        #
        # scan
        #

        sc = RepeatedRasterScanConfig()
        sc.bscans_per_volume = cfg.bscans_per_volume
        sc.ascans_per_bscan = cfg.ascans_per_bscan
        sc.repeat_count = 10
        sc.repeat_period = 10
        sc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        sc.volume_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        sc.samples_per_second = cfg.swept_source.triggers_per_second
        sc.loop = True

        scan = RepeatedRasterScan()
        scan.initialize(sc)
        self._scan = scan

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = cfg.bscans_per_volume * sc.repeat_count
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = False

        stack_format = FormatPlanner(get_logger('format', cfg.log_level))
        stack_format.initialize(fc)

        # format executors
        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(self._process.config.samples_per_ascan // 2)
        samples_to_save = cfec.sample_slice.count()

        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        stack_tensor_endpoint = StackDeviceTensorEndpoint(cfe, (sc.bscans_per_volume * sc.repeat_count, sc.ascans_per_bscan, samples_to_save), get_logger('cube', cfg.log_level))
        self._stack_tensor_endpoint = stack_tensor_endpoint

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [stack_format])
        ec.add_formatter(stack_format, [stack_tensor_endpoint])
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

        stack_widget = RasterEnFaceWidget(self._stack_tensor_endpoint)
        stack_widget.show()

        repeated_stack_widget = RepeatedRasterEnFaceWidget(self._stack_tensor_endpoint, repeat_count=self._scan.config.repeat_count)
        repeated_stack_widget.show()

        def cb(segments):
            stack_widget.notify_segments(segments)
            repeated_stack_widget.notify_segments(segments)
        self._stack_tensor_endpoint.aggregate_segment_callback = cb

        self._engine.start()

        try:
            while stack_widget.isVisible() and repeated_stack_widget.isVisible():
                if self._engine.wait_for(0.01):
                    break

                stack_widget.update()
                repeated_stack_widget.update()

                app.processEvents(QEventLoop.AllEvents, 10)

        except KeyboardInterrupt:
            pass
        finally:
            self._engine.stop()

if __name__ == '__main__':
    setup_logging()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS)
    engine.run()
