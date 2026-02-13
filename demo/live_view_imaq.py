import sys
import os

from time import time, sleep
from math import pi

import numpy

from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as get_logger
from vortex.scan import RasterScanConfig, RasterScan
from vortex.engine import EngineConfig, Engine, Block, dispersion_phasor, StackDeviceTensorEndpointInt8 as StackTensorEndpoint

from vortex.acquire import ImaqAcquisition, ImaqAcquisitionConfig
from vortex.process import CUDAProcessor, CUDAProcessorConfig
from vortex.io import DAQmxIO, DAQmxConfig, daqmx
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor

from vortex_tools.ui.display import RasterEnFaceWidget, CrossSectionImageWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS

class OCTEngine:
    def __init__(self, cfg: StandardEngineParams):
        #
        # scan
        #

        cfg.samples_per_ascan = 2044
        cfg.swept_source.triggers_per_second = 20000

        raster_sc = RasterScanConfig()
        raster_sc.bscans_per_volume = cfg.bscans_per_volume
        raster_sc.ascans_per_bscan = cfg.ascans_per_bscan
        raster_sc.bscan_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.volume_extent = Range(-cfg.scan_dimension, cfg.scan_dimension)
        raster_sc.bidirectional_segments = True
        raster_sc.bidirectional_volumes = False
        raster_sc.samples_per_second = cfg.swept_source.triggers_per_second
        raster_sc.loop = True

        raster_scan = RasterScan()
        raster_scan.initialize(raster_sc)
        self._raster_scan = raster_scan

        #
        # acquisition
        #

        ac = ImaqAcquisitionConfig()
        ac.samples_per_record = cfg.samples_per_ascan
        ac.records_per_block = cfg.ascans_per_block

        acquire = ImaqAcquisition(get_logger('acquire', cfg.log_level))
        acquire.initialize(ac)
        self._acquire = acquire

        #
        # OCT processing setup
        #

        pc = CUDAProcessorConfig()

        # match acquisition settings
        pc.samples_per_record = ac.samples_per_record
        pc.ascans_per_block = ac.records_per_block

        pc.slots = cfg.process_slots

        # spectral filter with dispersion correction
        window = numpy.hamming(pc.samples_per_ascan)
        phasor = dispersion_phasor(len(window), cfg.dispersion)
        pc.spectral_filter = window * phasor

        # DC subtraction per block
        pc.average_window = 2 * pc.ascans_per_block

        process = CUDAProcessor(get_logger('process', cfg.log_level))
        process.initialize(pc)
        self._process = process

        #
        # galvo control
        #

        # output
        ioc_out = DAQmxConfig()
        ioc_out.samples_per_block = ac.records_per_block
        ioc_out.samples_per_second = cfg.swept_source.triggers_per_second
        ioc_out.blocks_to_buffer = cfg.preload_count

        ioc_out.name = 'output'

        stream = Block.StreamIndex.GalvoTarget
        ioc_out.channels.append(daqmx.AnalogVoltageOutput('Dev1/ao0', 15 / 10, stream, 0))
        ioc_out.channels.append(daqmx.AnalogVoltageOutput('Dev1/ao1', 15 / 10, stream, 1))

        io_out = DAQmxIO(get_logger(ioc_out.name, cfg.log_level))
        io_out.initialize(ioc_out)
        self._io_out = io_out

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = False

        stack_format = FormatPlanner(get_logger('format', cfg.log_level))
        stack_format.initialize(fc)
        self._stack_format = stack_format

        # format executors
        cfec = StackFormatExecutorConfig()
        # cfec.sample_slice = SimpleSlice(0, pc.samples_per_ascan)
        samples_to_save = pc.samples_per_ascan #cfec.sample_slice.count()

        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        stack_tensor_endpoint = StackTensorEndpoint(cfe, (raster_sc.bscans_per_volume, raster_sc.ascans_per_bscan, samples_to_save), get_logger('cube', cfg.log_level))
        self._stack_tensor_endpoint = stack_tensor_endpoint
        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(acquire, [process], preload=False)
        ec.add_processor(process, [stack_format])
        ec.add_formatter(stack_format, [stack_tensor_endpoint])
        ec.add_io(io_out, lead_samples=round(cfg.galvo_delay * ioc_out.samples_per_second))

        ec.preload_count = cfg.preload_count
        ec.records_per_block = cfg.ascans_per_block
        ec.blocks_to_allocate = cfg.blocks_to_allocate
        ec.blocks_to_acquire = cfg.blocks_to_acquire

        ec.galvo_output_channels = len(io_out.config.channels)

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
        cross_widget = CrossSectionImageWidget(self._stack_tensor_endpoint)

        stack_widget.show()
        cross_widget.show()

        def cb(v):
            stack_widget.notify_segments(v)
            cross_widget.notify_segments(v)
        self._stack_tensor_endpoint.aggregate_segment_callback = cb

        self._engine.start()

        try:
            while stack_widget.isVisible() and cross_widget.isVisible():
                if self._engine.wait_for(0.01):
                    break

                stack_widget.update()
                cross_widget.update()

                app.processEvents(QEventLoop.AllEvents, 10)

        except KeyboardInterrupt:
            pass
        finally:
            self._engine.stop()

if __name__ == '__main__':
    setup_logging()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS)
    engine.run()
