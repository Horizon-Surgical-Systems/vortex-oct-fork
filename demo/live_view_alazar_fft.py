import sys
import os

from time import time, sleep
from math import pi

import numpy

from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as get_logger
from vortex.scan import RasterScanConfig, RasterScan
from vortex.engine import EngineConfig, Engine, Block, acquire_alazar_clock, find_rising_edges, compute_resampling, dispersion_phasor, StackTensorEndpointInt8 as StackTensorEndpoint

from vortex.acquire import AlazarAcquisition, AlazarConfig, AlazarFFTConfig, AlazarFFTAcquisition, alazar
from vortex.process import CopyProcessor, CopyProcessorConfig
from vortex.io import DAQmxIO, DAQmxConfig, AnalogVoltageOutput
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice, LinearTransform

from vortex_tools.ui.display import RasterEnFaceWidget, CrossSectionImageWidget

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS

class OCTEngine:
    def __init__(self, cfg: StandardEngineParams):
        #
        # scan
        #

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

        ac = AlazarFFTConfig()
        if cfg.internal_clock:
            ac.clock = alazar.InternalClock(cfg.clock_samples_per_second)
        else:
            ac.clock = alazar.ExternalClock()

        # only channel A supported for DSP
        ac.inputs.append(alazar.Input(alazar.Channel.A))
        ac.options.append(alazar.AuxIOTriggerOut())

        ac.records_per_block = cfg.ascans_per_block

        #
        # clocking
        #

        board = alazar.Board(ac.device.system_index, ac.device.board_index)
        if cfg.internal_clock:
            (clock_samples_per_second, clock) = acquire_alazar_clock(cfg.swept_source, ac, cfg.clock_channel, get_logger('acquire', cfg.log_level))
            cfg.swept_source.clock_edges_seconds = find_rising_edges(clock, clock_samples_per_second, len(cfg.swept_source.clock_edges_seconds))
            resampling = compute_resampling(cfg.swept_source, ac.samples_per_second, cfg.samples_per_ascan)

            # acquire enough samples to obtain the required ones
            ac.samples_per_record = board.info.smallest_aligned_samples_per_record(resampling.max())
        else:
            resampling = []
            ac.samples_per_record = board.info.smallest_aligned_samples_per_record(cfg.swept_source.clock_rising_edges_per_trigger)

        # ref: https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
        ac.fft_length = 1 << (ac.samples_per_record - 1).bit_length()
        ac.include_time_domain = False

        #
        # background subtraction
        #

        def get_background():
            acb = AlazarConfig()
            acb.clock = ac.clock
            acb.inputs = ac.inputs
            acb.options = ac.options

            acb.records_per_block = 1000
            acb.samples_per_record = ac.samples_per_record

            ab = AlazarAcquisition(get_logger('acquire', cfg.log_level))
            ab.initialize(acb)
            ab.prepare()

            buffer = numpy.zeros(acb.shape, dtype=numpy.uint16)

            count = None
            exc = None
            def cb(n, e):
                nonlocal count, exc
                count = n
                exc = e

            ab.next_async(buffer, cb)

            ab.start()
            sleep(1)
            ab.stop()

            # change the background from uint16 to int16
            bg = numpy.mean(buffer, axis=0).astype(numpy.int16)[:, 0] + 2**15 - 1

            return bg

        ac.background = numpy.concatenate((get_background(), numpy.zeros((ac.fft_length - ac.samples_per_record,))))

        # spectral filter
        window = numpy.hamming(ac.samples_per_record)
        phasor = numpy.conj(dispersion_phasor(len(window), cfg.dispersion))
        ac.spectral_filter = numpy.concatenate((window * phasor, numpy.zeros((ac.fft_length - ac.samples_per_record,))))

        acquire = AlazarFFTAcquisition(get_logger('acquire', cfg.log_level))
        acquire.initialize(ac)
        self._acquire = acquire

        #
        # OCT processing setup
        #

        pc = CopyProcessorConfig()

        # match acquisition settings
        pc.samples_per_record = ac.samples_per_ascan
        pc.ascans_per_block = ac.records_per_block

        pc.sample_slice = SimpleSlice(50, pc.samples_per_record)
        pc.sample_transform = LinearTransform(1/256, -10*numpy.log10(pc.samples_per_ascan))

        process = CopyProcessor(get_logger('process', cfg.log_level))
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
        ioc_in = ioc_out.copy()

        ioc_out.name = 'output'

        stream = Block.StreamIndex.GalvoTarget
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao0', 15 / 10, stream, 0))
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao1', 15 / 10, stream, 1))

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
        ec.add_acquisition(acquire, [process])
        ec.add_processor(process, [stack_format])
        ec.add_formatter(stack_format, [stack_tensor_endpoint])
        ec.add_io(io_out, lead_samples=round(cfg.galvo_delay * ioc_out.samples_per_second))
        # ec.add_io(io_in, preload=False)

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
