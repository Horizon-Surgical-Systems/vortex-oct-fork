import sys
from typing import Tuple
from dataclasses import dataclass
import logging

import numpy

from rainbow_logging_handler import RainbowLoggingHandler

from vortex import get_console_logger as get_logger
from vortex.engine import Block, source, acquire_alazar_clock, find_rising_edges, compute_resampling, dispersion_phasor

from vortex.acquire import AlazarConfig, AlazarAcquisition, alazar
from vortex.process import CUDAProcessor, CUDAProcessorConfig
from vortex.io import DAQmxIO, DAQmxConfig, daqmx
from vortex.engine import source, Source

@dataclass
class StandardEngineParams:

    # scan parameters
    scan_dimension: float
    bidirectional: bool
    ascans_per_bscan: int
    bscans_per_volume: int
    galvo_delay: float

    # acquisition parameters
    clock_samples_per_second: int
    blocks_to_acquire: int
    ascans_per_block: int
    samples_per_ascan: int
    trigger_delay_seconds: float

    # hardware configuration
    swept_source: Source
    internal_clock: bool
    clock_channel: alazar.Channel
    input_channel: alazar.Channel

    # engine memory parameters
    blocks_to_allocate: int
    preload_count: int

    # processing control
    process_slots: int
    dispersion: Tuple[float, float]

    # logging
    log_level: int

DEFAULT_ENGINE_PARAMS = StandardEngineParams(
    scan_dimension=5,
    bidirectional=False,
    ascans_per_bscan=500,
    bscans_per_volume=500,
    galvo_delay=95e-6,

    clock_samples_per_second=int(800e6),
    # zero blocks to acquire means infinite acquisition
    blocks_to_acquire=0,
    ascans_per_block=500,
    samples_per_ascan=2752,
    trigger_delay_seconds=0,

    blocks_to_allocate=128,
    preload_count=32,

    swept_source=source.Axsun200k,
    internal_clock=True,
    clock_channel=alazar.Channel.A,
    input_channel=alazar.Channel.B,

    process_slots=2,
    dispersion=(2.8e-5, 0),

    log_level=1,
)

class BaseEngine:
    def __init__(self, cfg: StandardEngineParams):

        #
        # acquisition
        #

        ac = AlazarConfig()
        if cfg.internal_clock:
            ac.clock = alazar.InternalClock(cfg.clock_samples_per_second)
        else:
            ac.clock = alazar.ExternalClock()

        ac.inputs.append(alazar.Input(cfg.input_channel))
        ac.options.append(alazar.AuxIOTriggerOut())

        ac.records_per_block = cfg.ascans_per_block
        ac.trigger.delay_samples = int(cfg.trigger_delay_seconds * cfg.clock_samples_per_second)

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

        acquire = AlazarAcquisition(get_logger('acquire', cfg.log_level))
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

        # reasmpling
        pc.resampling_samples = resampling

        # spectral filter with dispersion correction
        window = numpy.hanning(pc.samples_per_ascan)
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
        sc = ioc_out.copy()

        ioc_out.name = 'output'

        stream = Block.StreamIndex.GalvoTarget
        ioc_out.channels.append(daqmx.AnalogVoltageOutput('Dev1/ao0', 15 / 10, stream, 0))
        ioc_out.channels.append(daqmx.AnalogVoltageOutput('Dev1/ao1', 15 / 10, stream, 1))

        io_out = DAQmxIO(get_logger(ioc_out.name, cfg.log_level))
        io_out.initialize(ioc_out)
        self._io_out = io_out

        sc.name = 'strobe'
        sc.channels.append(daqmx.DigitalOutput('Dev1/port0', Block.StreamIndex.Strobes))
        strobe = DAQmxIO(get_logger(sc.name, cfg.log_level))
        strobe.initialize(sc)
        self._strobe = strobe

def setup_logging():
    # configure the root logger to accept all records
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(name)s] %(filename)s:%(lineno)d\t%(levelname)s:\t%(message)s')

    # set up colored logging to console
    console_handler = RainbowLoggingHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
