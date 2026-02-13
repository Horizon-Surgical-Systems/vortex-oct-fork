import sys
import os
from math import pi

from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import QApplication

from vortex import Range, get_console_logger as gcl
from vortex.scan import RasterScanConfig, RasterScan
from vortex.engine import EngineConfig, Engine, SpectraHDF5StackEndpoint, AscanHDF5StackEndpoint, SpectraStackEndpoint, AscanStackEndpoint, NullEndpoint
from vortex.storage import HDF5StackUInt16, HDF5StackInt8, HDF5StackConfig, HDF5StackHeader, SimpleStackUInt16, SimpleStackInt8, SimpleStackConfig, SimpleStackHeader
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice
from vortex.process import NullProcessor, NullProcessorConfig

# hack to simplify running demos
sys.path.append(os.path.dirname(__file__))

from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

class OCTEngine(BaseEngine):
    def __init__(self, cfg: StandardEngineParams, args):
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
        raster_sc.loop = False

        raster_scan = RasterScan()
        raster_scan.initialize(raster_sc)
        self._raster_scan = raster_scan

        #
        # null processor for spectra
        #

        pc = NullProcessorConfig()
        pc.samples_per_record = self._process.config.samples_per_record
        pc.ascans_per_block = self._process.config.ascans_per_block

        null_process = NullProcessor()
        null_process.initialize(pc)

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = cfg.bscans_per_volume
        fc.records_per_segment = cfg.ascans_per_bscan
        fc.adapt_shape = False

        stack_format_spectra = FormatPlanner(gcl('format-spectra', cfg.log_level))
        stack_format_spectra.initialize(fc)

        stack_format_ascans = FormatPlanner(gcl('format-ascans', cfg.log_level))
        stack_format_ascans.initialize(fc)

        # format executors
        cfec_spectra = StackFormatExecutorConfig()
        cfe_spectra  = StackFormatExecutor()
        cfe_spectra.initialize(cfec_spectra)

        cfec_ascans = StackFormatExecutorConfig()
        if args.discard_conjugate:
            cfec_ascans.sample_slice = SimpleSlice(args.discard_dc, self._process.config.samples_per_ascan // 2)
        else:
            cfec_ascans.sample_slice = SimpleSlice(args.discard_dc, self._process.config.samples_per_ascan)
        ascan_samples_to_save = cfec_ascans.sample_slice.count()
        cfe_ascans  = StackFormatExecutor()
        cfe_ascans.initialize(cfec_ascans)

        self._stack_spectra_storage = None
        self._stack_ascan_storage = None

        spectra_endpoints = []
        ascan_endpoints = []

        if args.format in ['matlab', 'hdf5']:

            fmt = HDF5StackHeader.MATLAB if args.format == 'matlab' else HDF5StackHeader.Empty
            suffix = '.mat' if args.format == 'matlab' else '.h5'

            if not args.no_save_spectra:
                # save spectra data
                h5sc = HDF5StackConfig()
                h5sc.shape = (cfg.bscans_per_volume, cfg.ascans_per_bscan, self._acquire.config.samples_per_record, self._acquire.config.channels_per_sample)
                h5sc.header = fmt
                h5sc.path = f'{args.prefix}spectra{suffix}'

                self._stack_spectra_storage = HDF5StackUInt16(gcl('hdf5-spectra', cfg.log_level))
                self._stack_spectra_storage.open(h5sc)

                if args.direct_mode:
                    spectra_endpoints.append(SpectraHDF5StackEndpoint(self._stack_spectra_storage, log=gcl('spectra', cfg.log_level)))
                else:
                    spectra_endpoints.append(SpectraHDF5StackEndpoint(cfe_spectra, self._stack_spectra_storage, log=gcl('spectra', cfg.log_level)))


            if not args.no_save_ascans:
                # save ascan data
                h5sc = HDF5StackConfig()
                h5sc.shape = (cfg.bscans_per_volume, cfg.ascans_per_bscan, ascan_samples_to_save, 1)
                h5sc.header = fmt
                h5sc.path = f'{args.prefix}ascans{suffix}'

                self._stack_ascan_storage = HDF5StackInt8(gcl('hdf5-ascan', cfg.log_level))
                self._stack_ascan_storage.open(h5sc)
                ascan_endpoints.append(AscanHDF5StackEndpoint(cfe_ascans, self._stack_ascan_storage, log=gcl('hdf5-ascan', cfg.log_level)))

        else:

            if not args.no_save_spectra:
                # save spectra data
                npsc = SimpleStackConfig()
                npsc.shape = (cfg.bscans_per_volume, cfg.ascans_per_bscan, self._acquire.config.samples_per_record, self._acquire.config.channels_per_sample)
                npsc.header = SimpleStackHeader.NumPy
                npsc.path = f'{args.prefix}spectra.npy'

                self._stack_spectra_storage = SimpleStackUInt16(gcl('npy-spectra', cfg.log_level))
                self._stack_spectra_storage.open(npsc)

                if args.direct_mode:
                    spectra_endpoints.append(SpectraStackEndpoint(self._stack_spectra_storage, log=gcl('spectra', cfg.log_level)))
                else:
                    spectra_endpoints.append(SpectraStackEndpoint(cfe_spectra, self._stack_spectra_storage, log=gcl('spectra', cfg.log_level)))

            if not args.no_save_ascans:
                # save ascan data
                npsc = SimpleStackConfig()
                npsc.shape = (cfg.bscans_per_volume, cfg.ascans_per_bscan, ascan_samples_to_save, 1)
                npsc.header = SimpleStackHeader.NumPy
                npsc.path = f'{args.prefix}ascans.npy'

                self._stack_ascan_storage = SimpleStackInt8(gcl('npy-ascan', cfg.log_level))
                self._stack_ascan_storage.open(npsc)
                ascan_endpoints.append(AscanStackEndpoint(cfe_ascans, self._stack_ascan_storage, log=gcl('npy-ascan', cfg.log_level)))

        # always have at least one endpoint
        if not (spectra_endpoints or ascan_endpoints):
            spectra_endpoints.append(NullEndpoint())

        #
        # engine setup
        #

        ec = EngineConfig()
        processors = []

        if ascan_endpoints:
            processors.append(self._process)
            ec.add_processor(self._process, [stack_format_ascans])
            ec.add_formatter(stack_format_ascans, ascan_endpoints)

        if spectra_endpoints:
            processors.append(null_process)
            ec.add_processor(null_process, [stack_format_spectra])
            ec.add_formatter(stack_format_spectra, spectra_endpoints)

        ec.add_acquisition(self._acquire, processors)
        ec.add_io(self._io_out, lead_samples=round(cfg.galvo_delay * self._io_out.config.samples_per_second))
        ec.add_io(self._strobe)

        ec.preload_count = cfg.preload_count
        ec.records_per_block = cfg.ascans_per_block
        ec.blocks_to_allocate = cfg.blocks_to_allocate
        ec.blocks_to_acquire = cfg.blocks_to_acquire

        ec.galvo_output_channels = len(self._io_out.config.channels)
        ec.galvo_input_channels = 0

        engine = Engine(gcl('engine', cfg.log_level))
        self._engine = engine

        engine.initialize(ec)
        engine.prepare()

    def run(self, count):

        for _ in range(count):
            self._engine.scan_queue.append(self._raster_scan)

        self._engine.start()
        self._engine.wait()
        self._engine.stop()

        if self._stack_spectra_storage:
            self._stack_spectra_storage.close()
        if self._stack_ascan_storage:
            self._stack_ascan_storage.close()

if __name__ == '__main__':
    setup_logging()

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='save volume to disk', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--format', choices=['matlab', 'hdf5', 'numpy'], default='numpy', help='file format for saving')
    parser.add_argument('--count', type=int, default=1, help='number of volumes to save')
    parser.add_argument('--no-save-ascans', action='store_true', help='do not save A-scans')
    parser.add_argument('--no-save-spectra', action='store_true', help='do not save spectra')
    parser.add_argument('--direct-mode', action='store_true', help='write directly to disk (only for spectra)')
    parser.add_argument('--discard-dc', type=int, default=0, help='number of samples near DC to discard in processed data')
    parser.add_argument('--discard-conjugate', action='store_true', help='discard the complex conjugate portion of processed data')
    parser.add_argument('--prefix', default='', help='prefix for output file names')
    args = parser.parse_args()

    engine = OCTEngine(DEFAULT_ENGINE_PARAMS, args)
    engine.run(args.count)
