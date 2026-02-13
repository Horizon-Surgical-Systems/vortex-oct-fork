import sys
from pathlib import Path
import logging
from textwrap import dedent

import numpy as np
import cupy as cp

from qtpy.QtCore import QTimer, pyqtSignal
from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QMessageBox

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas

from vortex import Range, get_console_logger as get_logger
from vortex.scan import RasterScanConfig, RasterScan
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget

# hack to simplify running demos
sys.path.append(Path(__file__).parent.parent.as_posix())
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

_log = logging.getLogger(__name__)

class GalvoTunerWindow(QWidget):
    stop_signal = pyqtSignal()

    def __init__(self, params: StandardEngineParams, **kwargs):
        super().__init__(**kwargs)
        params.bscans_per_volume = 100
        self._params = params

        self._system = BaseEngine(params)

        #
        # scan
        #

        self._scan_config = RasterScanConfig()
        self._scan_config.bscans_per_volume = params.bscans_per_volume
        self._scan_config.ascans_per_bscan = params.ascans_per_bscan
        self._scan_config.bscan_extent = Range(-params.scan_dimension, params.scan_dimension)
        self._scan_config.volume_extent = Range(0, 0)
        self._scan_config.bidirectional_segments = True
        self._scan_config.bidirectional_volumes = False
        self._scan_config.samples_per_second = params.swept_source.triggers_per_second
        self._scan_config.loop = False

        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.segments_per_volume = params.bscans_per_volume
        fc.records_per_segment = params.ascans_per_bscan
        fc.adapt_shape = False

        self._stack_format = FormatPlanner(get_logger('format', params.log_level))
        self._stack_format.initialize(fc)

        # format executors
        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(self._system._process.config.samples_per_ascan // 2)
        samples_to_save = cfec.sample_slice.count()

        cfe = StackFormatExecutor()
        cfe.initialize(cfec)
        self._stack_endpoint = StackDeviceTensorEndpoint(cfe, (self._scan_config.bscans_per_volume, self._scan_config.ascans_per_bscan, samples_to_save), get_logger('stack', params.log_level))

        self._engine = None

        #
        # UI
        #

        self.setWindowTitle('Vortex - Galvo Tuner')
        self.resize(720, 720)

        layout = QVBoxLayout(self)

        self._stack_widget = RasterEnFaceWidget(self._stack_endpoint)
        self._stack_endpoint.aggregate_segment_callback = self._stack_widget.notify_segments
        layout.addWidget(self._stack_widget)

        self._canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._canvas)
        self._axes = None

        panel = QHBoxLayout()
        layout.addLayout(panel)

        label = QLabel('Delay (s):')
        panel.addWidget(label)

        self._galvo_delay_spin = QDoubleSpinBox()
        self._galvo_delay_spin.setMinimum(0)
        self._galvo_delay_spin.setMaximum(1)
        self._galvo_delay_spin.setValue(0)
        self._galvo_delay_spin.setSingleStep(1 / self._scan_config.samples_per_second)
        self._galvo_delay_spin.setDecimals(6)
        panel.addWidget(self._galvo_delay_spin)

        label = QLabel('Velocity Limit (1/s):')
        panel.addWidget(label)

        self._galvo_velocity_spin = QDoubleSpinBox()
        self._galvo_velocity_spin.setMinimum(0)
        self._galvo_velocity_spin.setMaximum(1e12)
        self._galvo_velocity_spin.setValue(self._scan_config.limits[0].velocity)
        self._galvo_velocity_spin.setSingleStep(1)
        self._galvo_velocity_spin.setDecimals(0)
        panel.addWidget(self._galvo_velocity_spin)

        label = QLabel('Acceleration Limit (1/s^2):')
        panel.addWidget(label)

        self._galvo_acceleration_spin = QDoubleSpinBox()
        self._galvo_acceleration_spin.setMinimum(0)
        self._galvo_acceleration_spin.setMaximum(1e12)
        self._galvo_acceleration_spin.setValue(self._scan_config.limits[0].acceleration)
        self._galvo_acceleration_spin.setSingleStep(1)
        self._galvo_acceleration_spin.setDecimals(0)
        panel.addWidget(self._galvo_acceleration_spin)

        panel.addWidget(QWidget(), 1)

        run = QPushButton('Run')
        run.clicked.connect(self.start)
        panel.addWidget(run)

        self.stop_signal.connect(self.stop)

    def start(self):
        # create a new engine each time
        if self._engine:
            if not self._engine.done:
                _log.warning('engine is not stopped')
                return
            del self._engine
        self._engine = None

        # clear out prior volume
        if self._stack_endpoint.tensor.valid:
            with self._stack_endpoint.tensor as volume:
                volume[:] = 0
                # invalidate all B-scans
                self._stack_widget.notify_segments(range(volume.shape[0]))

        # apply parameters
        galvo_delay = self._galvo_delay_spin.value()

        for lim in self._scan_config.limits:
            lim.velocity = self._galvo_velocity_spin.value()
            lim.acceleration = self._galvo_acceleration_spin.value()

        scan = RasterScan()
        try:
            # prepare now to check scan limits
            scan.initialize(self._scan_config)
            scan.prepare()
        except RuntimeError as e:
            _log.error(f'unable to generate scan pattern: {e}')
            QMessageBox.critical(self, 'Invalid Parameters', dedent(f'''
                An error occurred while generating a scan pattern with the requested parameters:

                {e}

                Please adjust the limits and try again.

                    - For velocity limit errors, increase the velocity limit.
                    - For position limit errors, increase the acceleration limit.'''
            ))
            return

        #
        # engine setup
        #

        ec = EngineConfig()
        ec.add_acquisition(self._system._acquire, [self._system._process])
        ec.add_processor(self._system._process, [self._stack_format])
        ec.add_formatter(self._stack_format, [self._stack_endpoint])
        ec.add_io(self._system._io_out, lead_samples=round(galvo_delay * self._system._io_out.config.samples_per_second))

        ec.preload_count = self._params.preload_count
        ec.records_per_block = self._params.ascans_per_block
        ec.blocks_to_allocate = self._params.blocks_to_allocate
        ec.blocks_to_acquire = self._params.blocks_to_acquire

        ec.galvo_output_channels = len(self._system._io_out.config.channels)

        self._engine = Engine(get_logger('engine', self._params.log_level))

        # automatically stop when engine exits
        def handler(event, exc):
            if event == Engine.Event.Exit:
                self.stop_signal.emit()
            if exc:
                _log.exception(exc)
        self._engine.event_callback = handler

        self._engine.initialize(ec)
        self._engine.prepare()

        self._engine.scan_queue.append(scan)
        self._stack_format.reset()

        _log.info('starting engine')
        self._engine.start()

    def stop(self):
        if not self._engine:
            return

        _log.info('stopping engine')
        self._engine.stop()
        self._engine.wait()

        # plot shifting
        if self._stack_endpoint.tensor.valid:
            with self._stack_endpoint.tensor as volume:
                mip = cp.max(volume, axis=2)
                a = cp.mean(mip[::2], axis=0).get()
                b = cp.mean(mip[1::2], axis=0).get()

            if self._axes:
                self._axes.lines.clear()
            else:
                self._axes = self._canvas.figure.subplots()
                self._axes.set_xlabel('Sample Index')
                self._axes.set_ylabel('Intensity')
                self._axes.set_xlim(0, len(a))
                self._canvas.figure.tight_layout()

            self._axes.plot(a, 'C0')
            self._axes.plot(b, 'C1')

            self._canvas.draw()

if __name__ == '__main__':
    setup_logging()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # catch unhandled exceptions
    import traceback
    def handler(cls, ex, trace):
        traceback.print_exception(cls, ex, trace)
        app.closeAllWindows()
    sys.excepthook = handler

    # prevent Fortran routines in NumPy from catching interrupt signal
    import os
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

    # cause KeyboardInterrupt to exit the Qt application
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: app.exit())

    # regularly re-enter Python so the signal handler runs
    def keepalive(msec):
        QTimer.singleShot(msec, lambda: keepalive(msec))
    keepalive(10)

    window = GalvoTunerWindow(DEFAULT_ENGINE_PARAMS)
    window.show()

    app.exec_()

    window.stop()
