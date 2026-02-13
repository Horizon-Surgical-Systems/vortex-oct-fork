import sys
import logging
from pathlib import Path

from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from qtpy.QtGui import QIcon

from vortex import Range, get_console_logger as get_logger
from vortex.marker import Flags
from vortex.scan import RasterScanConfig, RasterScan, RadialScanConfig, RadialScan, limits
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice

from vortex_tools.ui.display import RasterEnFaceWidget, CrossSectionImageWidget

# hack to simplify running demos
sys.path.append(Path(__file__).parent.parent.as_posix())
from _common.engine import setup_logging, StandardEngineParams, DEFAULT_ENGINE_PARAMS, BaseEngine

RASTER_FLAGS = Flags(0x1)
AIMING_FLAGS = Flags(0x2)

_log = logging.getLogger(__name__)

def configure_scan(cfg, params: StandardEngineParams):
    # resolution
    cfg.bscans_per_volume = params.bscans_per_volume
    cfg.ascans_per_bscan = params.ascans_per_bscan

    # physical diemnsions
    cfg.bscan_extent = Range(-params.scan_dimension, params.scan_dimension)
    cfg.volume_extent = Range(-params.scan_dimension, params.scan_dimension)

    # timing
    cfg.samples_per_second = params.swept_source.triggers_per_second

    # morphology
    cfg.bidirectional_segments = params.bidirectional
    cfg.bidirectional_volumes = params.bidirectional
    cfg.loop = True

def make_raster_scan(params: StandardEngineParams):
    cfg = RasterScanConfig()

    configure_scan(cfg, params)

    # formatting
    cfg.flags = RASTER_FLAGS

    scan = RasterScan()
    scan.initialize(cfg)

    return scan

def make_aiming_scan(params: StandardEngineParams):
    cfg = RadialScanConfig()

    configure_scan(cfg, params)
    cfg.set_aiming()

    # formatting
    cfg.flags = AIMING_FLAGS

    scan = RadialScan()
    scan.initialize(cfg)

    return scan

def make_format_for_scan(scan, slice: SimpleSlice, params: StandardEngineParams):
    # set up format planner
    cfg = FormatPlannerConfig()
    cfg.segments_per_volume = scan.config.segments_per_volume
    cfg.records_per_segment = scan.config.samples_per_segment
    cfg.adapt_shape = False

    cfg.mask = scan.config.flags
    fmt = FormatPlanner(get_logger('format', params.log_level))
    fmt.initialize(cfg)

    # set up format executor
    cfg = StackFormatExecutorConfig()
    cfg.sample_slice = slice

    exe = StackFormatExecutor()
    exe.initialize(cfg)

    return (fmt, exe)

class MultiDisplayWidget(QWidget):
    def __init__(self, scan, slice, params, **kwargs):

        super().__init__(**kwargs)

        self._scan = scan

        # allocate an endpoint to store the formatted data
        (self._format, exe) = make_format_for_scan(scan, slice, params)
        self._endpoint = StackDeviceTensorEndpoint(exe, (scan.config.bscans_per_volume, scan.config.ascans_per_bscan, exe.config.sample_slice.count()), get_logger('endpoint', params.log_level))

        # register completion callbacks
        self._endpoint.aggregate_segment_callback = self.notify_segments

    def notify_segments(self, idxs):
        # distribute to all child widgets
        for widget in self.children():
            notify = getattr(widget, 'notify_segments', None)
            if notify:
                notify(idxs)

    def clear_volume(self):
        with self._endpoint.tensor as volume:
            volume[:] = 0
            # invalidate all B-scans
            self.notify_segments(range(volume.shape[0]))

    @property
    def scan(self):
        return self._scan
    @property
    def format(self):
        return self._format
    @property
    def endpoint(self):
        return self._endpoint

class RasterScanDisplayWidget(MultiDisplayWidget):
    def __init__(self, slice, params, **kwargs):
        scan = make_raster_scan(params)

        super().__init__(scan, slice, params, **kwargs)

        # display MIP and B-scan
        layout = QHBoxLayout(self)
        layout.addWidget(RasterEnFaceWidget(self.endpoint))
        layout.addWidget(CrossSectionImageWidget(self.endpoint))

    @property
    def name(self):
        return 'Raster'

class AimingScanDisplayWidget(MultiDisplayWidget):
    def __init__(self, slice, params, **kwargs):
        scan = make_aiming_scan(params)

        super().__init__(scan, slice, params, **kwargs)

        # display every B-scan in volume
        layout = QHBoxLayout(self)
        for idx in range(scan.config.bscans_per_volume):
            layout.addWidget(CrossSectionImageWidget(self.endpoint, fixed=idx))

    @property
    def name(self):
        return 'Aiming'

class System(BaseEngine):
    def __init__(self, params: StandardEngineParams):
        super().__init__(params)

    def configure(self, params, scan_widgets):
        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [w.format for w in scan_widgets])
        for w in scan_widgets:
            ec.add_formatter(w.format, [w.endpoint])
        ec.add_io(self._io_out, lead_samples=round(params.galvo_delay * self._io_out.config.samples_per_second))
        ec.add_io(self._strobe)

        ec.preload_count = params.preload_count
        ec.records_per_block = params.ascans_per_block
        ec.blocks_to_allocate = params.blocks_to_allocate
        ec.blocks_to_acquire = params.blocks_to_acquire

        ec.galvo_output_channels = len(self._io_out.config.channels)
        ec.galvo_input_channels = 0

        engine = Engine(get_logger('engine', params.log_level))
        self._engine = engine

        engine.initialize(ec)
        engine.prepare()

class EngineWindow(QWidget):
    def __init__(self, params: StandardEngineParams, **kwargs):
        super().__init__(**kwargs)

        #
        # engine setup - initial
        #

        self._system = System(params)

        #
        # UI setup
        #

        logo = QIcon((Path(__file__).parent.parent.parent / 'doc' / 'vortex-v0.svg').as_posix())

        self.setWindowTitle('Vortex - Tutorial 06')
        self.setWindowIcon(logo)
        self.resize(720, 720)

        layout = QVBoxLayout(self)
        self.setup_header(layout, logo)

        n = self._system._process.config.samples_per_ascan
        slice = SimpleSlice(0, n // 2)
        # slice = SimpleSlice(n // 2, n)

        raster_widget = RasterScanDisplayWidget(slice, params)
        aiming_widget = AimingScanDisplayWidget(slice, params)
        self._scan_widgets = [raster_widget, aiming_widget]
        for w in self._scan_widgets:
            layout.addWidget(w, 1)

        panel = QHBoxLayout()
        layout.addLayout(panel)
        start_button = QPushButton('Start')
        start_button.clicked.connect(self.start_engine)
        panel.addWidget(start_button)

        stop_button = QPushButton('Stop')
        stop_button.clicked.connect(self.stop_engine)
        panel.addWidget(stop_button)

        scan_buttons = [(QPushButton(w.name), w) for w in self._scan_widgets]
        for (b, w) in scan_buttons:
            b.clicked.connect(lambda _, w=w: self.change_scan(w))
            panel.addWidget(b)

        #
        # engine setup - final
        #

        self._system.configure(params, self._scan_widgets)

        # set initial scan
        self._active_scan = None
        self.change_scan(aiming_widget)

    def setup_header(self, layout, logo):
        header = QHBoxLayout()
        layout.addLayout(header)

        header.addStretch()

        header_logo = QLabel()
        header_logo.setPixmap(logo.pixmap(50, 50))
        header.addWidget(header_logo)

        header_text = QLabel('Tutorial 6: UI Integration with Scan Switching')
        header_text.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header_text.setStyleSheet('font-size: 16pt;')
        header.addWidget(header_text)

        header.addStretch()

    def change_scan(self, widget):
        if self._active_scan is widget:
            _log.warning(f'scan {widget.name} is already active')
            return

        _log.info(f'change to scan {widget.name}')
        self._active_scan = widget
        self._system._engine.scan_queue.interrupt(widget.scan)

        for w in self._scan_widgets:
            w.setVisible(w is widget)
        widget.clear_volume()

    def start_engine(self):
        # check if engine is startable
        if not self.engine.done:
            _log.warning('engine is already started')
            return
        self.engine.wait()

        # clear formatting state
        for w in self._scan_widgets:
            w.clear_volume()
            w.format.reset()

        # restart the scan
        self.engine.scan_queue.reset()
        self.engine.scan_queue.interrupt(self._active_scan.scan)

        # start the engine
        _log.info('starting engine')
        self.engine.start()

    def stop_engine(self):
        # check if engine is stoppable
        if self.engine.done:
            _log.warning('engine is already stopped')
            return

        # request that the engine stop
        # NOTE: that the engine will complete pending blocks in the background
        _log.info('requesting engine stop')
        self.engine.stop()

    @property
    def engine(self):
        return self._system._engine

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

    window = EngineWindow(DEFAULT_ENGINE_PARAMS)
    window.show()

    app.exec_()

    window.stop_engine()
    window.engine.wait()
