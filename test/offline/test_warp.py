import pytest
import numpy
from numpy.testing import assert_allclose

from vortex import Range
from vortex.scan import RasterScan, warp
from vortex_tools.scan import partition_segments_by_activity

@pytest.fixture
def waveform():
    return numpy.column_stack([numpy.linspace(-1, 1, 100)]*2)

def test_null_forward(waveform):
    w = warp.NoWarp()

    out = w.forward(waveform)
    assert_allclose(out, waveform)

def test_null_inverse(waveform):
    w = warp.NoWarp()

    out = w.inverse(waveform)
    assert_allclose(out, waveform)

def test_angular_forward(waveform):
    w = warp.Angular()

    out = w.forward(waveform)
    assert_allclose(out, waveform * w.factor)

def test_angular_inverse(waveform):
    w = warp.Angular()

    out = w.inverse(waveform)
    assert_allclose(out, waveform / w.factor)

def test_telecentric_forward(waveform):
    w = warp.Telecentric()

    out = w.forward(waveform)
    assert_allclose(out, numpy.tan(waveform * w.scale) * w.galvo_lens_spacing)

def test_telecentric_inverse(waveform):
    w = warp.Telecentric()

    out = w.inverse(waveform)
    assert_allclose(out, numpy.arctan2(waveform, w.galvo_lens_spacing) / w.scale)

def test_telecentric_raster():
    rs = RasterScan()
    cfg = rs.config
    cfg.warp = warp.Telecentric()
    cfg.warp.galvo_lens_spacing = 100
    rs.initialize(cfg)

    (active, _) = partition_segments_by_activity(rs.scan_markers(), rs.scan_buffer())
    galvo_waypoints = numpy.stack(active)

    sample_waypoints = cfg.warp.forward(galvo_waypoints)

    assert_allclose(sample_waypoints[..., 0].min(), cfg.volume_extent.min)
    assert_allclose(sample_waypoints[..., 0].max(), cfg.volume_extent.max)
    assert_allclose(sample_waypoints[..., 1].min(), cfg.bscan_extent.min)
    assert_allclose(sample_waypoints[..., 1].max(), cfg.bscan_extent.max)

if __name__ == '__main__':
    pytest.main()
