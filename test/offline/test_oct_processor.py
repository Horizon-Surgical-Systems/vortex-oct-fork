from math import pi

import pytest

import numpy as np

from vortex import Range

from util import get_compute_capability

def _round_clip_cast(xp, data, dtype):
    info = xp.iinfo(dtype)
    (lb, ub) = (info.min, info.max)
    # workaround for OverflowError for NumPy [2.0, 2.1)
    if xp.issubdtype(data.dtype, xp.integer):
        info = xp.iinfo(data.dtype)
        lb = max([info.min, lb])
        ub = min([info.max, ub])
    return data.round().clip(lb, ub).astype(dtype)

@pytest.fixture
def cpu_processor():
    try:
        from vortex.process import CPUProcessor
    except ImportError:
        pytest.skip('CPU processor not supported')

    return (CPUProcessor(), np)

@pytest.fixture
def cuda_processor():
    try:
        from vortex.process import CUDAProcessor
    except ImportError:
        pytest.skip('CUDA processor not supported')

    cupy = get_compute_capability(35)

    return (CUDAProcessor(), cupy)

@pytest.fixture(params=[1376, 2500])
def samples_per_record(request):
    return request.param

@pytest.fixture
def ascans_per_block():
    return 250

CHANNEL_INDEXING_SETUPS = [(1, 0, False), (2, 0, False), (2, 1, False), (2, 0, True), (2, 1, True)]
@pytest.fixture(params=CHANNEL_INDEXING_SETUPS, ids=[f'channels_per_sample={cps}-channel={c}-preindex={i}' for (cps, c, i) in CHANNEL_INDEXING_SETUPS])
def channel_indexing_setup(request):
    return request.param

@pytest.fixture(params=[100, 250])
def average_window(request):
    return request.param

@pytest.fixture(params=[(0, 40), (-21, 99)])
def levels(request):
    return Range(*request.param)

@pytest.fixture(params=['cpu_processor', 'cuda_processor'])
def scenario(request, samples_per_record, ascans_per_block, channel_indexing_setup):
    (proc, xp) = request.getfixturevalue(request.param)
    (channels_per_sample, channel, preindex) = channel_indexing_setup

    cfg = proc.config
    cfg.channels_per_sample = channels_per_sample
    cfg.samples_per_record = samples_per_record
    cfg.ascans_per_block = ascans_per_block
    cfg.channel = 0 if preindex else channel
    cfg.enable_ifft = False
    cfg.enable_log10 = False
    cfg.enable_square = False
    cfg.enable_magnitude = False

    xp.random.seed(1234)
    spectra = xp.random.randint(0, 10, cfg.input_shape).astype(xp.uint16)
    ascans = xp.random.randint(-128, 127, cfg.output_shape).astype(xp.int8)

    try:
        # avoid timing issues on Windows that lead to intermittent and poorly-reproducible test failures
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return (xp, proc, cfg, spectra, ascans, channel, preindex)

def test_copy(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = xp.real(spectra[..., (channel,)])
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_average(average_window, scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.average_window = average_window
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    data = spectra[..., (channel,)]
    if average_window > 0:
        data = data - xp.mean(data[-average_window:], axis=0)
    else:
        ref = data
    ref = xp.real(data)
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_resample_passthrough(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    rs = np.arange(0, spectra.shape[1])
    cfg.resampling_samples = rs
    # reduce output buffer to match
    ascans = ascans[:, rs, :].copy()

    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = xp.real(spectra[:, rs][..., (channel,)])
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_resample_skipping(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    rs = np.arange(0, spectra.shape[1], 2)
    cfg.resampling_samples = rs
    # reduce output buffer to match
    ascans = ascans[:, rs, :].copy()

    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = xp.real(spectra[:, rs][..., (channel,)])
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

CLOCK_CHANNEL_SETUPS = [(0, 1), (1, 0)]
@pytest.fixture(params=CLOCK_CHANNEL_SETUPS, ids=[f'S{s}C{c}' for (s, c) in CLOCK_CHANNEL_SETUPS])
def channels(request):
    return request.param

def test_resample_dynamic(cuda_processor, samples_per_record, ascans_per_block, channels):
    from vortex import __feature__
    if 'cuda_dynamic_resampling' not in __feature__:
        pytest.skip('dynamic resampling is not supported')

    scipy_signal = pytest.importorskip('scipy.signal')

    (spectra_channel, clock_channel) = channels
    (proc, xp) = cuda_processor

    cfg = proc.config
    cfg.channels_per_sample = 2
    cfg.samples_per_record = samples_per_record
    cfg.ascans_per_block = ascans_per_block
    cfg.channel = spectra_channel
    cfg.clock_channel = clock_channel
    cfg.enable_ifft = False
    cfg.enable_log10 = False
    cfg.enable_square = False
    cfg.enable_magnitude = False

    xp.random.seed(1234)

    # generate random linear spectra and noisy clock
    data = xp.empty(cfg.input_shape, xp.uint16)
    data[..., spectra_channel] = (100 * xp.linspace(-1, 1, samples_per_record)[None, :] + xp.random.randint(-20, 20, [ascans_per_block, 1])).round()
    data[..., clock_channel] = xp.random.randint(1000, 10000, [ascans_per_block, 1]) * (xp.sin(xp.arange(samples_per_record) / 2) / 2 + 1)[None, :] + xp.random.randint(0, 1000, [ascans_per_block, samples_per_record])

    ascans = xp.random.randint(-128, 127, cfg.output_shape).astype(xp.int8)

    proc.initialize(cfg)
    proc.next(data, ascans)

    spectra = data[..., spectra_channel].get()
    clock = data[..., clock_channel].get().astype(float)

    # NOTE: must remove mean of signal
    complex_clock = scipy_signal.hilbert(clock - np.mean(clock, axis=1, keepdims=True))

    # unwrap the phase
    phase_diff = np.diff(np.angle(complex_clock), axis=1)
    phase_offset = np.zeros_like(phase_diff)
    phase_offset[phase_diff < -pi] = 2*pi
    phase_offset[phase_diff > pi] = -2*pi

    phase = np.zeros(clock.shape, dtype=float)
    phase[:, 1:] = np.cumsum(phase_diff + phase_offset, axis=1)

    # normalization
    phase_max = np.max(phase, axis=1)
    phase_normalized = phase * ((samples_per_record - 1) / phase_max[:, None])
    idxs = np.arange(samples_per_record)

    # perform resampling
    resampling = np.zeros_like(clock)
    ref = np.zeros(ascans.shape)
    for (i, (k, s)) in enumerate(zip(phase_normalized, spectra)):
        resampling[i, :] = np.interp(idxs, k, idxs)
        ref[i, :, 0] = np.interp(resampling[i, :], idxs, s)

    # ignore final values and allow extra tolerance for more rounding errors
    xp.testing.assert_allclose(ascans[:, :-5], _round_clip_cast(xp, ref, xp.int8)[:, :-5], atol=3)

def test_filter(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    sf = np.random.random((spectra.shape[1]))
    cfg.spectral_filter = sf

    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = xp.real(spectra[..., (channel,)] * xp.asanyarray(sf[None, :, None]))
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_ifft(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_ifft = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = xp.real(xp.fft.ifft(spectra[..., (channel,)].astype(float), axis=1))
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_log10(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_log10 = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = 10 * xp.log10(spectra[..., (channel,)].astype(float) + xp.finfo(float).eps)
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_square(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_square = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = spectra[..., (channel,)].astype(float)**2
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_log10_square(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_log10 = True
    cfg.enable_square = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = 20 * xp.log10(spectra[..., (channel,)].astype(float) + xp.finfo(float).eps)
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_log10_real(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_log10 = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = 10 * xp.log10(spectra[..., (channel,)].astype(float) + xp.finfo(float).eps)
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_log10_magnitude(scenario):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.enable_log10 = True
    cfg.enable_magnitude = True
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    ref = 10 * xp.log10(spectra[..., (channel,)].astype(float) + xp.finfo(float).eps)
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

def test_levels(scenario, levels):
    (xp, proc, cfg, spectra, ascans, channel, preindex) = scenario

    cfg.levels = levels
    proc.initialize(cfg)

    if preindex:
        proc.next(spectra[..., channel], ascans)
    else:
        proc.next(spectra, ascans)

    scale = 255 / levels.length
    offset = -levels.min * scale - 128

    ref = scale * xp.abs(spectra[..., (channel,)]).astype(xp.float32) + offset
    xp.testing.assert_allclose(ascans, _round_clip_cast(xp, ref, xp.int8), atol=1)

if __name__ == '__main__':
    pytest.main()
