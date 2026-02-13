.. _how-to/repeated-bidirectional:

Repeated Scan with Bidirectional Segments
=========================================

A repeated scan that takes advantage of bidirectional segments for better scan efficiency is obtained using :class:`~vortex.scan.RepeatedPattern`.
It can be used directly for a custom scan or access conveniently through :class:`RepeatedRasterScan <vortex.scan.RepeatedRadialScan>` or :class:`~vortex.scan.RepeatedRadialScan`.
The sample below uses :class:`~vortex.scan.RepeatedRasterScan` to illustrate a bidirectional repeated scan.

#.  Import and instantiate :class:`~vortex.scan.RepeatedRasterScanConfig`.

    .. code-block::

        from vortex.scan import RepeatedRasterScanConfig, RepeatedRasterScan
        cfg = RepeatedRasterScanConfig()

#.  Configure bidirectional segments.

    .. code-block::

        cfg.bidirectional_segments = True

    Each segment within a bidirectional scan normally has a fixed direction.
    If :data:`~vortex.scan.RepeatedPattern.repeat_period` is odd, one segment may alternate directions.

#.  Configure repetition options using the fields inherited from :class:`~vortex.scan.RepeatedPattern`.
    :data:`~vortex.scan.RepeatedPattern.repeat_count` determines the number of repeats for each segment whereas :data:`~vortex.scan.RepeatedPattern.repeat_period` determines the number of segments between repeats.

    .. code-block::

        cfg.repeat_count = 2
        cfg.repeat_period = 4

    This configuration will repeat every segment twice (:data:`~vortex.scan.RepeatedPattern.repeat_count`) in groups of four (:data:`~vortex.scan.RepeatedPattern.repeat_period`).
    Each group of four will complete its repetitions before the next group of four starts.

#.  Initialize the :class:`~vortex.scan.RepeatedRasterScan` as usual.

    .. code-block::

        scan = RepeatedRasterScan()
        scan.initialize(cfg)

.. plot::

    from math import pi
    from vortex.scan import RepeatedRasterScanConfig, RepeatedRasterScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    cfg = RepeatedRasterScanConfig()
    cfg.bscans_per_volume = 8
    cfg.angle = pi / 2

    cfg.bidirectional_segments = True

    cfg.repeat_count = 2
    cfg.repeat_period = 4

    scan = RepeatedRasterScan()
    scan.initialize(cfg)

    path = scan.scan_buffer()

    _, ax = plt.subplots(constrained_layout=True)
    ax.plot(path)
    ax.set_title('XY Position vs Sample')
    ax.set_ylabel('position')
    ax.set_xlabel('sample')

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(path, scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Scan Pattern in XY Plane')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    fig.tight_layout()
