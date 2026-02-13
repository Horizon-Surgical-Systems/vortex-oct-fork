.. _how-to/interleave:

Interleave Different Scans
==========================

Complex scans, such as an aiming scan interleaved with a raster scan, are easily created using :class:`~vortex.scan.FreeformScan`.
Any combination of scan pattern generations or scan config objects can be used to generate the segments that :class:`~vortex.scan.FreeformScan` uses.
This example illustrates how to configure an aiming scan (i.e., a :class:`~vortex.scan.RadialScan` with two perpendicular segments) and insert it throughout a larger raster scan.
Each scan type is assigned different flags so the two can be distinguished later on.

#.  Import needed modules.

    .. code-block::

        from vortex.marker import Flags, ScanBoundary, VolumeBoundary
        from vortex.scan import RasterScanConfig, RadialScanConfig, FreeformScanConfig, FreeformScan

#.  Set up a :class:`~vortex.scan.RasterScanConfig` purely to generate its segments.
    Assign unique flags to distinguish from the aiming segments.

    .. code-block::

        rsc = RasterScanConfig()
        rsc.bscans_per_volume = 30

        # set unique flags for raster segments
        rsc.flags = Flags(0x1)

        raster_segments = rsc.to_segments()

#.  Now set up a :class:`~vortex.scan.RadialScanConfig` for its segments.
    Also assign unique flags to distinguish from the raster segments.

    .. code-block::

        asc = RadialScanConfig()

        asc.set_aiming()
        # add an offset just to illustrate
        asc.offset = (4, 0)

        # set unique flags for aiming segments
        asc.flags = Flags(0x2)

        aiming_segments = asc.to_segments()

#.  Combine the two scans by inserting aiming segments at 10 evenly-spaced intervals between the raster segments.

    .. code-block::

        pattern = []
        # segment indices within the raster scan to insert the aiming scan
        idx = np.linspace(0, len(raster_segments), 10, dtype=int)

        # loop over each pair of indices
        for (i, (a, b)) in enumerate(zip(idx[:-1], idx[1:])):

            # this if statement is only needed until an outstanding bug is fixed
            if i > 0:
                markers = raster_segments[a].markers
                markers.insert(0, VolumeBoundary(0, 0, False))
                markers.insert(0, ScanBoundary(0, 0))

            # append the next chunk of raster segments
            pattern += raster_segments[a:b]
            # append the aiming segments
            pattern += aiming_segments

    .. TODO update this example when the bug above is fixed

#.  :class:`~vortex.scan.FreeformScan` can now generate a scan based on this pattern.

    .. code-block::

        ffsc = FreeformScanConfig()
        ffsc.pattern = pattern
        ffsc.loop = True

        scan = FreeformScan()
        scan.initialize(ffsc)

#.  (Optional) Configure formatters with flags to route the segments of each type to different processing.

.. plot::

    from vortex.marker import Flags, ScanBoundary, VolumeBoundary
    from vortex.scan import RasterScanConfig, RadialScanConfig, FreeformScanConfig, FreeformScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    rsc = RasterScanConfig()
    rsc.flags = Flags(0x1)

    raster_segments = rsc.to_segments()

    asc = RadialScanConfig()
    asc.set_aiming()
    # add an offset just to illustrate
    asc.offset = (4, 0)
    asc.flags = Flags(0x2)

    aiming_segments = asc.to_segments()

    pattern = []
    idx = np.linspace(0, len(raster_segments), 10, dtype=int)
    for (i, (a, b)) in enumerate(zip(idx[:-1], idx[1:])):
        if i > 0:
            markers = raster_segments[a].markers
            markers.insert(0, VolumeBoundary(0, 0, False))
            markers.insert(0, ScanBoundary(0, 0))

        pattern += raster_segments[a:b]
        pattern += aiming_segments

    ffsc = FreeformScanConfig()
    ffsc.pattern = pattern
    ffsc.loop = True

    scan = FreeformScan()
    scan.initialize(ffsc)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Raster Scan and Offset Aiming Scan')
    fig.tight_layout()
