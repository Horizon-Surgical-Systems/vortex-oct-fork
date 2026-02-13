.. _module/scan:

Scan
####

.. warning::

    This document is under construction.

-   :class:`~vortex.scan.RasterScan`
-   :class:`~vortex.scan.RadialScan`
-   :class:`~vortex.scan.RepeatedRasterScan`
-   :class:`~vortex.scan.RepeatedRadialScan`
-   :class:`~vortex.scan.FreeformScan`

Scan Patterns
=============

One of *vortex*'s most powerful features is its scan pattern generation capability.
For *vortex*, a scan pattern pairs a **trajectory**, a uniformly-spaced time-series of waypoints across the imaging scene, and set of **markers** that annotate the trajectory with useful information.
The core scan execution components in *vortex* enforce no further structure on scan patterns.
There is no concept of a raster scan or a spiral scan; *vortex* works only with trajectories and markers.

This design strategy decouples the scanning and acquisition from artificial restrictions such as equal lengths for all segments or in order acquisition of a volume.
*vortex* defers responsibility for formatting the data into a meaningful structure (e.g., a rectangular volume) until the processing stage.
That said, *vortex* offers an extensive set of utility classes to generate and format common scan patterns, but you are free to design your own pattern by supplying the waypoints and markers.

Time-Optimal Trajectory Generation
----------------------------------

*vortex* incorporates the time-optimal motion planner Reflexxes_.
High-level scan pattern generation classes use Reflexxes_ to calculate optimally fast corners and flyback, once provided velocity and acceleration limits for the scanner hardware.
With properly calibrated dynamics limits, this allows *vortex* to execute the scan pattern as fast as possible without introducing distortion or artifact.
*vortex* aims to bring the scan hardware to the desired position in the imaging scene on time, every time.

*vortex* uses motion planning to generate the inactive portion of the scan pattern automatically.
There is no need to manually specify or calculate flyback or to intentionally round corners in the scan pattern.
*vortex* will generate an on-the-fly inactive segment that matches the entry and exit velocities of each active segment.
This capability significantly reduces distortion at pattern edges because the scanner hardware enters the active region at the necessary velocity.
That said, *vortex* offers alternate :ref:`inactive policies <inactive-policy>` should your application require a different approach.

.. _Reflexxes: https://github.com/Reflexxes/RMLTypeII

Segmented Scans
---------------

The set of positions through which a scan must pass is called its **waypoints**.
The waypoints object is a list of lists of points, where each list of points describes the path of a single segment.
A :class:`~vortex.scan.RasterScan`, for example, would have a rectangular grid as its waypoints, such as plotted below.
In the plot below, each "x" is a waypoint at which data collection occurs.

.. plot::

    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RasterWaypoints

    wpts = RasterWaypoints()
    wpts.volume_extent = Range.symmetric(1)
    wpts.bscan_extent = Range.symmetric(1)
    wpts.samples_per_segment = 20
    wpts.segments_per_volume = 10

    xy = wpts.to_waypoints()

    fig, ax = plt.subplots(constrained_layout=True)

    for segment in xy:
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    ax.set_title('Raster Scan Waypoints')
    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.axis('equal')

The order and direction in which to execute segments is called the scan's **pattern**.
Patterns can request features such as bidirectional segments or repetition of segments.
The line in the plots below illustrates the time-optimal scan trajectory that passes through the waypoints from above using two different patterns.

.. plot::

    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RasterScan, RasterScanConfig

    cfg = RasterScanConfig()
    cfg.volume_extent = Range.symmetric(1)
    cfg.bscan_extent = Range.symmetric(1)
    cfg.samples_per_segment = 20
    cfg.segments_per_volume = 10
    for limit in cfg.limits:
        limit.velocity *= 10
        limit.acceleration *= 40
    cfg.loop = True

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfgs.append(cfg.copy())
    names.append('Raster Scan - Unidirectional')

    cfgs.append(cfg.copy())
    cfgs[-1].bidirectional_segments = True
    names.append('Raster Scan - Bidirectional')

    for (name, cfg, ax) in zip(names, cfgs, axs):
        scan = RasterScan()
        scan.initialize(cfg)

        for segment in cfg.to_waypoints():
            ax.plot(segment[:, 0], segment[:, 1], 'x')

        path = scan.scan_buffer()
        ax.plot(path[:, 0], path[:, 1], fg, lw=1, zorder=-1)

        ax.set_xlabel('x (au)')
        ax.set_ylabel('y (au)')
        ax.set_title(name)

Markers
-------

Markers encode the high-level structure of the scan pattern and additional useful information.
Without them, the trajectory is merely a sequence of waypoints.
Commonly, markers denote the status of a waypoint (active or inactive) and the boundaries between substructures within the scan pattern.
For example, :class:`~vortex.marker.SegmentBoundary` indicates a transition point between segments within the scan.
More complex uses include :class:`~vortex.marker.Event`, which allow the user to associate custom information with a waypoint.

*vortex* supports multiplexing multiple scan patterns or markers through the use of **flags**, implemented by :class:`~vortex.marker.Flags`.
This is a bitmask which processing code tests to determine if it should act on a marker.
A typical usage scenario is to insert an aiming scan within a raster scan and then subject the data acquired during each scan to separate processing or display (:ref:`see example <how-to/interleave>`).

Components
==========

.. tip::

    For examples of how to use these scan components, download the source code that generated a given figure by clicking the "code" link immediately below it.

Waypoints
---------

.. class:: vortex.scan.XYWaypoints

    Generate waypoints on a 2D uniform grid of waypoints defined by its origin, rotation about that origin, physical extents, and dimension in samples.

    This class is present in C++ but is abstracted away in the Python bindings.
    It cannot be instantiated from Python.

    .. property:: samples_per_segment
        :type: int

        Number of samples :math:`m` within each segment.
        Default is ``100``.

    .. property:: segments_per_volume
        :type: int

        Number of segments :math:`n` in each volume.
        Default is ``100``.

    .. property:: ascans_per_bscan
        :type: int

        Alias for :data:`samples_per_segment`

    .. property:: bscans_per_volume
        :type: int

        Alias for :data:`segments_per_volume`.

    .. property:: shape
        :type: list[int]

        Scan shape as [:data:`segments_per_volume`, :data:`samples_per_segment`].

    .. property:: bscan_extent
        :type: Range

        Alias for :data:`segment_extent`.

    .. property:: extents
        :type: list[Range]

        Scan extents as [:data:`volume_extent`, :data:`segment_extent`].

    .. property:: offset
        :type: tuple[float]

        Offset :math:`(x_o, y_o)` of the scan origin in :math:`x` and :math:`y`, respectively.
        Default is ``(0, 0)``.

    .. property:: angle
        :type: float

        Counter-clockwise rotation :math:`\theta` about the scan origin (:data:`offset`) in radians.
        Default is ``0``.

    .. property:: warp
        :type: [~vortex.scan.warp.NoWarp | ~vortex.scan.warp.AngularWarp | ~vortex.scan.warp.TelecentricWarp]

        Warp to apply to waypoints, which may transform the units of the scan waypoints.
        Default is ``NoWarp()``.

    .. method:: to_waypoints()

        Generate an :math:`n \times m \times 2` grid of waypoints :math:`W_{ij} = (x_i, y_i)` for :math:`i \in [0, n-1]` and :math:`j \in [0, m-1]`.
        The specific waypoint positions are determined by the base class.

        :returns numpy.ndarray[numpy.float64]: The waypoints 3D array with shape [:data:`segments_per_volume`, :data:`samples_per_segment`, ``2``], where the channels are :math:`x`` and :math:`y`` positions.

Raster
^^^^^^

.. plot::

    from math import pi, cos, sin

    import numpy as np
    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RasterScan, RasterScanConfig
    from vortex_tools.scan import plot_annotated_waveforms_space

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfg = RasterScanConfig()
    cfg.segment_extent = Range.symmetric(1)
    cfg.segments_per_volume = 10
    cfg.samples_per_segment = 50
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True

    # change offset
    names.append('Offset')
    cfgs.append(cfg.copy())
    cfgs[-1].offset = (1, 0)

    # change extent
    names.append('Extent')
    cfgs.append(cfg.copy())
    cfgs[-1].volume_extent = Range(2, -2)
    cfgs[-1].segment_extent = Range(0, 1)

    # change shape
    names.append('Shape')
    cfgs.append(cfg.copy())
    cfgs[-1].segments_per_volume = 5

    # change rotation
    names.append('Angle')
    cfgs.append(cfg.copy())
    cfgs[-1].angle = pi / 6

    for (name, cfg, ax) in zip(names, cfgs, axs.flat):
        scan = RasterScan()
        scan.initialize(cfg)
        plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg, axes=ax)
        ax.set_title(name)

        if not np.allclose(cfg.offset, (0, 0)):
            ax.plot([0, cfg.offset[0]], [0, cfg.offset[1]], 'ro-', zorder=20)
        if cfg.angle != 0:
            ax.plot([1, 0, cos(cfg.angle)], [0, 0, sin(cfg.angle)], 'ro-', zorder=20)

.. class:: vortex.scan.RasterWaypoints

    Base: :class:`~vortex.scan.XYWaypoints`

    Generate waypoints on a :math:`n \times m` rectangular grid with :math:`n` segments per volume and :math:`m` samples per segment.
    The volume direction is along the positive the :math:`x`-axis whereas the segment or B-scan direction is along the positive :math:`y`-axis.
    The physical extents of the waypoints have no required units and may be interpreted as linear (e.g., millimeters) or angular (e.g., radians or degrees).
    The waypoint positions :math:`W_{ij} = (x_i, y_j)` as described mathematically below.

    .. math::

        \begin{align}
            u_i &= \left(\frac{i}{n - 1}\right) x_\mathrm{min} + \left(1 - \frac{i}{n-1}\right) x_\mathrm{max} &
            v_j &= \left(\frac{j}{m - 1}\right) y_\mathrm{min} + \left(1 - \frac{j}{m-1}\right) y_\mathrm{max} \\
            x_i &= u_i \cos \theta - v_i \sin \theta + x_o &
            y_j &= u_j \sin \theta + v_j \cos \theta + y_o
        \end{align}

    The parameters that define :math:`W_{ij}` are adjusted via the the following properties and those inherited from :class:`~vortex.scan.XYWaypoints`.

    .. property:: segment_extent
        :type: Range

        Unitless minimum :math:`y_\mathrm{min}` and maximum :math:`y_\mathrm{max}` position of each segment.
        To flip the segment direction, set a negative maximum and a positive minimum.
        No requirement for symmetry.
        Default is ``Range(-2, 2)``.

    .. property:: volume_extent
        :type: Range

        Unitless minimum :math:`x_\mathrm{min}` and maximum :math:`x_\mathrm{max}` position of each volume.
        To flip the volume direction, set a negative maximum and a positive minimum.
        No requirement for symmetry.
        Default is ``Range(-1, 1)``.

Radial
^^^^^^

.. plot::

    from math import pi, cos, sin

    import numpy as np
    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RadialScan, RadialScanConfig
    from vortex_tools.scan import plot_annotated_waveforms_space

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfg = RadialScanConfig()
    cfg.segment_extent = Range.symmetric(1)
    cfg.segments_per_volume = 10
    cfg.samples_per_segment = 50
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True

    # change offset
    names.append('Offset')
    cfgs.append(cfg.copy())
    cfgs[-1].offset = (1, 0)

    # change extent
    names.append('Extent')
    cfgs.append(cfg.copy())
    cfgs[-1].volume_extent = Range(0, 5)
    cfgs[-1].segment_extent = Range(0.5, 1)

    # change shape
    names.append('Shape')
    cfgs.append(cfg.copy())
    cfgs[-1].segments_per_volume = 5

    # change rotation
    names.append('Angle')
    cfgs.append(cfg.copy())
    cfgs[-1].angle = pi / 6

    for (name, cfg, ax) in zip(names, cfgs, axs.flat):
        scan = RadialScan()
        scan.initialize(cfg)
        plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg, axes=ax)
        ax.set_title(name)

        if not np.allclose(cfg.offset, (0, 0)):
            ax.plot([0, cfg.offset[0]], [0, cfg.offset[1]], 'ro-', zorder=20)
        if cfg.angle != 0:
            ax.plot([1, 0, cos(cfg.angle)], [0, 0, sin(cfg.angle)], 'ro-', zorder=20)

    x = np.abs(axs.flat[0].get_xlim()).max()
    axs.flat[0].set_xlim(-x, x)

.. class:: vortex.scan.RadialWaypoints

    Base: :class:`~vortex.scan.XYWaypoints`

    Generate waypoints on a :math:`n \times m` polar grid with :math:`n` segments per volume and :math:`m` samples per segment.
    The volume direction is along the positive the :math:`\theta`-axis whereas the segment or B-scan direction is along the positive :math:`r`-axis.
    The physical extents of the waypoints have no required units and may be interpreted as linear (e.g., millimeters) or angular (e.g., radians or degrees).
    The waypoint positions :math:`W_{ij} = (x_i, y_j)` as described mathematically below.

    .. math::

        \begin{align}
            u_i &= \left(\frac{i}{n - 1}\right) \theta_\mathrm{min} + \left(1 - \frac{i}{n-1}\right) \theta_\mathrm{max} &
            v_j &= \left(\frac{j}{m - 1}\right) r_\mathrm{min} + \left(1 - \frac{j}{m-1}\right) r_\mathrm{max} \\
            x_i &= v_i \cos ( u_i + \theta ) + x_o &
            y_j &= v_j \sin ( u_j + \theta ) + y_o
        \end{align}

    The parameters that define :math:`W_{ij}` are adjusted via the the following properties and those inherited from :class:`~vortex.scan.XYWaypoints`.

    .. property:: segment_extent
        :type: Range

        Unitless minimum :math:`r_\mathrm{min}` and maximum :math:`r_\mathrm{max}` position of each segment.
        To flip the segment direction, set a negative maximum and a positive minimum.
        No requirement for symmetry.
        Default is ``Range(-2, 2)``.

    .. property:: volume_extent
        :type: Range

        Unitless minimum :math:`\theta_\mathrm{min}` and maximum :math:`\theta_\mathrm{max}` position of each volume.
        To flip the volume direction, set a negative maximum and a positive minimum.
        No requirement for symmetry.
        Default is ``Range(0, pi)``.

    .. method:: set_half_evenly_spaced(n)

        Adjust :attr:`volume_extent` and :attr:`segments_per_volume` yield ``n`` evenly spaced segments over 180 degrees, omitting the final segment.
        For segments that span the origin, this yields a scan that covers the full circle without overlap.

        :param int n:
            Number of segments to use.

    .. method:: set_evenly_spaced(n)

        Adjust :attr:`volume_extent` and :attr:`segments_per_volume` yield ``n`` evenly spaced segments over 360 degrees, omitting the final segment.

        :param int n:
            Number of segments to use.

    .. method:: set_aiming()

        Adjust :attr:`volume_extent` and :attr:`segments_per_volume` to yield two orthogonal segments.
        Internally calls :meth:`set_half_evenly_spaced` with ``n = 2``.

Patterns
--------

Sequential
^^^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RasterScan, RasterScanConfig
    from vortex_tools.scan import plot_annotated_waveforms_space

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfg = RasterScanConfig()
    cfg.segment_extent = Range.symmetric(1)
    cfg.volume_extent = Range.symmetric(2)
    cfg.segments_per_volume = 10
    cfg.samples_per_segment = 50
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True

    names.append('Default')
    cfgs.append(cfg.copy())

    names.append('Segments')
    cfgs.append(cfg.copy())
    cfgs[-1].bidirectional_segments = True

    names.append('Volumes')
    cfgs.append(cfg.copy())
    cfgs[-1].bidirectional_volumes = True

    names.append('Segments + Volumes')
    cfgs.append(cfg.copy())
    cfgs[-1].bidirectional_segments = True
    cfgs[-1].bidirectional_volumes = True

    for (name, cfg, ax) in zip(names, cfgs, axs.flat):
        scan = RasterScan()
        scan.initialize(cfg)
        plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg, axes=ax)
        ax.set_title(name)

.. class:: vortex.scan.SequentialPattern

    Visit segments in sequential order with optional bidirectionality.

    .. property:: bidirectional_segments
        :type: bool

        If ``True``, flip the direction of every odd-indexed segment.
        Odd-indexed segments are marked as reversed for unflipping during formatting.
        If ``False``, maintain the direction of each segment.
        Default is ``False``.

    .. property:: bidirectional_volumes
        :type: bool

        If ``True``, produce a scan with one volume with the segments in order followed by a second volume using the reverse segment order.
        The second volume's segments are marked with the physically-based destination index for reordering during formatting.
        If ``False``, produce a scan with one volume with he segments in order.
        Default is ``False``.

    .. property:: flags
        :type: ~vortex.marker.Flags

        The flags to apply to these segments.
        Default is ``Flags()``.

    .. method:: to_pattern(waypoints)

        Generate a scan pattern from the given waypoints.

        :param numpy.ndarray[float] | List[numpy.ndarray[float]] waypoints:
            Active sample positions as a 3D array or list of active segments as 2D arrays.
            If a 3D array is provided, the segments are extracted along the first axis.

        :return List[~vortex.scan.Segment]:
            The scan pattern as a list of segments.

Repeated
^^^^^^^^

.. plot::

    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RepeatedRasterScan, RepeatedRasterScanConfig
    from vortex_tools.scan import plot_annotated_waveforms_space

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfg = RepeatedRasterScanConfig()
    cfg.segment_extent = Range.symmetric(1)
    cfg.volume_extent = Range.symmetric(2)
    cfg.segments_per_volume = 6
    cfg.samples_per_segment = 50
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True

    names.append('Standard')
    cfgs.append(cfg.copy())
    cfgs[-1].repeat_count = 1
    cfgs[-1].repeat_period = 1

    names.append('Repeat')
    cfgs.append(cfg.copy())

    names.append('Even Repeat Period + Bidirectional')
    cfgs.append(cfg.copy())
    cfgs[-1].repeat_period = 2
    cfgs[-1].bidirectional_segments = True

    names.append('Odd Repeat Period + Bidirectional')
    cfgs.append(cfg.copy())
    cfgs[-1].repeat_period = 3
    cfgs[-1].bidirectional_segments = True

    for (name, cfg, ax) in zip(names, cfgs, axs.flat):
        scan = RepeatedRasterScan()
        scan.initialize(cfg)
        plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg, axes=ax)
        ax.set_title(name)

.. class:: vortex.scan.RepeatedPattern

    Repeat segments with specified repetition count and period.

    .. property:: repeat_count
        :type: int

        Number of times to repeat each segment.
        Default is ``3``.

    .. property:: repeat_period
        :type: int

        Number of segments to execute in order before repeating.
        Default is ``2``.

    .. property:: repeat_strategy
        :type: [~vortex.scan.RepeatOrder | ~vortex.scan.RepeatPack | ~vortex.scan.RepeatFlags]

        The strategy for handling repeated segments in memory.

        -   :class:`~vortex.scan.RepeatOrder`:
            Scan is performed ``ABCABCABC`` and is stored in memory as ``ABCABCABC``.

        -   :class:`~vortex.scan.RepeatPack`:
            Scan is performed ``ABCABCABC`` but is stored in memory as ``AAABBBCCC``.

        -   :class:`~vortex.scan.RepeatFlags`:
            Each repeat is marked with different flags, allowing the user to route them to different processing.

        Default is ``RepeatPack()``.

    .. property:: bidirectional_segments
        :type: bool

        If ``True``, flip the direction of every odd-indexed segment.
        Odd-indexed segments are marked as reversed for unflipping during formatting.
        If ``False``, maintain the direction of each segment.
        Default is ``False``.

        .. caution::

            An odd repeat period will cause repetitions of a given segment to alternate directions.

        .. seealso::

            See :ref:`how-to/repeated-bidirectional` for an example.

    .. method:: to_pattern(waypoints)

        Generate a scan pattern from the given waypoints.

        :param numpy.ndarray[float] | List[numpy.ndarray[float]] waypoints:
            Active sample positions as a 3D array or list of active segments as 2D arrays.
            If a 3D array is provided, the segments are extracted along the first axis.

        :return List[~vortex.scan.Segment]:
            The scan pattern as a list of segments.

.. _inactive-policy:

Inactive Policy
---------------

The strategy for generating the inactive segments of a :class:`~vortex.scan.SegmentedScan` is encapsulated by an inactive policy object.
The available inactive policies are described below.

.. plot::

    from matplotlib import pyplot as plt

    from vortex import Range
    from vortex.scan import RasterScanConfig, RasterScan, inactive_policy
    from vortex_tools.scan import plot_annotated_waveforms_space, partition_segments_by_activity

    fg = plt.rcParams['lines.color']
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
    cfgs = []
    names = []

    cfg = RasterScanConfig()
    cfg.segment_extent = Range.symmetric(1)
    cfg.volume_extent = Range.symmetric(2)
    cfg.segments_per_volume = 6
    cfg.samples_per_segment = 50
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True

    names.append('Minimum Dynamic Limited')
    cfgs.append(cfg.copy())
    cfgs[-1].inactive_policy = inactive_policy.MinimumDynamicLimited()

    names.append('Fixed Dynamic Limited')
    cfgs.append(cfg.copy())
    cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

    names.append('Fixed Linear')
    cfgs.append(cfg.copy())
    cfgs[-1].inactive_policy = inactive_policy.FixedLinear()

    for (name, cfg, ax) in zip(names, cfgs, axs.flat):
        scan = RasterScan()
        scan.initialize(cfg)
        plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg, axes=ax)
        ax.set_title(name)

    for ax in axs.flat[len(names):]:
        fig.delaxes(ax)

.. class:: vortex.scan.inactive_policy.MinimumDynamicLimited

    Generate minimum-duration inactive segments that satisfy the scan's dynamics limits.

.. class:: vortex.scan.inactive_policy.FixedDynamicLimited

    Generate fixed-duration inactive segments that satisfy the scan's dynamics limits.

    .. method:: __init__(inter_segment_samples=100, inter_volume_samples=100)

        Create a new object.

        :param int inter_segment_samples:
            Initial value for :attr:`inter_segment_samples`.
        :param int inter_volume_samples:
            Initial value for :attr:`inter_volume_samples`.

    .. property:: inter_segment_samples
        :type: int

        Number of samples between segments.
        Deafult is ``100``.

    .. property:: inter_volume_samples
        :type: int

        Number of samples between volumes, for scans that loop.
        Deafult is ``100``.

.. class:: vortex.scan.inactive_policy.FixedLinear

    Generate fixed-duration inactive segments that follow a straight line.

    .. method:: __init__(inter_segment_samples=100, inter_volume_samples=100)

        Create a new object.

        :param int inter_segment_samples:
            Initial value for :attr:`inter_segment_samples`.
        :param int inter_volume_samples:
            Initial value for :attr:`inter_volume_samples`.

    .. property:: inter_segment_samples
        :type: int

        Number of samples between segments.
        Deafult is ``100``.

    .. property:: inter_volume_samples
        :type: int

        Number of samples between volumes, for scans that loop.
        Deafult is ``100``.

Scans
-----

.. class:: vortex.scan.SegmentedScan

    A scan pattern that consists of an alternating sequence of active and inactive segments.

    The active segments are instances of :class:`~vortex.scan.Segment`, which define the segment's active sample positions.
    This is the base class for all segmented scans.
    Typically, subclasses generate the active segments internally from some parameterization, such as :class:`~vortex.scan.RasterWaypoints` or :class:`~vortex.scan.RadialWaypoints`.

    This class is present in C++ but is abstracted away in the Python bindings.
    It cannot be instantiated from Python.

    .. method:: initialize(config)

        Initialize the scan using the supplied configuration.

        :param SegmentedScanConfig config:
            New configuration to apply.
            Type must match the derived scan type.

    .. method:: change(config, restart=False, event_id=0)

        Change the scan parameters to those in this new configuration.
        The change takes effect immediately and will interrupt the current segment.

        .. warning::

            Changing :attr:`~vortex.scan.SegmentedScanConfig.channels_per_sample` is not permitted.

        :param SegmentedScanConfig config:
            New configuration to apply.
            Type must match the derived scan type.
        :param bool restart:
            If ``True``, the current scan state is mapped onto the new scan state to the nearest segment boundary, such that a scan changed mid-volume will continue from the same segment index within that volume.
            If ``False``, the scan starts from the first segment.
        :param int event_id:
            Identifying number of the change event to insert into the scan at the instant the changes take effect.

    .. method:: prepare(count=None)

        Buffer and consolidate the complete scan for rapid generation.
        If ``count`` is specified, buffer at minimum the requested number of samples.
        Otherwise, buffer the complete scan.
        Consolidation will be performed if enabled and the scan becomes fully buffered.

        :param Optional[int] count:
            Number of samples to buffer.

    .. method:: restart(sample=0, position=None, velocity=None, include_start=True)

        Reset the scan internal state to the given sample, position, and velocity.
        Subsequent calls for pattern generation will include a pre-scan inactive segment from this state to the first active segment.

        .. note::

            Specify no arguments or all arguments.
            If no arguments are specified, the scan resets to the origin at rest.

        :param int sample:
            New sample time.
        :param numpy.ndarray[float] position:
            New position for all channels.
        :param numpy.ndarray[float] velocity:
            New velocity for all channels.
        :param bool include_start:
            If ``True``, generate a sample at this new state.
            If ``False``, do not generate a starting sample, such as when another scan has already produced it.

    .. method:: scan_markers()

        Prepare the scan and return all markers.

        :return List[~vortex.marker.ScanBoundary | ~vortex.marker.VolumeBoundary | ~vortex.marker.SegmentBoundary | ~vortex.marker.ActiveLines | ~vortex.marker.InactiveLines | ~vortex.marker.Event]:
            List of all markers in the scan, sorted by sequence number.

    .. method:: scan_buffer()

        Prepare the scan and return the complete scan waveform for all channels.

        :return numpy.ndarray[float]:
            Scan waveforms generated at the given sampling rate.
            Each row is a sample, and each column is a channel.

    .. method:: scan_segments()

        Return a list of segments in this scan.

        :return List[~vortex.scan.Segment]:
            The scan segments.

    .. property:: config
        :type: ~vortex.scan.SegmentedScanConfig

        Copy of the active configuration.
        Type corresponds to the derived class.

.. class:: vortex.scan.Segment

    An active segment as part of a :class:`~vortex.scan.SegmentedScan`.

    .. property:: position
        :type: numpy.ndarray[float]

        An ordered sequence of each sample positions in this segment.
        The positions are specified as a 2D array where each row is a position and each column is a channel.
        For example, a 2D segment with N positions has shape [``N``, ``2``].
        The number of channels must match that of the associated :class:`SegmentedScan`.

    .. property:: entry_position
        :type: numpy.ndarray[float]

        Accessor for the first sample position.
        Read only.

    .. property:: exit_position
        :type: numpy.ndarray[float]

        Accessor for the last sample position.
        Read only.

    .. method:: entry_velocity(samples_per_second)

        Approximate the scan velocity at the first sample using the specified sampling rate and the first finite difference.

        :param float samples_per_second:
            Sampling rate for the velocity calculation.
        :return  numpy.ndarray[float]:
            The calculated velocity.

    .. method:: exit_velocity(samples_per_second)

        Approximate the scan velocity at the last sample using the specified sampling rate and the first finite difference.

        :param float samples_per_second:
            Sampling rate for the velocity calculation.
        :return  numpy.ndarray[float]:
            The calculated velocity.

    .. property:: markers
        :type: List[~vortex.marker.ScanBoundary | ~vortex.marker.VolumeBoundary | ~vortex.marker.SegmentBoundary | ~vortex.marker.ActiveLines | ~vortex.marker.InactiveLines | ~vortex.marker.Event]

        List of markers associated with this segment.

.. class:: vortex.scan.SegmentedScanConfig

    Shared configuration options for segmented scan classes.

    This class is present in C++ but is abstracted away in the Python bindings.
    It cannot be instantiated from Python.

    .. property:: channels_per_sample
        :type: int

        Number of channels per sample.
        Default is ``2`` for a 2D scan pattern.

    .. property:: samples_per_second
        :type: int

        Number of samples per second for the scan.
        This is used to ensure that dynamic limits are met.
        Default is ``100_000``.

    .. property:: sampling_interval
        :type: float

        Interval between samples computed from :attr:`samples_per_second`.
        Read-only.

    .. property:: loop
        :type: bool

        Restart the scan after it completes.
        This produces an infinite scan pattern.
        Default is ``True``.

    .. property:: consolidate
        :type: bool

        Optimize internal scan buffers once the scan has looped once.
        Default is ``False``.

    .. property:: limits
        :type: List[~vortex.scan.Limits]

        Dynamics limits for each channel of the scan.
        These are used for limit checks and inactive segment policies.
        Default is ``[Limits(position=(-12.5, 12.5), velocity=8e3, acceleration=5e6)]*2``.

    .. property:: bypass_limits_check
        :type: bool

        If ``False``, check that the scan pattern satisfies position, velocity, and acceleration limits.
        If ``True``, skip this check.
        This check does not apply to generated inactive segments.
        Default is ``False``.

    .. property:: inactive_policy
        :type: [~vortex.scan.inactive_policy.MinimumDynamicLimited | ~vortex.scan.inactive_policy.FixedDynamicLimited | ~vortex.scan.inactive_policy.FixedLinear]

        Policy to use for generating inactive segments.
        Default is ``MinimumDynamicLimited()``.

    .. method:: to_segments()

        Generate the segments described by this configuration.

        :return List[~vortex.scan.Segment]:
            The pattern segments.

        .. seealso::

            See :ref:`how-to/interleave` for an example application.

    .. method:: validate()

        Check the configuration for errors.

        :raises RuntimeError:
            If the configuration is invalid.

.. class:: vortex.scan.Limits

    Dynamics limits for a single axis.

    .. method:: __init__(position=(-10, 10), velocity=100, acceleration=10_000)

        Create a new object.

        :param ~vortex.Range position:
            Initial value for :attr:`position`.
        :param float velocity:
            Initial value for :attr:`velocity`.
        :param float acceleration:
            Initial value for :attr:`acceleration`.

    .. property:: position
        :type: ~vortex.Range

        Upper and lower positions bounds.

    .. property:: velocity
        :type: float

        Absolute value of velocity limit.

    .. property:: acceleration
        :type: float

        Absolute value of velocity limit.

Raster Scan
^^^^^^^^^^^

.. plot::

    from vortex import Range
    from vortex.scan import RasterScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    scan = RasterScan()
    cfg = scan.config
    cfg.volume_extent = Range.symmetric(2)
    cfg.samples_per_segment = 100
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True
    scan.initialize(cfg)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Raster')
    fig.tight_layout()

.. class:: vortex.scan.RasterScan

    Base: :class:`~vortex.scan.SegmentedScan`

    A raster scan with non-repeated segments.

    All members are inherited.

.. class:: vortex.scan.RasterScanConfig

    Bases: :class:`~vortex.scan.RasterWaypoints`, :class:`~vortex.scan.SequentialPattern`, :class:`~vortex.scan.SegmentedScanConfig`

    Configuration object for :class:`~vortex.scan.RasterScan`.

    All members are inherited.

Repeated Raster Scan
^^^^^^^^^^^^^^^^^^^^

.. plot::

    from vortex import Range
    from vortex.scan import RepeatedRasterScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    scan = RepeatedRasterScan()
    cfg = scan.config
    cfg.volume_extent = Range.symmetric(2)
    cfg.samples_per_segment = 100
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True
    scan.initialize(cfg)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Repeated Raster')
    fig.tight_layout()

.. class:: vortex.scan.RepeatedRasterScan

    Base: :class:`~vortex.scan.SegmentedScan`

    A raster scan with repeated segments.

    All members are inherited.

    .. seealso::

        See :ref:`how-to/repeated-bidirectional` and :ref:`demo/live-view-repeated` for example applications of :class:`~vortex.scan.RepeatedRasterScan`.

.. class:: vortex.scan.RepeatedRasterScanConfig

    Bases: :class:`~vortex.scan.RasterWaypoints`, :class:`~vortex.scan.RepeatedPattern`, :class:`~vortex.scan.SegmentedScanConfig`

    Configuration object for :class:`~vortex.scan.RepeatedRasterScan`.

    All members are inherited.

Radial Scan
^^^^^^^^^^^

.. plot::

    from vortex.scan import RadialScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    scan = RadialScan()
    cfg = scan.config
    cfg.samples_per_segment = 100
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True
    scan.initialize(cfg)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Radial')
    fig.tight_layout()

.. class:: vortex.scan.RadialScan

    Base: :class:`~vortex.scan.SegmentedScan`

    A radial scan with non-repeated segments.

    All members are inherited.

.. class:: vortex.scan.RadialScanConfig

    Bases: :class:`~vortex.scan.RadialWaypoints`, :class:`~vortex.scan.SequentialPattern`, :class:`~vortex.scan.SegmentedScanConfig`

    Configuration object for :class:`~vortex.scan.RadialScan`.

    All members are inherited.

Repeated Radial Scan
^^^^^^^^^^^^^^^^^^^^

.. plot::

    from vortex.scan import RepeatedRadialScan
    from vortex_tools.scan import plot_annotated_waveforms_space

    scan = RepeatedRadialScan()
    cfg = scan.config
    cfg.samples_per_segment = 100
    for limit in cfg.limits:
        limit.acceleration *= 5
    cfg.loop = True
    scan.initialize(cfg)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Repeated Radial')
    fig.tight_layout()

.. class:: vortex.scan.RepeatedRadialScan

    Base: :class:`~vortex.scan.SegmentedScan`

    A radial scan with repeated segments.

    All members are inherited.

.. class:: vortex.scan.RepeatedRadialScanConfig

    Bases: :class:`~vortex.scan.RepeatedRadialWaypoints`, :class:`~vortex.scan.SequentialPattern`, :class:`~vortex.scan.SegmentedScanConfig`

    Configuration object for :class:`~vortex.scan.RepeatedRadialScan`.

    All members are inherited.

Freeform Scan
^^^^^^^^^^^^^

.. plot::

    from math import pi

    from vortex.scan import FreeformScanConfig, FreeformScan, SequentialPattern
    from vortex_tools.scan import plot_annotated_waveforms_space

    (r, theta) = np.meshgrid(
        [2, 3, 4],
        np.linspace(0, 2*pi, 200),
        indexing='ij'
    )

    x = (r + 0.1*np.sin(r + 20*theta)) * np.sin(theta)
    y = (r + 0.1*np.sin(r + 20*theta)) * np.cos(theta)

    waypoints = np.stack((x, y), axis=-1)
    pattern = SequentialPattern().to_pattern(waypoints)

    cfg = FreeformScanConfig()
    cfg.pattern = pattern
    for limit in cfg.limits:
        limit.velocity *= 5
        limit.acceleration *= 10
    cfg.bypass_limits_check = True
    cfg.loop = True

    scan = FreeformScan()
    scan.initialize(cfg)

    fg = plt.rcParams['lines.color']
    fig, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line=fg)
    ax.set_title('Freeform Scan')
    fig.tight_layout()

.. class:: vortex.scan.FreeformScan

    Base: :class:`~vortex.scan.SegmentedScan`

    A scan pattern that allows user-specified active segments.

    All members are inherited.

.. class:: vortex.scan.FreeformScanConfig

    Base: :class:`~vortex.scan.SegmentedScanConfig`

    Configuration object for :class:`~vortex.scan.FreeformScan`.

    .. property:: segments
        :type: List[~vortex.scan.Segment]

        Active segments in this scan pattern.
