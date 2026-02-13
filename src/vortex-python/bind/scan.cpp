#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>
#include <vortex-python/bind/marker.hpp>
#include <vortex-python/bind/scan.hpp>

#include <vortex/scan.hpp>

#include <vortex/driver/motion.hpp>

template<typename C>
static void setup_warp(py::class_<C>& c) {

    c.def("forward", [](const C& o, xt::pyarray<double>& in) {
        xt::pyarray<double> out;
        o.forward(in, out);
        return out;
    }, "in"_a);
    c.def("inverse", [](const C& o, xt::pyarray<double>& in) {
        xt::pyarray<double> out;
        o.inverse(in, out);
        return out;
    }, "in"_a);

    //c.def("forward", [](const C& o, xt::pyarray<double>& in, xt::pyarray<double>& out) { o.forward(in, out); }, "in"_a, "out"_a);
    //c.def("inverse", [](const C& o, xt::pyarray<double>& in, xt::pyarray<double>& out) { o.inverse(in, out); }, "in"_a, "out"_a);

}

static auto bind_warp(py::module& root){
    auto m = root.def_submodule("warp");

    {
        using C = vortex::scan::warp::none_t;
        CLS_VAL(NoWarp);

        c.def(py::init());
        c.def("__repr__", [](const C& v){return "NoWarp";});

        setup_warp<C>(c);
    }
    {
        using C = vortex::scan::warp::angular_t;
        CLS_VAL(Angular);

        c.def(py::init());
        c.def(py::init<double>(), "factor"_a);
        c.def("__repr__", [](const C& v){return fmt::format("Angular(factor={})", v.factor);});

        RW_VAR(factor);
        setup_warp<C>(c);
    }
    {
        using C = vortex::scan::warp::telecentric_t;
        CLS_VAL(Telecentric);

        c.def(py::init());
        c.def(py::init<double>(), "galvo_lens_spacing"_a);
        c.def(py::init<double, double>(), "galvo_lens_spacing"_a, "scale"_a);
        c.def("__repr__", [](const C& v){return fmt::format("Telecentric(galvo_lens_spacing={}, scale={})",
                                                            v.galvo_lens_spacing, v.scale);});

        RW_VAR(galvo_lens_spacing);
        RW_VAR(scale);
        setup_warp<C>(c);
    }
}

static auto bind_inactive_policy(py::module& root){
    auto m = root.def_submodule("inactive_policy");

#if defined(VORTEX_ENABLE_REFLEXXES)
    {
        using C = vortex::scan::inactive_policy::minimum_dynamic_limited_t;
        CLS_VAL(MinimumDynamicLimited);

        c.def(py::init());
        c.def("__repr__", [](const C& v){return "MinimumDynamicLimited()";});
    }
    {
        using C = vortex::scan::inactive_policy::fixed_dynamic_limited_t;
        CLS_VAL(FixedDynamicLimited);

        c.def(py::init<size_t, size_t>(), "inter_segment_samples"_a = 100, "inter_volume_samples"_a = 100);
        c.def("__repr__", [](const C& v){return fmt::format("FixedDynamicLimited(inter_segment_samples={}, inter_volume_samples{})", v.inter_segment_samples, v.inter_volume_samples);});

        RW_VAR(inter_segment_samples);
        RW_VAR(inter_volume_samples);

        FXN(set_samples);
    }
#endif
    {
        using C = vortex::scan::inactive_policy::fixed_linear_t;
        CLS_VAL(FixedLinear);


        c.def(py::init<size_t, size_t>(), "inter_segment_samples"_a = 100, "inter_volume_samples"_a = 100);
        c.def("__repr__", [](const C& v){return fmt::format("FixedLinear(inter_segment_samples={}, inter_volume_samples{})", v.inter_segment_samples, v.inter_volume_samples);});

        RW_VAR(inter_segment_samples);
        RW_VAR(inter_volume_samples);

        FXN(set_samples);
    }
}

template<typename T>
static auto bind_limits(py::module& m) {
    using C = vortex::motion::limits_t<T>;
    CLS_VAL(Limits);
    c.def("__repr__", [](const C& o) {
        return fmt::format("Limits(position={}, velocity={}, acceleration={})",
            PY_REPR(o.position), PY_REPR(o.velocity), PY_REPR(o.acceleration));
        });

    c.def(py::init<vortex::range_t<T>, T, T>(), "position"_a = vortex::range_t<T>{ -10, 10 }, "velocity"_a = 100, "acceleration"_a = 10'000);

    RW_VAR(position);
    RW_VAR(velocity);
    RW_VAR(acceleration);

    SHALLOW_COPY();

    return c;
}

template<typename C>
static void setup_scan_config(py::class_<C>& c) {
    RW_VAR(loop);
    RW_VAR(consolidate);

    RW_VAR(samples_per_second);
    RO_ACC(sampling_interval);

    RW_VAR(channels_per_sample);

    SHALLOW_COPY();
}

template<typename C>
static void setup_segmented_scan_config(py::class_<C>& c) {
    setup_scan_config(c);

    FXN(to_segments);

    RW_VAR(limits);
    RW_VAR(bypass_limits_check);

    RW_VAR(inactive_policy);
}

template<typename C>
static void setup_xy_waypoints(py::class_<C>& c) {
    // TODO: change to an approach that allows in-place editing for xtensor_fixed
    RW_VAR_XT_FIXED(offset);
    RW_VAR(angle);

    RW_VAR(extents);
    RW_ACC(bscan_extent);
    RW_ACC(segment_extent);
    RW_ACC(volume_extent);

    RW_VAR(shape);
    RW_ACC(ascans_per_bscan);
    RW_ACC(bscans_per_volume);
    RW_ACC(samples_per_segment);
    RW_ACC(segments_per_volume);

    RW_VAR(warp);

    FXN(to_waypoints);
}

template<typename C>
static void setup_pattern(py::class_<C>& c) {
    c.def("to_pattern", [](const C& o, const xt::pytensor<double, 3>& waypoints) {
        return o.template to_pattern<double>(waypoints);
    });
    c.def("to_pattern", [](const C& o, const std::vector<xt::pytensor<double, 2>>& waypoints) {
        return o.template to_pattern<double>(waypoints.begin(), waypoints.end());
    });

    SHALLOW_COPY();
}

template<typename C>
static void setup_sequential_pattern(py::class_<C>& c) {
    RW_VAR(flags);

    RW_VAR(bidirectional_segments);
    RW_VAR(bidirectional_volumes);

    setup_pattern(c);
}

template<typename C>
static void setup_repeated_pattern(py::class_<C>& c) {
    RW_VAR(repeat_count);
    RW_VAR(repeat_period);

    RW_VAR(repeat_strategy);

    RW_VAR(bidirectional_segments);

    setup_pattern(c);
}

template<typename C>
static void setup_scan(py::class_<C, std::shared_ptr<C>>& c) {

    c.def("restart", [](C& o) {
        py::gil_scoped_release gil;
        o.restart();
    }, doc(c, "restart"));
    c.def("restart", [](C& o, vortex::counter_t sample, xt::pytensor<double, 1>& position, xt::pytensor<double, 1>& velocity, bool include_start) {
        py::gil_scoped_release gil;
        return o.restart(sample, position, velocity, include_start);
    }, "sample"_a, "position"_a, "velocity"_a, "include_start"_a, doc(c, "restart"));

    c.def("next", [](C& o, const vortex::cpu_view_t<typename C::element_t>& buffer) {

        std::vector<vortex::default_marker_t> markers;
        auto n = o.next(markers, buffer);

        return std::make_tuple(n, std::move(markers));

    }, "buffer"_a, doc(c, "next"));
    c.def("next", [](C& o, std::vector<vortex::default_marker_t>& markers, const vortex::cpu_view_t<typename C::element_t>& buffer) {

        return o.next(markers, buffer);

    }, "markers"_a, "buffer"_a, doc(c, "next"));

    RO_ACC(config);

    c.def("prepare", [](C& o) { o.prepare(); }, doc(c, "prepare"));
    c.def("prepare", [](C& o, size_t count) { o.prepare(count); }, "count"_a, doc(c, "prepare"));

    FXN(scan_markers);
    FXN(scan_buffer);
    FXN(scan_segments);

}

template<typename C>
static void bind_xy_scan(py::module& m, const std::string& name) {
    using config_t = typename C::config_t;
    auto c = py::class_<C, std::shared_ptr<C>>(m, name.c_str());

    setup_scan(c);

    c.def(py::init());

    FXN_GIL(initialize, "config"_a);

    c.def("change", [](C& o, const config_t& config, bool restart, vortex::marker::event::eid_t event_id) {
        py::gil_scoped_release gil;
        o.change(config, restart, event_id);
    }, "config"_a, "restart"_a = false, "event_id"_a = 0);

}

void bind_scan(py::module& root) {
    auto m = root.def_submodule("scan");

    bind_limits<double>(m);

    auto limits = m.def_submodule("limits");
    limits.attr("ThorLabs_GVS_5mm") = vortex::motion::limits_t<double>{ {-12.5, 12.5}, 8e3, 5e6 };  // ThorLabs, assuming square wave at 100 Hz
    limits.attr("ScannerMax_Saturn_5B") = vortex::motion::limits_t<double>{ {-27.5, 27.5}, 80e3, 25e6 }; // ScannerMax, experimentally adjusted for factory tuning #4

    {
        using C = vortex::scan::segment_t<double, vortex::default_marker_t>;
        CLS_VAL(Segment);

        c.def(py::init());
        c.def("__repr__", [](const C& v) {
            return fmt::format("Segment:\n{},\nentry_delta={}, exit_delta={}, markers={}\n",
                PY_REPR(v.position),
                PY_REPR(v.entry_delta),
                PY_REPR(v.exit_delta),
                PY_REPR(v.markers));
        });

        RW_VAR_XT(position);
        RO_ACC(entry_position);
        RO_ACC(exit_position);

        // TODO: fix these accessors
        //RW_VAR(entry_delta);
        //RW_VAR(exit_delta);
        FXN(entry_velocity, "samples_per_second"_a);
        FXN(exit_velocity, "samples_per_second"_a);

        RW_VAR(markers);

        SHALLOW_COPY();

        py::bind_vector<std::vector<C>>(m, "SegmentList")
            .def("__repr__", [](const std::vector<C>& v) {
            return list_repr(v, "SegmentList");
        });
    }

    bind_warp(m);
    bind_inactive_policy(m);

    bind_xy_scan<vortex::raster_scan_t>(m, "RasterScan");
    bind_xy_scan<vortex::repeated_raster_scan_t>(m, "RepeatedRasterScan");
    bind_xy_scan<vortex::radial_scan_t>(m, "RadialScan");
    bind_xy_scan<vortex::repeated_radial_scan_t>(m, "RepeatedRadialScan");
    bind_xy_scan<vortex::spiral_scan_t>(m, "SpiralScan");
    bind_xy_scan<vortex::freeform_scan_t>(m, "FreeformScan");

    {
        using C = vortex::scan::raster_waypoints_t<double, vortex::default_warp_t>;
        CLS_VAL(RasterWaypoints);

        c.def(py::init());
        setup_xy_waypoints(c);
    }
    {
        using C = vortex::raster_scan_t::config_t;
        CLS_VAL(RasterScanConfig);
        c.def("__repr__", [](const C& v) {
            return fmt::format("RasterScanConfig(angle={},\n  ascans_per_bscan={},\n  bidirectional_segments={},\n  bidirectional_volumes={},\n  bscan_extent={},\n  bscans_per_volume={},\n  bypass_limits_check={},\n  channels_per_sample={},\n  consolidate={},\n  extents={},\n  flags={},\n  limits={},\n  loop={},\n  offset={},\n  samples_per_second={},\n  samples_per_segment={},\n  sampling_interval={},\n  segment_extent={},\n  segments_per_volume={},\n  shape={},\n  volume_extent={},\n  warp={}\n)",
                v.angle, v.ascans_per_bscan(), v.bidirectional_segments, v.bidirectional_volumes, PY_REPR(v.bscan_extent()), v.bscans_per_volume(), v.bypass_limits_check, v.channels_per_sample, v.consolidate, PY_REPR(v.extents), PY_REPR(v.flags), PY_REPR(v.limits), v.loop, v.offset, v.samples_per_second, v.samples_per_segment(), v.sampling_interval(), PY_REPR(v.segment_extent()), v.segments_per_volume(), v.shape, PY_REPR(v.volume_extent()), PY_REPR(v.warp));
            });

        c.def(py::init());
        setup_xy_waypoints(c);
        setup_sequential_pattern(c);
        setup_segmented_scan_config(c);
    }
    {
        using C = vortex::repeated_raster_scan_t::config_t;
        CLS_VAL(RepeatedRasterScanConfig);

        c.def(py::init());
        setup_xy_waypoints(c);
        setup_repeated_pattern(c);
        setup_segmented_scan_config(c);
    }
    {
        using C = vortex::scan::radial_waypoints_t<double, vortex::default_warp_t>;
        CLS_VAL(RadialWaypoints);

        c.def(py::init());
        setup_xy_waypoints(c);

        FXN(set_half_evenly_spaced);
        FXN(set_evenly_spaced);
        FXN(set_aiming);
    }
    {
        using C = vortex::radial_scan_t::config_t;
        CLS_VAL(RadialScanConfig);

        c.def(py::init());
        setup_xy_waypoints(c);
        setup_sequential_pattern(c);
        setup_segmented_scan_config(c);

        FXN(set_half_evenly_spaced);
        FXN(set_evenly_spaced);
        FXN(set_aiming);
    }
    {
        using C = vortex::repeated_radial_scan_t::config_t;
        CLS_VAL(RepeatedRadialScanConfig);

        c.def(py::init());
        setup_xy_waypoints(c);
        setup_repeated_pattern(c);
        setup_segmented_scan_config(c);

        FXN(set_half_evenly_spaced);
        FXN(set_evenly_spaced);
        FXN(set_aiming);
    }

    {
        using C = vortex::scan::spiral_waypoints_t<double, vortex::default_warp_t>;
        CLS_VAL(SpiralWaypoints);

        c.def(py::init());
        setup_xy_waypoints(c);

        RW_VAR(angular_velocity);
        RW_VAR(linear_velocity);

        RW_ACC(inner_radius);
        RW_ACC(outer_radius);
        RW_VAR(rings_per_spiral);

        RW_VAR(acceleration_limit);
        RO_ACC(radial_pitch);

        FXN(set_hybrid);
        FXN(set_isotropic);
    }
    {
        using C = vortex::spiral_scan_t::config_t;
        CLS_VAL(SpiralScanConfig);

        c.def(py::init());
        setup_xy_waypoints(c);
        setup_sequential_pattern(c);
        setup_segmented_scan_config(c);

        RW_VAR(angular_velocity);
        RW_VAR(linear_velocity);

        RW_ACC(inner_radius);
        RW_ACC(outer_radius);
        RW_VAR(rings_per_spiral);

        RW_VAR(acceleration_limit);
        RO_ACC(radial_pitch);

        FXN(set_hybrid);
        FXN(set_isotropic);

    }

    {
        using C = vortex::freeform_scan_t::config_t;
        CLS_VAL(FreeformScanConfig);

        c.def(py::init());

        setup_segmented_scan_config(c);

        RW_VAR(pattern);
    }

    {
        using C = vortex::scan::sequential_pattern_t<vortex::default_marker_t, vortex::default_marker_flags_t>;
        CLS_VAL(SequentialPattern);

        c.def(py::init());

        setup_sequential_pattern(c);
    }

    {
        using C = vortex::scan::repeated_pattern_t<vortex::default_marker_t, vortex::default_marker_flags_t>;
        CLS_VAL(RepeatedPattern);

        c.def(py::init());

        setup_repeated_pattern(c);

        using P = C;
        auto& p = c;
        {
            using C = typename P::repeat_order;
            CLS_VAL(RepeatOrder);

            c.def(py::init<vortex::default_marker_flags_t>(), "flags"_a = vortex::default_marker_flags_t{});

            RW_VAR(flags);

            SHALLOW_COPY();
        }
        {
            using C = typename P::repeat_pack;
            CLS_VAL(RepeatPack);

            c.def(py::init<vortex::default_marker_flags_t>(), "flags"_a = vortex::default_marker_flags_t{});

            RW_VAR(flags);

            SHALLOW_COPY();
        }
        {
            using C = typename P::repeat_flags;
            CLS_VAL(RepeatFlags);

            c.def(py::init<std::vector<vortex::default_marker_flags_t>>(), "flags"_a = std::vector<vortex::default_marker_flags_t>{});

            RW_VAR(flags);

            SHALLOW_COPY();
        }
    }

}
