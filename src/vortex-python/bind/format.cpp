#include <vortex/format.hpp>

#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>

PYBIND11_MAKE_OPAQUE(vortex::format::format_plan_t);

template<typename C>
static void setup_stack_format_executor_config(py::class_<C>& c) {
    RW_VAR(erase_after_volume);

    RW_VAR(sample_slice);
    RW_VAR(sample_transform);

    SHALLOW_COPY();
}

template<typename T>
static void bind_linear_transform(py::module& m, const std::string& name) {
    using C = vortex::copy::transform::linear_t<T>;
    auto c = py::class_<C>(m, name.c_str());

    c.def(py::init());
    c.def(py::init<T, T>());
    c.def("__repr__", [name](const C& v){
                          return fmt::format("{}(scale={}, offset={})",
                                             name.c_str(), PY_REPR(v.scale), PY_REPR(v.offset));
                      });

    RW_VAR(scale);
    RW_VAR(offset);

    SHALLOW_COPY();
}

void bind_action(py::module& m){
    {
        using C = vortex::format::action::copy;
        CLS_VAL(Copy);
        c.def(py::self == py::self);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("Copy(count={}, block_offset={}, buffer_segment={}, buffer_record={}, reverse={})",
                                                 v.count, v.block_offset, v.buffer_segment, v.buffer_record, v.reverse);
                          });

        RW_VAR(count);
        RW_VAR(block_offset);
        RW_VAR(buffer_segment);
        RW_VAR(buffer_record);
        RW_VAR(reverse);

    }
    {
        using C = vortex::format::action::resize;
        CLS_VAL(Resize);
        c.def(py::self == py::self);

        c.def(py::init());
                c.def("__repr__", [](const C& v){
                              return fmt::format("Resize(segments_per_volume={}, records_per_segment={})",
                                                 v.shape[0], v.shape[1]);
                                  });
        RW_VAR(shape);

        RW_ACC(segments_per_volume);
        RW_ACC(records_per_segment);
    }
    {
        using C = vortex::format::action::finish_segment;
        CLS_VAL(FinishSegment);
        c.def(py::self == py::self);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("FinishSegment(sample={}, scan_index={}, volume_index={}, segment_index_buffer={})",
                                                 v.sample, v.scan_index, v.volume_index, v.segment_index_buffer);
                          });

        RW_VAR(sample);
        RW_VAR(scan_index);
        RW_VAR(volume_index);
        RW_VAR(segment_index_buffer);

    }
    {
        using C = vortex::format::action::finish_volume;
        CLS_VAL(FinishVolume);
        c.def(py::self == py::self);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("FinishVolume(sample={}, scan_index={}, volume_index={})",
                                                 v.sample, v.scan_index, v.volume_index);
                          });

        RW_VAR(sample);
        RW_VAR(scan_index);
        RW_VAR(volume_index);

    }
    {
        using C = vortex::format::action::finish_scan;
        CLS_VAL(FinishScan);
        c.def(py::self == py::self);

        c.def(py::init());
                c.def("__repr__", [](const C& v){
                              return fmt::format("FinishScan(sample={}, scan_index={})",
                                                 v.sample, v.scan_index);
                          });

        RW_VAR(sample);
        RW_VAR(scan_index);

    }
}

template<typename T1, typename T2, typename C>
void bind_cuda_format_method(py::class_<C, std::shared_ptr<C>>& c) {
#if defined(VORTEX_ENABLE_CUDA)
    c.def("format", [](C& o,
        const vortex::cuda::cuda_view_t<T1>& volume_buffer,
        const vortex::cuda::cuda_view_t<const T2>& segment_buffer,
        size_t segment_index, size_t record_index, bool reverse) {

        py::gil_scoped_release gil;
        o.format(volume_buffer, segment_buffer, segment_index, record_index, reverse);

    }, "volume_buffer"_a, "segment_buffer"_a, "segment_index"_a, "record_index"_a = 0, "reverse"_a = false);
#endif
}

template<typename T1, typename T2, typename T3, typename C>
void bind_cuda_with_galvo_format_method(py::class_<C, std::shared_ptr<C>>& c) {
#if defined(VORTEX_ENABLE_CUDA)
    c.def("format", [](C& o,
        const vortex::cuda::cuda_view_t<T1>& volume_buffer,
        const vortex::cuda::cuda_view_t<const T2>& sample_target,
        const vortex::cuda::cuda_view_t<const T2>& sample_actual,
        const vortex::cuda::cuda_view_t<const T3>& segment_buffer,
        size_t segment_index, size_t record_index, bool reverse) {

        py::gil_scoped_release gil;
        o.format(volume_buffer, sample_target, sample_actual, segment_buffer, segment_index, record_index, reverse);

    }, "volume_buffer"_a, "sample_target"_a, "sample_actual"_a, "segment_buffer"_a, "segment_index"_a, "record_index"_a = 0, "reverse"_a = false);
#endif
}

void bind_format(py::module& root) {
    auto m = root.def_submodule("format");

    {
        using C = vortex::format::format_planner_config_t;
        CLS_VAL(FormatPlannerConfig);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("FormatPlannerConfig(shape={}, segments_per_volume={}, records_per_segments={}, mask={}, flip_reversed={}, strip_inactive={}, adapt_shape={})",
                                                 PY_REPR(v.shape), PY_REPR(v.segments_per_volume()), PY_REPR(v.records_per_segment()), PY_REPR(v.mask), PY_REPR(v.flip_reversed), PY_REPR(v.strip_inactive), PY_REPR(v.adapt_shape));
                          });

        RW_VAR(shape);
        RW_ACC(segments_per_volume);
        RW_ACC(records_per_segment);

        RW_VAR(mask);

        RW_VAR(flip_reversed);
        RW_VAR(strip_inactive);
        RW_VAR(adapt_shape);

        SHALLOW_COPY();
    }

    bind_action(m);

    py::bind_vector<vortex::format::format_plan_t>(m, "FormatPlan")
        .def("__repr__", [](const vortex::format::format_plan_t& v){
                             return list_repr(v, "FormatPlan");
                         });

    {
        using C = vortex::format_planner_t;
        CLS_PTR(FormatPlanner);

        c.def(py::init());
        c.def(py::init<std::shared_ptr<spdlog::logger>>());

        RO_ACC(config);

        FXN(initialize);
        FXN(reset);

        RO_ACC(segments_per_volume);
        RO_ACC(records_per_segment);
    }

    {
        using C = vortex::copy::slice::none_t;
        CLS_VAL(NullSlice);

        c.def(py::init());
        c.def("__repr__", [](const C& o){return "NullSlice";});

        SHALLOW_COPY();
    }
    {
        using C = vortex::copy::slice::simple_t;
        CLS_VAL(SimpleSlice);

        c.def(py::init());
        c.def(py::init<size_t>());
        c.def(py::init<size_t, size_t>());
        c.def(py::init<size_t, size_t, size_t>());
        // TODO: add constructor for Python slice object
        c.def("__repr__", [](const C& v){
                              return fmt::format("SimpleSlice(start={}, stop={}, step={})",
                                                 v.start, v.stop, v.step);
                          });

        RW_VAR(start);
        RW_VAR(stop);
        RW_VAR(step);

        // RO_ACC(count);
        c.def("count", py::overload_cast<>(&C::count, py::const_));

        SHALLOW_COPY();
    }

    {
        using C = vortex::copy::transform::none_t;
        CLS_VAL(NullTransform);

        c.def(py::init());
        c.def("__repr__", [](const C& v){return "NullTransform";});

        SHALLOW_COPY();
    }
    //bind_linear_transform<float>(m, "FloatLinearTransform");
    bind_linear_transform<double>(m, "LinearTransform");

    {
        using C = vortex::stack_format_executor_t::config_t;
        CLS_VAL(StackFormatExecutorConfig);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("StackFormatExecutorConfig(erase_after_volume={}, sample_slice={}, sample_transform={})",
                                                 PY_REPR(v.erase_after_volume), PY_REPR(v.sample_slice), PY_REPR(v.sample_transform));
                          });

        setup_stack_format_executor_config(c);
    }

    {
        using C = vortex::stack_format_executor_t;
        CLS_PTR(StackFormatExecutor);

        c.def(py::init());

        RO_ACC(config);

        FXN(initialize);

        // TODO: implement stack_format_executor_t.format(...)
        //bind_cuda_format_method<int8_t, int8_t>(c);
        //bind_cuda_format_method<int8_t, float>(c);
        //bind_cuda_format_method<uint16_t, uint16_t>(c);
    }
    {
        using C = vortex::broct_format_executor_t;
        auto c = py::class_<C, std::shared_ptr<C>, vortex::stack_format_executor_t>(m, "BroctFormatExecutor");

        c.def(py::init());
    }

    {
        using C = vortex::position_format_executor_t::config_t;
        CLS_VAL(PositionFormatExecutorConfig);

        c.def(py::init());

        setup_stack_format_executor_config(c);

        RW_VAR_XT_FIXED(transform);
        RW_VAR(use_target_position);
        RW_VAR(channels);

        c.def("set", [](C& o, std::array<double, 2> pitch, std::array<double, 2> offset, double angle) {
            o.set(pitch, offset, angle);
        }, "pitch"_a = std::array<double, 2>{ {1, 1} }, "offset"_a = std::array<double, 2>{ {0, 0} }, "angle"_a = 0);
    }

    {
        using C = vortex::position_format_executor_t;
        CLS_PTR(PositionFormatExecutor);

        c.def(py::init());

        RO_ACC(config);

        FXN(initialize);

        bind_cuda_with_galvo_format_method<int8_t, double, int8_t>(c);
        bind_cuda_with_galvo_format_method<int8_t, double, float>(c);
        bind_cuda_with_galvo_format_method<uint16_t, double, uint16_t>(c);
    }

    {
        using C = vortex::radial_format_executor_t::config_t;
        CLS_VAL(RadialFormatExecutorConfig);

        c.def(py::init());
        c.def("__repr__", [](const C& v){
                              return fmt::format("RadialFormatExecutorConfig(\n\
    erase_after_volume={},\n\
    sample_slice={},\n\
    sample_transform={},\n\
    volume_xy_extent={},\n\
    x_extent={},\n\
    y_extent={},\n\
    segment_rt_extent={},\n\
    radial_extent={},\n\
    angular_extent={},\n\
    radial_shape={},\n\
    radial_segments_per_volume={},\n\
    radial_records_per_segment={}\n)",
                                PY_REPR(v.erase_after_volume),
                                PY_REPR(v.sample_slice),
                                PY_REPR(v.sample_transform),
                                PY_REPR(v.volume_xy_extent),
                                PY_REPR(v.x_extent()),
                                PY_REPR(v.y_extent()),
                                PY_REPR(v.segment_rt_extent),
                                PY_REPR(v.radial_extent()),
                                PY_REPR(v.angular_extent()),
                                PY_REPR(v.radial_shape),
                                PY_REPR(v.radial_segments_per_volume()),
                                PY_REPR(v.radial_records_per_segment())
                                );});

        setup_stack_format_executor_config(c);

        RW_VAR(volume_xy_extent);
        RW_ACC(x_extent);
        RW_ACC(y_extent);

        RW_VAR(segment_rt_extent);
        RW_ACC(radial_extent);
        RW_ACC(angular_extent);

        RW_VAR(radial_shape);
        RW_ACC(radial_segments_per_volume);
        RW_ACC(radial_records_per_segment);
    }

    {
        using C = vortex::radial_format_executor_t;
        CLS_PTR(RadialFormatExecutor);

        c.def(py::init());

        RO_ACC(config);

        FXN(initialize);

        bind_cuda_format_method<int8_t, int8_t>(c);
        bind_cuda_format_method<int8_t, float>(c);
        bind_cuda_format_method<uint16_t, uint16_t>(c);

    }

    {
        using C = vortex::spiral_format_executor_t::config_t;
        CLS_VAL(SpiralFormatExecutorConfig);

        c.def(py::init());

        setup_stack_format_executor_config(c);

        RW_VAR(volume_xy_extent);
        RW_ACC(x_extent);
        RW_ACC(y_extent);

        RW_VAR(segment_extent);
        RW_ACC(radial_extent);

        RW_VAR(spiral_shape);
        RW_ACC(rings_per_spiral);
        RW_ACC(samples_per_spiral);

        RW_VAR(spiral_velocity);
    }
    {
        using C = vortex::spiral_format_executor_t;
        CLS_PTR(SpiralFormatExecutor);

        c.def(py::init());

        RO_ACC(config);

        FXN(initialize);

        bind_cuda_format_method<int8_t, int8_t>(c);
        bind_cuda_format_method<int8_t, float>(c);
        bind_cuda_format_method<uint16_t, uint16_t>(c);
    }

}
