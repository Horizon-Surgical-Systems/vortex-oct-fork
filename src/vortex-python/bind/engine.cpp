#include <spdlog/logger.h>

#include <vortex/engine/source.hpp>
#include <vortex/engine/clock.hpp>
#include <vortex/engine/dispersion.hpp>

#if defined(VORTEX_ENABLE_ENGINE)
#  include <vortex/endpoint/cpu.hpp>
#  include <vortex/endpoint/cuda.hpp>
#  include <vortex/endpoint/storage.hpp>
#endif

#include <vortex/acquire.hpp>
#include <vortex/engine.hpp>
#include <vortex/io.hpp>
#include <vortex/process.hpp>
#include <vortex/scan.hpp>

#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/marker.hpp>
#include <vortex-python/bind/scan.hpp>
#include <vortex-python/bind/memory.hpp>
#include <vortex-python/bind/exception.hpp>
#include <vortex-python/bind/engine.hpp>

template<typename T>
using sp = std::shared_ptr<T>;
template<typename T>
using spl = std::shared_ptr<vortex::sync::lockable<T>>;

template<typename T>
static void bind_utility_functions(py::module& m) {
    {
        m.def("find_rising_edges", [](const xt::pytensor<T, 1>& signal, size_t samples_per_second) {
            return vortex::engine::find_rising_edges(signal, samples_per_second);
        });
        m.def("find_rising_edges", [](const xt::pytensor<T, 1>& signal, size_t samples_per_second, size_t number_of_edges) {
            return vortex::engine::find_rising_edges(signal, samples_per_second, number_of_edges);
        });
        m.def("find_rising_edges", [](const xt::pyarray<T>& signal, size_t samples_per_second) {
            return vortex::engine::find_rising_edges(signal, samples_per_second);
        });
        m.def("find_rising_edges", [](const xt::pyarray<T>& signal, size_t samples_per_second, size_t number_of_edges) {
            return vortex::engine::find_rising_edges(signal, samples_per_second, number_of_edges);
        });
    }

    {
        m.def("compute_resampling", [](const vortex::engine::source_t<T>& source, size_t samples_per_second, size_t samples_per_ascan) {
            return vortex::engine::compute_resampling(source, samples_per_second, samples_per_ascan);
        });
        m.def("compute_resampling", [](const vortex::engine::source_t<T>& source, size_t samples_per_second, size_t samples_per_ascan, size_t clock_delay_samples) {
            return vortex::engine::compute_resampling(source, samples_per_second, samples_per_ascan, clock_delay_samples);
        });
    }

    m.def("dispersion_phasor", &vortex::engine::dispersion_phasor<T>);
}

#if defined(VORTEX_ENABLE_ENGINE)

template<typename C>
void bind_notify(py::class_<C, sp<C>>& c) {
    RW_VAR(update_callback);
    RW_VAR(segment_callback);
    RW_VAR(aggregate_segment_callback);
    RW_VAR(block_segment_callback);
    RW_VAR(volume_callback);
    RW_VAR(scan_callback);
    RW_VAR(event_callback);
}

template<typename C>
static void setup_cpu_endpoint(py::class_<C, sp<C>>& c) {
    bind_notify(c);

    RO_ACC(tensor);
    RO_ACC(executor);
}

template<typename C>
static void setup_cuda_endpoint(py::class_<C, sp<C>>& c) {
    setup_cpu_endpoint(c);

    auto cupy = try_import_cupy();
    if (cupy) {
        c.def_property_readonly("stream", [cupy](C& o) {
            return cupy.attr("cuda").attr("ExternalStream")(intptr_t(o.stream().handle()));
        });
    } else {
        c.def_property_readonly("stream", [](C& o) {
            throw std::runtime_error("CuPy (https://cupy.dev/) is required for CUDA tensor interoperability");
        });
    }
}

#define BIND_MEMORY_ENDPOINT(executor, tensor) \
    c.def(py::init<sp<vortex::executor>, std::vector<size_t>, sp<spdlog::logger>>(), "executor"_a, "shape"_a, "log"_a = nullptr); \
    c.def(py::init<sp<vortex::executor>, spl<vortex::tensor<T>>, sp<spdlog::logger>>(), "executor"_a, "tensor"_a, "log"_a = nullptr); \
    m.def("_bind", [](sp<C>& a) { return vortex::engine::bind::template endpoint<block_t>(a); });

#define BIND_CPU_ENDPOINT(type, name, executor) \
    { \
        using C = vortex::endpoint::type<T>; \
        auto c = py::class_<C, sp<C>>(m, (#name + suffix).c_str()); \
    \
        BIND_MEMORY_ENDPOINT(executor, cpu_tensor_t); \
    \
        setup_cpu_endpoint(c); \
    }

#define BIND_CUDA_ENDPOINT(type, name, executor, tensor) \
    { \
        using C = vortex::endpoint::type<T>; \
        auto c = py::class_<C, sp<C>>(m, (#name + suffix).c_str()); \
    \
        BIND_MEMORY_ENDPOINT(executor, cuda::tensor); \
    \
        setup_cuda_endpoint(c); \
    }

template<typename engine_t>
static void bind_memory_endpoints(py::module& m) {
    {
        using T = typename engine_t::acquire_element_t;
        auto suffix = std::string(detail::dtype<T>::display_name);

        BIND_CPU_ENDPOINT(spectra_stack_cpu_tensor, SpectraStackTensorEndpoint, stack_format_executor_t);

        BIND_CUDA_ENDPOINT(spectra_stack_cuda_host_tensor, SpectraStackHostTensorEndpoint, stack_format_executor_t, cuda_host_tensor_t);

        BIND_CUDA_ENDPOINT(spectra_stack_cuda_device_tensor, SpectraStackDeviceTensorEndpoint, stack_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(spectra_radial_cuda_device_tensor, SpectraRadialDeviceTensorEndpoint, radial_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(spectra_spiral_cuda_device_tensor, SpectraSpiralDeviceTensorEndpoint, spiral_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(spectra_position_cuda_device_tensor, SpectraPositionDeviceTensorEndpoint, position_format_executor_t, cuda_device_tensor_t);
    }

    {
        using T = typename engine_t::process_element_t;
        auto suffix = std::string(detail::dtype<T>::display_name);

        BIND_CPU_ENDPOINT(ascan_stack_cpu_tensor, StackTensorEndpoint, stack_format_executor_t);

        BIND_CUDA_ENDPOINT(ascan_stack_cuda_host_tensor, StackHostTensorEndpoint, stack_format_executor_t, cuda_host_tensor_t);

        BIND_CUDA_ENDPOINT(ascan_stack_cuda_device_tensor, StackDeviceTensorEndpoint, stack_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(ascan_radial_cuda_device_tensor, RadialDeviceTensorEndpoint, radial_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(ascan_spiral_cuda_device_tensor, AscanSpiralDeviceTensorEndpoint, spiral_format_executor_t, cuda_device_tensor_t);
        BIND_CUDA_ENDPOINT(ascan_position_cuda_device_tensor, PositionDeviceTensorEndpoint, position_format_executor_t, cuda_device_tensor_t);
    }
}

template<typename C>
static void setup_storage_endpoint(py::class_<C, sp<C>>& c) {
    bind_notify(c);

    RO_ACC(storage);
}

#define BIND_STREAM_STORAGE_ENDPOINT(type) \
    c.def(py::init<sp<vortex::simple_stream_t<type>>, sp<spdlog::logger>>(), "storage"_a, "log"_a = nullptr); \
    \
    setup_storage_endpoint(c);

#define BIND_STREAM_STORAGE_STREAMS_ENDPOINT(stream, type, name) \
    { \
        using C = vortex::endpoint::streams_stream_storage<vortex::cast(engine_t::block_t::stream_index_t::stream), type>; \
        CLS_PTR(name); \
    \
        BIND_STREAM_STORAGE_ENDPOINT(type); \
    }

#define BIND_STACK_STORAGE_ENDPOINT(stack_t, T) \
    c.def(py::init<sp<vortex::stack_t<T>>, sp<spdlog::logger>>(), "storage"_a, "log"_a = nullptr); \
    c.def(py::init<sp<vortex::stack_format_executor_t>, sp<vortex::stack_t<T>>, vortex::endpoint::buffer_strategy_t, sp<spdlog::logger>>(), "executor"_a, "storage"_a, "buffer_strategy"_a = vortex::endpoint::buffer_strategy_t::volume, "log"_a = nullptr); \
    \
    setup_storage_endpoint(c); \
    \
    RO_ACC(executor);

#define BIND_STACK_STORAGE_STREAMS_ENDPOINT(stream, type, name) \
    { \
        using C = vortex::endpoint::streams_stack_storage<vortex::cast(engine_t::block_t::stream_index_t::stream), type>; \
        CLS_PTR(name); \
    \
        BIND_STACK_STORAGE_ENDPOINT(simple_stack_t, type); \
    }

#define BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(stream, type, name) \
    { \
        using C = vortex::endpoint::streams_hdf5_stack_storage<vortex::cast(engine_t::block_t::stream_index_t::stream), type>; \
        CLS_PTR(name); \
    \
        BIND_STACK_STORAGE_ENDPOINT(hdf5_stack_t, type); \
    }

template<typename engine_t>
static void bind_stream_and_stack_storage_endpoints(py::module& m) {
    {
        {
            using C = vortex::endpoint::spectra_stream_storage<typename engine_t::acquire_element_t>;
            CLS_PTR(SpectraStreamEndpoint);

            BIND_STREAM_STORAGE_ENDPOINT(typename engine_t::acquire_element_t);
        }
        {
            using C = vortex::endpoint::ascan_stream_storage<typename engine_t::process_element_t>;
            CLS_PTR(AscanStreamEndpoint);

            BIND_STREAM_STORAGE_ENDPOINT(typename engine_t::process_element_t);
        }

        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(counter, vortex::counter_t, CounterStreamEndpoint);
        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(galvo_actual, double, GalvoActualStreamEndpoint);
        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(galvo_target, double, GalvoTargetStreamEndpoint);
        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(sample_actual, double, SampleActualStreamEndpoint);
        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(sample_target, double, SampleTargetStreamEndpoint);
        BIND_STREAM_STORAGE_STREAMS_ENDPOINT(strobes, typename engine_t::digital_element_t, StrobesStreamEndpoint);
    }

    {
        {
            using C = vortex::endpoint::spectra_stack_storage<typename engine_t::acquire_element_t>;
            CLS_PTR(SpectraStackEndpoint);

            BIND_STACK_STORAGE_ENDPOINT(simple_stack_t, typename engine_t::acquire_element_t);
        }
        {
            using C = vortex::endpoint::ascan_stack_storage<typename engine_t::process_element_t>;
            CLS_PTR(AscanStackEndpoint);

            BIND_STACK_STORAGE_ENDPOINT(simple_stack_t, typename engine_t::process_element_t);
        }

        BIND_STACK_STORAGE_STREAMS_ENDPOINT(counter, vortex::counter_t, CounterStackEndpoint);
        BIND_STACK_STORAGE_STREAMS_ENDPOINT(galvo_actual, double, GalvoActualStackEndpoint);
        BIND_STACK_STORAGE_STREAMS_ENDPOINT(galvo_target, double, GalvoTargetStackEndpoint);
        BIND_STACK_STORAGE_STREAMS_ENDPOINT(sample_actual, double, SampleActualStackEndpoint);
        BIND_STACK_STORAGE_STREAMS_ENDPOINT(sample_target, double, SampleTargetStackEndpoint);
        BIND_STACK_STORAGE_STREAMS_ENDPOINT(strobes, typename engine_t::digital_element_t, StrobesStackEndpoint);
    }

#if defined(VORTEX_ENABLE_HDF5)
    {
        {
            using C = vortex::endpoint::spectra_hdf5_stack_storage<typename engine_t::acquire_element_t>;
            CLS_PTR(SpectraHDF5StackEndpoint);

            BIND_STACK_STORAGE_ENDPOINT(hdf5_stack_t, typename engine_t::acquire_element_t);
        }
        {
            using C = vortex::endpoint::ascan_hdf5_stack_storage<typename engine_t::process_element_t>;
            CLS_PTR(AscanHDF5StackEndpoint);

            BIND_STACK_STORAGE_ENDPOINT(hdf5_stack_t, typename engine_t::process_element_t);
        }

        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(counter, vortex::counter_t, CounterHDF5StackEndpoint);
        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(galvo_actual, double, GalvoActualHDF5StackEndpoint);
        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(galvo_target, double, GalvoTargetHDF5StackEndpoint);
        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(sample_actual, double, SampleActualHDF5StackEndpoint);
        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(sample_target, double, SampleTargetHDF5StackEndpoint);
        BIND_HDF5_STACK_STORAGE_STREAMS_ENDPOINT(strobes, typename engine_t::digital_element_t, StrobesHDF5StackEndpoint);
    }
#endif

}

#endif

template<typename T>
static void bind_source(py::module& m, const std::string& name) {
    using C = vortex::engine::source_t<T>;
    auto c = py::class_<C>(m, name.c_str());

    c.def(py::init<size_t, size_t, T, T>(),
        "triggers_per_second"_a = 100'000,
        "clock_rising_edges_per_trigger"_a = 1376,
        "duty_cycle"_a = 0.5,
        "imaging_depth_meters"_a = 0.01
    );

    RW_VAR(triggers_per_second);
    RW_VAR(clock_rising_edges_per_trigger);
    RW_VAR(duty_cycle);
    RW_VAR(imaging_depth_meters);

    RO_ACC(has_clock);
    RW_ACC_XT(clock_edges_seconds);

    c.def("__repr__", [](const C& o) {
        return fmt::format("Source(triggers_per_second={}, clock_rising_edges_per_second={}, duty_cycle={}, imaging_depth_meters={})", o.triggers_per_second, o.clock_rising_edges_per_trigger, o.duty_cycle, o.imaging_depth_meters);
    });

    SHALLOW_COPY();
}

#if defined(VORTEX_ENABLE_ENGINE)

template<typename engine_t, typename scan_t, typename C>
static void setup_online_scan_queue(py::class_<C, sp<C>>& c) {
    using callback_t = typename engine_t::scan_queue_t::scan_callback_t;
    c.def("append", [](C& o, sp<scan_t>& scan, callback_t callback, vortex::marker::scan_boundary marker) {
        o.append(scan, std::move(callback), std::move(marker));
    }, "scan"_a, "callback"_a = callback_t(), "marker"_a = vortex::marker::scan_boundary());
}

template<typename engine_t, typename scan_t, typename C>
static void setup_scan_queue(py::class_<C, sp<C>>& c) {
    using callback_t = typename engine_t::scan_queue_t::scan_callback_t;

    c.def("append", [](C& o, sp<scan_t>& scan, callback_t callback, vortex::marker::scan_boundary marker) {
        py::gil_scoped_release gil;
        o.append(scan, std::move(callback), std::move(marker));
    }, "scan"_a, "callback"_a = callback_t(), "marker"_a = vortex::marker::scan_boundary());

    c.def("interrupt", [](C& o, sp<scan_t>& scan, callback_t callback, vortex::marker::scan_boundary marker) {
        py::gil_scoped_release gil;
        o.interrupt(scan, std::move(callback), std::move(marker));
    }, "scan"_a, "callback"_a = callback_t(), "marker"_a = vortex::marker::scan_boundary());
}

template<typename C>
static void setup_flagged_strobe(py::class_<C>& c) {
    c.def(py::init<size_t, vortex::engine::strobe::polarity_t, size_t, size_t, vortex::engine::strobe::flags_t>(), "line"_a = 0, "polarity"_a = vortex::engine::strobe::polarity_t::high, "duration"_a = 10, "delay"_a = 0, "flags"_a = vortex::default_marker_flags_t::all());

    RW_VAR(line);
    RW_VAR(polarity);
    RW_VAR(duration);
    RW_VAR(delay);
    RW_VAR(flags);
}

template<typename engine_t, typename T>
static void bind_node_binder(py::module& m, const std::string& name) {
    m.def(name.c_str(), [m](py::sequence nodes) {
        auto bind = m.attr("_bind");
        T root;

        // inspect all arguments
        for (auto node : nodes) {
            try {
                // check if already bound
                auto n = node.cast<typename engine_t::config_t::node>();

                // binding is already done
                root.subgraph.push_back(n);
            } catch (const py::cast_error&) {
                // bind now
                auto v = bind(node).cast<typename engine_t::processor_t>();
                root.subgraph.push_back(v);
            }
        }

        return root;
    });
}

template<typename engine_t>
static void bind_engine_engine(py::module& m) {
    {
        using C = typename engine_t::scan_queue_t;
        using point_t = xtensor_to_pytensor_t<typename C::point_t>;
        CLS_PTR(ScanQueue);

        c.def(py::init());
        c.def(py::init([](vortex::counter_t sample, const point_t& position, const point_t& velocity) {
            return std::make_shared<C>(sample, position, velocity);
        }));

        setup_scan_queue<engine_t, vortex::raster_scan_t>(c);
        setup_scan_queue<engine_t, vortex::repeated_raster_scan_t>(c);
        setup_scan_queue<engine_t, vortex::radial_scan_t>(c);
        setup_scan_queue<engine_t, vortex::repeated_radial_scan_t>(c);
        setup_scan_queue<engine_t, vortex::spiral_scan_t>(c);
        setup_scan_queue<engine_t, vortex::freeform_scan_t>(c);

        c.def("generate", [](C& o, std::vector<vortex::default_marker_t>& markers, const vortex::cpu_view_t<typename C::element_t>& buffer, bool zero_order_hold) {

            py::gil_scoped_release gil;
            return o.generate(markers, buffer, zero_order_hold);

        }, "markers"_a, "buffer"_a, "zero_order_hold"_a = true);

        c.def("reset", py::overload_cast<>(&C::reset));
        c.def("reset", [](C& o, vortex::counter_t sample, const point_t& position, const point_t& velocity) {
            o.reset(sample, position, velocity);
        }, "sample"_a, "position"_a, "velocity"_a);

        FXN(clear);

        RW_MUT(empty_callback);

        {
            using C2 = typename engine_t::scan_queue_t::online_scan_queue_t;
            auto c2 = py::class_<C2, sp<C2>>(c, "OnlineScanQueue");

            setup_online_scan_queue<engine_t, vortex::raster_scan_t>(c2);
            setup_online_scan_queue<engine_t, vortex::repeated_raster_scan_t>(c2);
            setup_online_scan_queue<engine_t, vortex::radial_scan_t>(c2);
            setup_online_scan_queue<engine_t, vortex::repeated_radial_scan_t>(c2);
            setup_online_scan_queue<engine_t, vortex::spiral_scan_t>(c2);
            setup_online_scan_queue<engine_t, vortex::freeform_scan_t>(c2);
        }

        py::enum_<typename C::event_t>(c, "Event")
            .value("Start", C::event_t::start)
            .value("Finish", C::event_t::finish)
            .value("Interrupt", C::event_t::interrupt)
            .value("Abort", C::event_t::abort);
    }

    bind_node_binder<engine_t, typename engine_t::config_t::divide_t>(m, "divide");
    bind_node_binder<engine_t, typename engine_t::config_t::cycle_t>(m, "cycle");

    // shallow bindings since this functionality is hidden from the user
    py::class_<typename engine_t::config_t::divide_t>(m, "_Divide");
    py::class_<typename engine_t::config_t::cycle_t>(m, "_Cycle");

    {
        using C = typename engine_t::config_t;
        CLS_VAL(EngineConfig);

        c.def(py::init());

        RW_VAR(records_per_block);
        RW_VAR(blocks_to_allocate);
        RW_VAR(preload_count);

        RW_VAR(blocks_to_acquire);
        RW_VAR(post_scan_records);

        RW_VAR(scanner_warp);
        RW_VAR(galvo_output_channels);
        RW_VAR(galvo_input_channels);

        RW_VAR(lead_strobes);
        RW_VAR(lead_marker);

        RW_VAR(strobes);

        c.def("add_acquisition", [m](C& o, py::object acquisition, py::sequence processors, bool preload, bool master) {
            auto bind = m.attr("_bind");

            auto k = bind(acquisition).cast<typename engine_t::acquisition_t>();
            auto& config = o.acquisitions[k];

            // update options
            config.preload = preload;
            config.master = master;

            // let Python handle unpacking each py::object
            using divide_t = typename engine_t::config_t::divide_t;
            auto other = m.attr("divide")(processors).cast<divide_t>();

            std::visit(vortex::overloaded{
                [&](divide_t& n) {
                    // merge with existing divide
                    std::move(other.subgraph.begin(), other.subgraph.end(), std::back_inserter(n.subgraph));
                },
                [&](auto& n) {
                    // append graph to any existing one by dividing
                    config.graph = engine_t::config_t::divide(config.graph, other);
                }
            }, config.graph);

        }, "acquisition"_a, "processors"_a, "preload"_a = true, "master"_a = true);

        c.def("add_processor", [m](C& o, py::object processor, py::sequence formatters) {
            auto bind = m.attr("_bind");

            auto k = bind(processor).cast<typename engine_t::processor_t>();
            auto& config = o.processors[k];

            for (auto formatter : formatters) {
                auto v = bind(formatter).cast<typename engine_t::formatter_t>();
                config.graph.push_back(v);
            }
        });

        c.def("add_formatter", [m](C& o, py::object formatter, py::sequence endpoints) {
            auto bind = m.attr("_bind");

            auto k = bind(formatter).cast<typename engine_t::formatter_t>();
            auto& config = o.formatters[k];

            for (auto endpoint : endpoints) {
                auto v = bind(endpoint).cast<typename engine_t::endpoint_t>();
                config.graph.push_back(v);
            }
        });

        c.def("add_io", [m](C& o, py::object io, bool preload, bool master, size_t lead_samples) {
            auto bind = m.attr("_bind");

            auto k = bind(io).cast<typename engine_t::io_t>();
            auto& config = o.ios[k];

            config.preload = preload;
            config.master = master;
            config.lead_samples = lead_samples;
        }, "io"_a, "preload"_a = true, "master"_a = false, "lead_samples"_a = 0);

        FXN(validate);

        SHALLOW_COPY();
    }

    {
        using C = typename engine_t::engine_status;
        CLS_VAL(EngineStatus);

        RW_VAR(active);
        RW_VAR(dispatched_blocks);
        RW_VAR(inflight_blocks);
        RW_VAR(dispatch_completion);
        RW_VAR(block_utilization);

        SHALLOW_COPY();

        c.def("__repr__", [](C& o) {
            return fmt::format("EngineStatus(active={}, dispatched={}, inflight={}, completion={}, utilization={})", o.active, o.dispatched_blocks, o.inflight_blocks, o.dispatch_completion, o.block_utilization);
        });
    }

    {
        using C = engine_t;
        CLS_PTR(Engine);

        c.def(py::init<sp<spdlog::logger>>(), "logger"_a = nullptr);

        FXN_GIL(initialize);
        FXN_GIL(prepare);

        FXN_GIL(wait);
        c.def("wait_for", [](C& o, double dt) {
            return o.wait_for(std::chrono::duration_cast<std::chrono::microseconds>(vortex::seconds(dt)));
        }, py::call_guard<py::gil_scoped_release>(), "timeout"_a);

        FXN_GIL(start);
        FXN_GIL(stop);
        FXN_GIL(shutdown, "interrupt"_a = false);
        RO_ACC_GIL(done);
        FXN_GIL(status);

        RW_ACC(scan_queue);
        RO_ACC(config);

        RW_MUT_GIL(event_callback);
        RW_MUT_GIL(job_callback);

        {
            using C = vortex::engine::event_t;
            py::enum_<C>(c, "Event")
                .value("Launch", C::launch)
                .value("Start", C::start)
                .value("Run", C::run)
                .value("Stop", C::stop)
                .value("Complete", C::complete)
                .value("Shutdown", C::shutdown)
                .value("Exit", C::exit)
                .value("Error", C::error)
                .value("Abort", C::abort);
        }

        auto engine = c;
        {
            using C = vortex::engine::timing_t;
            CLS_VAL(JobTiming);

            c.def(py::init());

            RW_VAR(create);
            RW_VAR(service);
            RW_VAR(scan);
            RW_VAR(acquire);
            RW_VAR(process);
            RW_VAR(format);
            RW_VAR(recycle);
        }

        {
            using C = vortex::engine::session_status_t;
            CLS_VAL(SessionStatus);

            c.def(py::init());

            RW_VAR(allocated);
            RW_VAR(inflight);
            RW_VAR(dispatched);
            RW_VAR(limit);

            RO_ACC(available);
            RO_ACC(utilization);
            RO_ACC(progress);
        }
    }

    // strobes

    {
        using C = vortex::engine::strobe::polarity_t;
        py::enum_<C>(m, "Polarity")
            .value("Low", C::low)
            .value("High", C::high);
    }
    {
        using C = vortex::engine::strobe::sample;
        CLS_VAL(SampleStrobe);

        c.def(py::init<size_t, size_t, vortex::engine::strobe::polarity_t, size_t, size_t>(), "line"_a = 0, "divisor"_a = 2, "polarity"_a = vortex::engine::strobe::polarity_t::high, "duration"_a = 1, "phase"_a = 0);

        RW_VAR(line);
        RW_VAR(polarity);
        RW_VAR(divisor);
        RW_VAR(phase);
        RW_VAR(duration);
    }
    {
        using C = vortex::engine::strobe::segment;
        CLS_VAL(SegmentStrobe);
        setup_flagged_strobe(c);
    }
    {
        using C = vortex::engine::strobe::volume;
        CLS_VAL(VolumeStrobe);
        setup_flagged_strobe(c);
    }
    {
        using C = vortex::engine::strobe::scan;
        CLS_VAL(ScanStrobe);
        setup_flagged_strobe(c);
    }
    {
        using C = vortex::engine::strobe::event;
        CLS_VAL(EventStrobe);
        setup_flagged_strobe(c);
    }

    py::bind_vector<std::vector<vortex::engine::strobe_t>>(m, "StrobeList");

    // shallow bindings for adapters since this functionality is hidden from users
    {
        py::class_<typename engine_t::acquisition_t>(m, "_AcquisitionAdapter");
        py::class_<typename engine_t::processor_t>(m, "_ProcessorAdapter");
        py::class_<typename engine_t::formatter_t>(m, "_FormatterAdapter");
        py::class_<typename engine_t::endpoint_t>(m, "_EndpointAdapter");
        py::class_<typename engine_t::io_t>(m, "_IOAdapter");
    }

    {
        using C = typename engine_t::block_t;
        CLS_PTR(Block);

        c.def(py::init());

        RW_VAR(id);
        RW_VAR(timestamp);
        RW_VAR(sample);
        RW_VAR(length);
        RW_VAR(markers);

        RO_VAR(counter);
        RO_VAR(galvo_target);
        RO_VAR(sample_target);
        RO_VAR(galvo_actual);
        RO_VAR(sample_actual);
        RO_VAR(strobes);

        py::enum_<typename C::stream_index_t>(c, "StreamIndex")
            .value("Counter", C::stream_index_t::counter)
            .value("GalvoTarget", C::stream_index_t::galvo_target)
            .value("SampleTarget", C::stream_index_t::sample_target)
            .value("GalvoActual", C::stream_index_t::galvo_actual)
            .value("SampleActual", C::stream_index_t::sample_actual)
            .value("Strobes", C::stream_index_t::strobes);

        // TODO: bind the rest of the members
    }

}

#endif

void bind_engine(py::module& root) {
    auto m = root.def_submodule("engine");

    bind_utility_functions<double>(m);

#if defined(VORTEX_ENABLE_ENGINE)
    {
        using C = vortex::endpoint::null;
        CLS_PTR(NullEndpoint);

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        bind_notify(c);
    }
    {
        using C = vortex::endpoint::streams_stack_cuda_host_tensor<vortex::cast(engine_t::block_t::stream_index_t::counter), vortex::counter_t>;
        CLS_PTR(CounterStackHostTensorEndpoint);

        c.def(py::init<sp<vortex::stack_format_executor_t>, std::vector<size_t>, sp<spdlog::logger>>(), "executor"_a, "shape"_a, "log"_a = nullptr);

        setup_cuda_endpoint(c);
    }
    {
        using C = vortex::endpoint::streams_stack_cuda_host_tensor<vortex::cast(engine_t::block_t::stream_index_t::galvo_actual), double>;
        CLS_PTR(GalvoActualStackHostTensorEndpoint);

        c.def(py::init<sp<vortex::stack_format_executor_t>, std::vector<size_t>, sp<spdlog::logger>>(), "executor"_a, "shape"_a, "log"_a = nullptr);

        setup_cuda_endpoint(c);
    }

    bind_memory_endpoints<engine_t>(m);

    //
    // storage endpoints
    //

    py::enum_<vortex::endpoint::buffer_strategy_t>(m, "BufferStrategy")
        .value("Block", vortex::endpoint::buffer_strategy_t::none)
        .value("Segment", vortex::endpoint::buffer_strategy_t::segment)
        .value("Volume", vortex::endpoint::buffer_strategy_t::volume);

    {
        using C = vortex::endpoint::stream_dump_storage;
        CLS_PTR(StreamDumpStorage);

        c.def(py::init<sp<vortex::stream_dump_t>, size_t>(), "storage"_a, "lead_samples"_a = 0);

        RO_ACC(storage);
    }
    {
        using C = vortex::endpoint::marker_log_storage;
        CLS_PTR(MarkerLogStorage);

        c.def(py::init<sp<vortex::marker_log_t>>(), "storage"_a);

        RO_ACC(storage);
    }

    {
        using C = vortex::endpoint::broct_storage;
        CLS_PTR(BroctStorageEndpoint);

        c.def(py::init<sp<vortex::broct_format_executor_t>, sp<vortex::broct_storage_t>, vortex::endpoint::buffer_strategy_t, sp<spdlog::logger>>(), "executor"_a, "storage"_a, "buffer_strategy"_a = vortex::endpoint::buffer_strategy_t::volume, "log"_a = nullptr);

        setup_storage_endpoint(c);

        RO_ACC(executor);
    }

    bind_stream_and_stack_storage_endpoints<engine_t>(m);

    //
    // engine and adapters
    //

    {
        bind_engine_engine<engine_t>(m);

        m.def("_bind", [](sp<vortex::null_acquisition_t<uint16_t>>& a) {
            return vortex::engine::bind::acquisition<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::file_acquisition_t<uint16_t>>& a) {
            return vortex::engine::bind::acquisition<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::null_processor_t>& a) {
            return vortex::engine::bind::processor<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::copy_processor_t<engine_t::acquire_element_t, engine_t::process_element_t>>& a) {
            return vortex::engine::bind::processor<block_t>(a);
        });
#if defined(VORTEX_ENABLE_FFTW)
        m.def("_bind", [](sp<vortex::cpu_processor_t<engine_t::acquire_element_t, engine_t::process_element_t>>& a) {
            return vortex::engine::bind::processor<block_t>(a);
        });
#endif
        m.def("_bind", [](sp<vortex::cuda_processor_t<engine_t::acquire_element_t, engine_t::process_element_t>>& a) {
            return vortex::engine::bind::processor<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::null_io_t>& a) {
            return vortex::engine::bind::io<block_t>(a);
        });
#if defined(VORTEX_ENABLE_ASIO)
        m.def("_bind", [](sp<vortex::machdsp_io_t>& a) {
            return vortex::engine::bind::io<block_t>(a);
        });
#endif
        m.def("_bind", [](sp<vortex::format_planner_t>& a) {
            return vortex::engine::bind::formatter<block_t>(a);
        });

        m.def("_bind", [](sp<vortex::endpoint::null>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
        {
            using C = vortex::endpoint::streams_stack_cpu_tensor<vortex::cast(block_t::stream_index_t::counter), vortex::counter_t>;
            m.def("_bind", [](sp<C>& a) { return vortex::engine::bind::template endpoint<block_t>(a); });
        }
        {
            using C = vortex::endpoint::streams_stack_cpu_tensor<vortex::cast(block_t::stream_index_t::galvo_actual), double>;
            m.def("_bind", [](sp<C>& a) { return vortex::engine::bind::template endpoint<block_t>(a); });
        }
        {
            using C = vortex::endpoint::streams_stack_cuda_host_tensor<vortex::cast(block_t::stream_index_t::counter), vortex::counter_t>;
            m.def("_bind", [](sp<C>& a) { return vortex::engine::bind::template endpoint<block_t>(a); });
        }
        {
            using C = vortex::endpoint::streams_stack_cuda_host_tensor<vortex::cast(block_t::stream_index_t::galvo_actual), double>;
            m.def("_bind", [](sp<C>& a) { return vortex::engine::bind::template endpoint<block_t>(a); });
        }
        m.def("_bind", [](sp<vortex::endpoint::broct_storage>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });

        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::counter), vortex::counter_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::galvo_actual), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::galvo_target), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::sample_actual), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::sample_target), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stream_storage<vortex::cast(block_t::stream_index_t::strobes), engine_t::digital_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::spectra_stream_storage<engine_t::acquire_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::ascan_stream_storage<engine_t::process_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });

        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::counter), vortex::counter_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::galvo_actual), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::galvo_target), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::sample_actual), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::sample_target), double>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::streams_stack_storage<vortex::cast(block_t::stream_index_t::strobes), engine_t::digital_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::spectra_stack_storage<engine_t::acquire_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });
        m.def("_bind", [](sp<vortex::endpoint::ascan_stack_storage<engine_t::process_element_t>>& a) { return vortex::engine::bind::endpoint<block_t>(a); });

#if defined(VORTEX_ENABLE_HDF5)
        m.def("_bind", [](sp<vortex::endpoint::streams_hdf5_stack_storage<vortex::cast(block_t::stream_index_t::counter), vortex::counter_t>>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::endpoint::streams_hdf5_stack_storage<vortex::cast(block_t::stream_index_t::galvo_actual), double>>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::endpoint::spectra_hdf5_stack_storage<engine_t::acquire_element_t>>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::endpoint::ascan_hdf5_stack_storage<engine_t::process_element_t>>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
#endif
        m.def("_bind", [](sp<vortex::endpoint::stream_dump_storage>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
        m.def("_bind", [](sp<vortex::endpoint::marker_log_storage>& a) {
            return vortex::engine::bind::endpoint<block_t>(a);
        });
    }

#endif

    bind_source<double>(m, "Source");

    auto sources = m.def_submodule("source");
    sources.attr("Axsun100k") = vortex::engine::source_t<double>{ 100'000, 1376, 0.50, 3.7e-3 }; // Axsun 100k
    sources.attr("Axsun200k") = vortex::engine::source_t<double>{ 200'000, 1024, 0.61, 3.7e-3 }; // Axsun 200k
    sources.attr("ThorlabsVCSEL400k") = vortex::engine::source_t<double>{ 400'000, 1024, 0.54, 0 }; // Thorlabs VCSEL 400k
}
