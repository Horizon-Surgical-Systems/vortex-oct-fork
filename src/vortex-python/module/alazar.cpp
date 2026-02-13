#include <vortex/acquire.hpp>
#include <vortex/io.hpp>
#include <vortex/engine/source.hpp>
#include <vortex/engine/clock.hpp>

#include <vortex/driver/alazar/board.hpp>
#include <vortex/driver/alazar/db.hpp>

#include <vortex-python/module/helper.hpp>
#include <vortex-python/bind/acquire.hpp>
#include <vortex-python/bind/io.hpp>
#include <vortex-python/bind/engine.hpp>
#include <vortex-python/bind/memory.hpp>
#include <vortex-python/bind/exception.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<vortex::alazar_acquisition_t::config_t::option_t>);
PYBIND11_MAKE_OPAQUE(std::vector<vortex::alazar::input_t>);

// namespace pybind11 {
//     namespace detail {
//         template <typename... Ts>
//         struct type_caster<vortex::acquire::clock_t<Ts...>> : variant_caster<vortex::acquire::clock_t<Ts...>> {};
//         template <typename... Ts>
//         struct type_caster<vortex::acquire::trigger_t<Ts...>> : variant_caster<vortex::acquire::trigger_t<Ts...>> {};
//         template <typename... Ts>
//         struct type_caster<vortex::acquire::option_t<Ts...>> : variant_caster<vortex::acquire::option_t<Ts...>> {};
//     }
// }

static void _bind_driver(py::module& m) {

    py::enum_<vortex::alazar::channel_t>(m, "Channel")
        .value("A", vortex::alazar::channel_t::A)
        .value("B", vortex::alazar::channel_t::B)
        .value("C", vortex::alazar::channel_t::C)
        .value("D", vortex::alazar::channel_t::D)
        .value("E", vortex::alazar::channel_t::E)
        .value("F", vortex::alazar::channel_t::F)
        .value("G", vortex::alazar::channel_t::G)
        .value("H", vortex::alazar::channel_t::H)
        .value("I", vortex::alazar::channel_t::I)
        .value("J", vortex::alazar::channel_t::J)
        .value("K", vortex::alazar::channel_t::K)
        .value("L", vortex::alazar::channel_t::L)
        .value("M", vortex::alazar::channel_t::M)
        .value("N", vortex::alazar::channel_t::N)
        .value("O", vortex::alazar::channel_t::O)
        .value("P", vortex::alazar::channel_t::P);

    py::enum_<vortex::alazar::coupling_t>(m, "Coupling")
        .value("DC", vortex::alazar::coupling_t::DC)
        .value("AC", vortex::alazar::coupling_t::AC);

    py::enum_<vortex::alazar::clock_edge_t>(m, "ClockEdge")
        .value("Rising", vortex::alazar::clock_edge_t::rising)
        .value("Falling", vortex::alazar::clock_edge_t::falling);

    py::enum_<vortex::alazar::trigger_slope_t>(m, "TriggerSlope")
        .value("Positive", vortex::alazar::trigger_slope_t::positive)
        .value("Negative", vortex::alazar::trigger_slope_t::negative);

    m.attr("InfiniteAcquisition") = vortex::alazar::infinite_acquisition;
    m.attr("TriggerRangeTTL") = vortex::alazar::trigger_range_TTL;

    {
        using C = vortex::alazar::board_t::info_t::type_t;
        CLS_VAL(InfoType);

        RW_VAR(id);
        RW_VAR(model);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::pcie_t;
        CLS_VAL(InfoPCIe);

        RW_VAR(speed);
        RW_VAR(width);
        RW_VAR(speed_gbps);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::input_combination_t;
        CLS_VAL(InfoInputCombination);

        RW_VAR(impedance_ohms);
        RW_VAR(input_range_millivolts);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::features_t;
        CLS_VAL(InfoFeatures);

        RW_VAR(set_external_clock_level);
        RW_VAR(adc_calibration_sampling_rate);
        RW_VAR(dual_edge_sampling);
        RW_VAR(sample_skipping);
        RW_VAR(dual_port_memory);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::dsp_t::version_t;
        CLS_VAL(InfoDspVersion);

        RW_VAR(major);
        RW_VAR(minor);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::dsp_t;
        CLS_VAL(InfoDsp);

        RW_VAR(type);
        RW_VAR(version);
        RW_VAR(max_record_length);
        RW_VAR(supported_channels);
        RW_VAR(fft_time_domain_supported);
        RW_VAR(fft_subtractor_supported);

        FXN(max_trigger_rate_per_fft_length);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t::dac_t;
        CLS_VAL(InfoDac);

        RW_VAR(sequence_count);
        RW_VAR(slot_sizes);
        RW_VAR(volts_per_lsb);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::board_t::info_t;
        CLS_VAL(Info);

        RW_VAR(system_index);
        RW_VAR(board_index);
        RW_VAR(serial_number);
        RW_VAR(calibration_date);
        RW_VAR(onboard_memory_size);
        RW_VAR(bits_per_sample);
        RW_VAR(type);
        RW_VAR(pcie);

        RO_VAR(supported_channels);
        RW_VAR(supported_sampling_rates);
        RO_ACC(max_sampling_rate);

        RW_VAR(supported_input_combinations);

        RW_VAR(max_pretrigger_samples);
        RW_VAR(min_samples_per_record);

        RW_VAR(sample_alignment_divisor);
        FXN(nearest_aligned_samples_per_record);
        FXN(smallest_aligned_samples_per_record);

        RW_VAR(features);
        RW_VAR(dsp);
        RW_VAR(dac);

        SHALLOW_COPY();
    }

    {
        using C = vortex::alazar::board_t;
        CLS_PTR(Board);

        c.def(py::init(), doc(c, "__init__"));
        c.def(py::init<U32, U32>(), doc(c, "__init__"));

        c.def_property_readonly("handle", [](C& o) { return reinterpret_cast<uintptr_t>(o.handle()); });
        RO_ACC(valid)

            RO_ACC(info);

        // TODO: supply other methods
    }

    m.def("enumerate", vortex::alazar::enumerate);

}

template<typename C>
static void setup_alazar_config(py::class_<C>& c) {
    RW_VAR(device);
    RW_VAR(clock);
    RW_VAR(trigger);
    RW_VAR(inputs);
    RW_VAR(options);

    RW_VAR(resampling);

    RW_VAR(acquire_timeout);
    RW_VAR(stop_on_error);

    RO_ACC(channels_per_sample);

    RO_ACC(bytes_per_multisample);
    RO_ACC(channel_mask);

    RW_ACC(samples_per_second);
    RO_ACC(samples_per_second_is_known);

    RO_ACC(recommended_minimum_records_per_block);
}

template<typename C>
static std::string alazar_inner_repr(const C& v) {
    std::string result = fmt::format("(\n\
    shape = {},\n\
    channels_per_sample = {},\n\
    samples_per_record = {},\n\
    records_per_block = {},\n\
    device = {},\n\
    clock = {},\n\
    trigger = {},\n\
    inputs = {},\n\
    options = {},\n\
    acquire_timeout = {},\n\
    stop_on_error = {},\n\
    bytes_per_multisample = {},\n\
    channel_mask = {},\n\
    samples_per_second = {},\n\
    samples_per_second_is_known = {},\n\
    recommended_minimum_records_per_block = {},",
        PY_REPR(v.shape()),
        v.channels_per_sample(),
        v.samples_per_record(),
        v.records_per_block(),
        PY_REPR(v.device),
        PY_REPR(v.clock),
        PY_REPR(v.trigger),
        PY_REPR(v.inputs),
        PY_REPR(v.options),
        PY_REPR(v.acquire_timeout),
        PY_REPR(v.stop_on_error),
        v.bytes_per_multisample(),
        PY_REPR(v.channel_mask()),
        v.samples_per_second(),
        PY_REPR(v.samples_per_second_is_known()),
        v.recommended_minimum_records_per_block()
    );
    return result;
}
static void _bind_acquire_config(py::module& m) {

    py::bind_vector<std::vector<vortex::alazar_acquisition_t::config_t::option_t>>(m, "VectorOption")
        .def("__repr__", [](const std::vector<vortex::alazar_acquisition_t::config_t::option_t>& v) {
        return list_repr(v, "VectorOption");
            });
    py::bind_vector<std::vector<vortex::alazar::input_t>>(m, "VectorInput")
        .def("__repr__", [](const std::vector<vortex::alazar::input_t>& v) {
        return list_repr(v, "VectorInput");
            });

    {
        using C = vortex::alazar::clock::internal_t;
        CLS_VAL(InternalClock);

        c.def(py::init<size_t>(), "samples_per_second"_a = 800'000'000, doc(c, "__init__"));
        c.def("__repr__", [](const C& v) { return fmt::format("InternalClock(samples_per_second={})", PY_REPR(v.samples_per_second)); });

        RW_VAR(samples_per_second);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::clock::external_t;
        CLS_VAL(ExternalClock);

        c.def(py::init<float, vortex::alazar::coupling_t, vortex::alazar::clock_edge_t, bool>(),
            "level_ratio"_a = 0.5f, "coupling"_a = vortex::alazar::coupling_t::AC, "edge"_a = vortex::alazar::clock_edge_t::rising, "dual"_a = false, doc(c, "__init__"));
        c.def("__repr__", [](const C& v) {
            return fmt::format("ExternalClock(coupling={}, dual={}, edge={}, level_ratio={:0.2f})",
                PY_REPR(v.coupling), PY_REPR(v.dual), PY_REPR(v.edge), v.level_ratio); });

        RW_VAR(level_ratio);
        RW_VAR(coupling);
        RW_VAR(edge);
        RW_VAR(dual);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::trigger::single_external_t;
        CLS_VAL(SingleExternalTrigger);

        c.def(py::init<size_t, float, size_t,vortex::alazar::trigger_slope_t,vortex::alazar::coupling_t>(),
            "range_millivolts"_a = 2500, "level_ratio"_a = 0.09f, "delay_samples"_a = 80, "slope"_a = vortex::alazar::trigger_slope_t::positive,
            "coupling"_a = vortex::alazar::coupling_t::DC, doc(c, "__init__"));
        c.def("__repr__",
            [](const C& v) {
                return fmt::format("SingleExternalTrigger(range_millivolts={}, level_ratio={:0.2f}, delay_samples={}, slope={}, coupling={})",
                    PY_REPR(v.range_millivolts), v.level_ratio, PY_REPR(v.delay_samples), PY_REPR(v.slope), PY_REPR(v.coupling));
            });

        RW_VAR(range_millivolts);
        RW_VAR(level_ratio);
        RW_VAR(delay_samples);
        RW_VAR(slope);
        RW_VAR(coupling);
    }
    {
        using C = vortex::alazar::trigger::dual_external_t;
        CLS_VAL(DualExternalTrigger);

        c.def(py::init<size_t, std::array<float, 2>, size_t, vortex::alazar::trigger_slope_t, vortex::alazar::coupling_t>(),
            "range_millivolts"_a = 2500, "level_ratios"_a = std::array<float, 2>{0.09f, 0.09f}, "delay_samples"_a = 80,
            "initial_slope"_a = vortex::alazar::trigger_slope_t::positive, "coupling"_a = vortex::alazar::coupling_t::DC, doc(c, "__init__"));
        c.def("__repr__",
            [](const C& v) {
                return fmt::format("DualExternalTrigger(range_millivolts={}, level_ratios={}, delay_samples={}, initial_slope={}, coupling={})",
                    PY_REPR(v.range_millivolts), PY_REPR(v.level_ratios), PY_REPR(v.delay_samples), PY_REPR(v.initial_slope), PY_REPR(v.coupling));
            });

        RW_VAR(range_millivolts);
        RW_VAR(level_ratios);
        RW_VAR(delay_samples);
        RW_VAR(initial_slope);
        RW_VAR(coupling);
    }

    {
        using C = vortex::alazar::input_t;
        CLS_VAL(Input);

        c.def(py::init<vortex::alazar::channel_t, size_t, size_t, vortex::alazar::coupling_t>(),
            "channel"_a = vortex::alazar::channel_t::B, "range_millivolts"_a = 400, "impedance_ohms"_a = 50, "coupling"_a = vortex::alazar::coupling_t::DC, doc(c, "__init__"));
        c.def("__repr__", [](const C& v) {
            return fmt::format("Input(channel={}, range_millivolts={}, impedance_ohms={}, coupling={}, bytes_per_samples={})",
                PY_REPR(v.channel), PY_REPR(v.range_millivolts), PY_REPR(v.impedance_ohms), PY_REPR(v.coupling), PY_REPR(v.bytes_per_sample()));
            });

        RW_VAR(channel);
        RW_VAR(range_millivolts);
        RW_VAR(impedance_ohms);
        RW_VAR(coupling);

        RO_ACC(bytes_per_sample);

        SHALLOW_COPY();
    }

    {
        using C = vortex::alazar::option::auxio_trigger_out_t;
        CLS_VAL(AuxIOTriggerOut);

        c.def(py::init(), doc(c, "__init__"));
        c.def(py::self == py::self);
        c.def("__repr__", [](const C& v) {return "AuxIOTriggerOut"; });

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::option::auxio_clock_out_t;
        CLS_VAL(AuxIOClockOut);

        c.def(py::init(), doc(c, "__init__"));
        c.def(py::self == py::self);
        c.def("__repr__", [](const C& v) {return "AuxIOClockOut"; });

        SHALLOW_COPY();
    }

    {
        using C = vortex::alazar::option::auxio_pacer_out_t;
        CLS_VAL(AuxIOPacerOut);

        c.def(py::init<U32>(), "divider"_a = 2, doc(c, "__init__"));
        c.def(py::self == py::self);
        c.def("__repr__", [](const C& v) {return fmt::format("AuxIOPacerOut(divider={})", v.divider); });

        RW_VAR(divider);

        SHALLOW_COPY();
    }
    {
        using C = vortex::alazar::option::oct_ignore_bad_clock_t;
        CLS_VAL(OCTIgnoreBadClock);

        c.def(py::init<double, double>(), "good_seconds"_a = 4.95e-6, "bad_seconds"_a = 4.95e-6, doc(c, "__init__"));
        c.def(py::self == py::self);
        c.def("__repr__", [](const C& v) {
            return fmt::format("OCTIgnoreBadClock(good_seconds={}, bad_seconds={})",
                v.good_seconds, v.bad_seconds);
            });
        RW_VAR(good_seconds);
        RW_VAR(bad_seconds);

        SHALLOW_COPY();
    }
}

static void _bind_acquire(py::module& m) {

    auto alazar = m.def_submodule("alazar");
    _bind_driver(alazar);
    _bind_acquire_config(alazar);

    {
        using T = vortex::alazar_acquisition_t;
        {
            using C = T::config_t::device_t;
            CLS_VAL(AlazarDevice);

            c.def(py::init<U32, U32>(), "system_index"_a = 1, "board_index"_a = 1);
            c.def("__repr__", [](const C& v) {
                return fmt::format("AlazarDevice(system_index={}, board_index={})",
                    PY_REPR(v.system_index), PY_REPR(v.board_index));
                });

            RW_VAR(system_index);
            RW_VAR(board_index);

            SHALLOW_COPY();
        }

        {
            using C = T::config_t;
            CLS_VAL(AlazarConfig);
            c.def("__repr__", [](const C& v) {return fmt::format("AlazarConfig{}\n)", alazar_inner_repr(v)); });

            setup_acquire_config(c);
            setup_alazar_config(c);
        }

        {
            using C = T;
            CLS_PTR(AlazarAcquisition);

            setup_acquisition<vortex::cpu_view_t<typename C::output_element_t>>(c);

            RO_ACC(board);
        }
    }

    {
        using T = vortex::alazar_fft_acquisition_t;
        {
            using C = T::config_t;
            CLS_VAL(AlazarFFTConfig);
            c.def("__repr__", [](const C& v) {return fmt::format("AlazarFFTConfig{}\n)", alazar_inner_repr(v)); });

            setup_acquire_config(c);
            setup_alazar_config(c);

            RW_VAR(fft_length);
            RW_VAR_XT(spectral_filter);
            RW_VAR_XT(background);
            RW_VAR(include_time_domain);

            RO_ACC(samples_per_ascan);
            RW_ACC(ascans_per_block);

            SHALLOW_COPY();
        }

        {
            using C = T;
            CLS_PTR(AlazarFFTAcquisition);

            setup_acquisition<vortex::cpu_view_t<typename C::output_element_t>>(c);

            RO_ACC(board);
        }
    }

#if defined(VORTEX_ENABLE_ALAZAR_GPU)
    {
        using T = vortex::alazar_gpu_acquisition_t;
        {
            using C = T::config_t;
            CLS_VAL(AlazarGPUConfig);
            c.def("__repr__", [](const C& v) {
                return fmt::format("AlazarGPUConfig{}\n    gpu_device = {},\n)",
                    alazar_inner_repr(v), v.gpu_device_index);
                });

            setup_acquire_config(c);
            setup_alazar_config(c);

            RW_VAR(gpu_device_index);

            SHALLOW_COPY();
        }

        {
            using C = T;
            CLS_PTR(AlazarGPUAcquisition);

            setup_acquisition<vortex::alazar::alazar_view_t<typename C::output_element_t>>(c);

            RO_ACC(board);
        }
    }
#endif

}

#if defined(VORTEX_ENABLE_ALAZAR_DAC)
static void _bind_io_config(py::module& m) {
    {
        using C = vortex::alazar::analog_channel_t;
        CLS_VAL(AnalogChannel);

        c.def(py::init([](double logical_units_per_physical_unit, size_t stream, size_t channel, double park) {
            C o;
            o.stream = stream;
            o.channel = channel;
            o.park = park;
            o.logical_units_per_physical_unit = logical_units_per_physical_unit;
            return o;
        }), "logical_units_per_physical_unit"_a = 1.0, "stream"_a = 0, "channel"_a = 0, "park"_a = 0);

        RW_VAR(stream);
        RW_VAR(channel);

        RW_VAR(park);

        RW_VAR(logical_units_per_physical_unit);
        RW_VAR(limits);
    }
}
#endif

static void _bind_io(py::module& m) {
#if defined(VORTEX_ENABLE_ALAZAR_DAC)

    auto alazar = m.def_submodule("alazar");
    _bind_io_config(alazar);

    using T = vortex::alazar_io_t;
    {
        using C = T::config_t;
        CLS_VAL(AlazarIOConfig);

        c.def(py::init(), doc(c, "__init__"));

        RW_VAR(analog_output_channels);

        RW_ACC(samples_per_block);
        RW_VAR(blocks_to_buffer);
        RW_VAR(divisor);

        RW_VAR(stop_on_error);

        FXN(validate);

        SHALLOW_COPY();
    }

    {
        using C = T;
        CLS_PTR(AlazarIO);

        c.def(py::init<>(), doc(c, "__init__"));
        c.def(py::init<std::shared_ptr<spdlog::logger>>(), doc(c, "__init__"));

        setup_io(c);

        FXN_GIL(stop, "force"_a = false);
    }

#endif
}

static void _bind_engine(py::module& m) {
#if defined(VORTEX_ENABLE_ENGINE)
    m.def("_bind", [](std::shared_ptr<vortex::alazar_acquisition_t>& a) {
        return vortex::engine::bind::acquisition<block_t>(a);
    });
    m.def("_bind", [](std::shared_ptr<vortex::alazar_fft_acquisition_t>& a) {
        return vortex::engine::bind::acquisition<block_t>(a);
    });
#  if defined(VORTEX_ENABLE_ALAZAR_GPU)
    m.def("_bind", [](std::shared_ptr<vortex::alazar_gpu_acquisition_t>& a) {
        return vortex::engine::bind::acquisition<block_t>(a);
    });
#  endif
#  if defined(VORTEX_ENABLE_ALAZAR_DAC)
    m.def("_bind", [](std::shared_ptr<vortex::alazar_io_t>& a) {
        return vortex::engine::bind::io<block_t>(a);
    });
#   endif

    m.def("acquire_alazar_clock", [](const vortex::engine::source_t<double>& source, const vortex::alazar_acquisition_t::config_t& acquire_config, vortex::alazar::channel_t clock_channel, std::shared_ptr<spdlog::logger> log) {
        return vortex::engine::acquire_alazar_clock<vortex::alazar_acquisition_t>(source, acquire_config, clock_channel, log);
    }, "source"_a, "acquire_config"_a, "clock_channel"_a, "log"_a = nullptr, py::call_guard<py::gil_scoped_release>());

#endif
}

#if defined(VORTEX_ENABLE_CUDA) && defined(VORTEX_ENABLE_ALAZAR_GPU)

template<typename T>
static void bind_alazar_device_tensor(py::module& m, py::object& cupy) {
    bind_cuda_tensor< vortex::alazar::alazar_device_tensor_t<T>>(m, "AlazarDeviceTensor", cupy);
}

#endif

static void _bind_memory(py::module& m) {
#if defined(VORTEX_ENABLE_CUDA) && defined(VORTEX_ENABLE_ALAZAR_GPU)
    auto cupy = try_import_cupy();

    bind_alazar_device_tensor<int8_t>(m, cupy);
    bind_alazar_device_tensor<uint16_t>(m, cupy);
    bind_alazar_device_tensor<uint64_t>(m, cupy);
    bind_alazar_device_tensor<float>(m, cupy);
    bind_alazar_device_tensor<double>(m, cupy);
#endif
}

VORTEX_MODULE(alazar) {
    VORTEX_BIND(acquire);
    VORTEX_BIND(io);
    VORTEX_BIND(engine);
    VORTEX_BIND(memory);
}
