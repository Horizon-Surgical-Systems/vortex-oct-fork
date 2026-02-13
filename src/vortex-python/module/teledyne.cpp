#include <vortex/acquire.hpp>

#include <vortex-python/module/helper.hpp>
#include <vortex-python/bind/acquire.hpp>
#include <vortex-python/bind/engine.hpp>
#include <vortex-python/bind/exception.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<vortex::teledyne::input_t>);

static void _bind_driver(py::module& m) {

    using namespace py::literals;

    {
        using C = vortex::teledyne::board_t;
        CLS_PTR(Teledyne);

        c.def(py::init<unsigned int>(), py::arg("board_index") = 0);

        FXN(start_capture);
        FXN(stop_capture);

        RO_ACC(info);
        RO_ACC(valid);
        RO_ACC(running);

        c.def_property_readonly("handle", [](C& o) { return reinterpret_cast<uintptr_t>(o.handle()); });
    }

    {
        using C = vortex::teledyne::board_t::info_t;
        CLS_VAL(Info);

        SHALLOW_COPY();
    }

    py::enum_<vortex::teledyne::trigger_source_t>(m, "TriggerSource")
        .value("PortTrig", vortex::teledyne::trigger_source_t::port_trig)
        .value("PortSync", vortex::teledyne::trigger_source_t::port_sync)
        .value("PortGPIO", vortex::teledyne::trigger_source_t::port_gpio)
        .value("Periodic", vortex::teledyne::trigger_source_t::periodic);

    py::enum_<vortex::teledyne::clock_generator_t>(m, "ClockGenerator")
        .value("InternalPLL", vortex::teledyne::clock_generator_t::internal_pll)
        .value("ExternalClock", vortex::teledyne::clock_generator_t::external_clock);

    py::enum_<vortex::teledyne::clock_reference_source_t>(m, "ClockReferenceSource")
        .value("Internal", vortex::teledyne::clock_reference_source_t::internal)
        .value("Port_CLK", vortex::teledyne::clock_reference_source_t::port_clk)
        .value("PXIE_10M", vortex::teledyne::clock_reference_source_t::PXIE_10M);

    py::enum_<vortex::teledyne::clock_edges_t>(m, "ClockEdges")
        .value("Rising", vortex::teledyne::clock_edges_t::rising)
        .value("Falling", vortex::teledyne::clock_edges_t::falling)
        .value("Both", vortex::teledyne::clock_edges_t::both);

    py::enum_<vortex::teledyne::fft_mode_t>(m, "FFTMode")
        .value("Disabled", vortex::teledyne::fft_mode_t::disabled)
        .value("Complex", vortex::teledyne::fft_mode_t::complex)
        .value("Magnitude", vortex::teledyne::fft_mode_t::magnitude)
        .value("LogMagnitude", vortex::teledyne::fft_mode_t::log_magnitude);

    py::enum_<ADQProductID_Enum>(m, "ADQProductID")
        .value("ADQ214", ADQProductID_Enum::PID_ADQ214)
        .value("ADQ114", ADQProductID_Enum::PID_ADQ214)
        .value("ADQ112", ADQProductID_Enum::PID_ADQ112)
        .value("SphinxHS", ADQProductID_Enum::PID_SphinxHS)
        .value("SphinxLS", ADQProductID_Enum::PID_SphinxLS)
        .value("ADQ108", ADQProductID_Enum::PID_ADQ108)
        .value("ADQDSP", ADQProductID_Enum::PID_ADQDSP)
        .value("SphinxAA14", ADQProductID_Enum::PID_SphinxAA14)
        .value("SphinxAA16", ADQProductID_Enum::PID_SphinxAA16)
        .value("ADQ412", ADQProductID_Enum::PID_ADQ412)
        .value("ADQ212", ADQProductID_Enum::PID_ADQ212)
        .value("SphinxAA_LS2", ADQProductID_Enum::PID_SphinxAA_LS2)
        .value("SphinxHS_LS2", ADQProductID_Enum::PID_SphinxHS_LS2)
        .value("SDR14", ADQProductID_Enum::PID_SDR14)
        .value("ADQ1600", ADQProductID_Enum::PID_ADQ1600)
        .value("SphinxXT", ADQProductID_Enum::PID_SphinxXT)
        .value("ADQ208", ADQProductID_Enum::PID_ADQ208)
        .value("DSU", ADQProductID_Enum::PID_DSU)
        .value("ADQ14", ADQProductID_Enum::PID_ADQ14)
        //.value("SDR14RF", ADQProductID_Enum::PID_SDR14RF)
        .value("EV12AS350_EVM", ADQProductID_Enum::PID_EV12AS350_EVM)
        .value("ADQ7", ADQProductID_Enum::PID_ADQ7)
        .value("ADQ8", ADQProductID_Enum::PID_ADQ8)
        .value("ADQ12", ADQProductID_Enum::PID_ADQ12)
        .value("ADQ32", ADQProductID_Enum::PID_ADQ32)
        .value("ADQSM", ADQProductID_Enum::PID_ADQSM)
        //.value("TX320", ADQProductID_Enum::PID_TX320)
        //.value("RX320", ADQProductID_Enum::PID_RX320)
        //.value("S6000", ADQProductID_Enum::PID_S6000)
        ;

    {
        using C = vortex::teledyne::device_list_entry_t;
        CLS_VAL(DeviceInfo);

        RO_VAR(ProductID);

        c.def("__repr__", [](const C& info) {
            return "<ProductID={}>"_s.format(info.ProductID);
            });

        SHALLOW_COPY();
    }

    m.def("enumerate", vortex::teledyne::enumerate);
}

static void _bind_config(py::module& m) {

    py::bind_vector<std::vector<vortex::teledyne::input_t>>(m, "VectorInput")
        .def("__repr__", [](const std::vector<vortex::teledyne::input_t>& v) {
            return list_repr(v, "VectorInput");
        });

    {
        using C = vortex::teledyne::clock_t;
        CLS_VAL(Clock);

        c.def(py::init<size_t, size_t, vortex::teledyne::clock_generator_t, vortex::teledyne::clock_reference_source_t, size_t, bool>(),
            "sampling_frequency"_a = 2'500'000'000,
            "reference_frequency"_a = 10'000'000,
            "clock_generator"_a = vortex::teledyne::clock_generator_t::internal_pll,
            "reference_source"_a = vortex::teledyne::clock_reference_source_t::internal,
            "delay_adjustment"_a = 0,
            "low_jitter_mode_enabled"_a = true,
            doc(c, "__init__"));
        c.def("__repr__", [](const C& v) {
            return fmt::format("Clock(sampling_frequency={}, reference_frequency={}, clock_generator={}, reference_source={}, delay_adjustment={}, low_jitter_mode_enabled={})",
                PY_REPR(v.sampling_frequency), PY_REPR(v.reference_frequency), PY_REPR(v.clock_generator), PY_REPR(v.reference_source), PY_REPR(v.delay_adjustment), PY_REPR(v.low_jitter_mode_enabled)); });

        RW_VAR(sampling_frequency);
        RW_VAR(reference_frequency);
        RW_VAR(clock_generator);
        RW_VAR(reference_source);
        RW_VAR(delay_adjustment);
        RW_VAR(low_jitter_mode_enabled);

        SHALLOW_COPY();
    }

    {
        using C = vortex::teledyne::input_t;
        CLS_VAL(Input);

        c.def(py::init<int>(),
            "channel"_a = 0, doc(c, "__init__"));
        c.def("__repr__", [](const C& v) {
            return fmt::format("Input(channel={})", PY_REPR(v.channel)); });

        RW_VAR(channel);

        SHALLOW_COPY();
    }
}

static void _bind_acquire(py::module& m) {

    auto teledyne = m.def_submodule("teledyne");
    _bind_driver(teledyne);
    _bind_config(teledyne);

    {
        using C = vortex::teledyne_config_t;
        CLS_VAL(TeledyneConfig);

        setup_acquire_config(c);

        RW_VAR(inputs);
        RW_VAR(acquire_timeout);

        RW_VAR(clock);

        RW_VAR(test_pattern_signal);

        RW_VAR(trigger_source);
        RW_VAR(trigger_skip_factor);
        RW_VAR(trigger_offset_samples);
        RW_VAR(trigger_sync_passthrough);
        RW_VAR(periodic_trigger_frequency);

        RW_VAR(sample_skip_factor);

        RW_VAR(enable_fwoct);
        RW_VAR(resampling_factor);
        RW_VAR(clock_delay_samples);
        RW_VAR(clock_edges);
        RW_VAR(background);
        RW_VAR(spectral_filter);
        RW_VAR(fft_mode);

        RW_VAR(enable_hugepages);

        RO_ACC(channels_per_sample);
        RO_ACC(channel_mask);

        RW_ACC(samples_per_second);

        FXN(validate);
    }

    {
        using C = vortex::teledyne_acquisition_t;
        CLS_PTR(TeledyneAcquisition);

        setup_acquisition<vortex::cpu_view_t<typename C::output_element_t>>(c);

        c.def_property_readonly("board_handle", [](C& o) -> uintptr_t {
            if(o.board()) {
                return reinterpret_cast<uintptr_t>(o.board()->handle());
            } else {
                return 0;
            }
        });
    }
}

static void _bind_engine(py::module& m) {
#if defined(VORTEX_ENABLE_ENGINE)
    m.def("_bind", [](std::shared_ptr<vortex::acquire::teledyne_acquisition_t>& a) {
        return vortex::engine::bind::acquisition<block_t>(a);
    });
#endif
}

VORTEX_MODULE(teledyne) {
    VORTEX_BIND(acquire);
    VORTEX_BIND(engine);
}
