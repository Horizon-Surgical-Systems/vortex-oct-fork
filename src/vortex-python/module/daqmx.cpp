#include <vortex/io.hpp>

#include <vortex/driver/daqmx.hpp>

#include <vortex-python/module/helper.hpp>
#include <vortex-python/bind/io.hpp>
#include <vortex-python/bind/engine.hpp>
#include <vortex-python/bind/memory.hpp>

static void _bind_driver(py::module& m){

    {
        using C = vortex::daqmx::edge_t;
        auto c = py::enum_<C>(m, "Edge")
            .value("Rising", C::rising)
            .value("Falling", C::falling);
        c.export_values();
    }
    {
        using C = vortex::daqmx::terminal_t;
        auto c = py::enum_<C>(m, "Terminal")
            .value("Referenced", C::referenced)
            .value("Unreferenced", C::unreferenced)
            .value("Differential", C::differential)
            .value("Pseudodifferential", C::pseudodifferential);
        c.export_values();
    }
    {
        using C = vortex::daqmx::sample_mode_t;
        auto c = py::enum_<C>(m, "SampleMode")
            .value("Finite", C::finite)
            .value("Continuous", C::continuous)
            .value("Hardware", C::hardware);
        c.export_values();
    }

    {
        using C = vortex::daqmx::daqmx_t;
        CLS_PTR(DAQmx);

        c.def(py::init<std::string>(), "task_name"_a, doc(c, "__init__"));

        FXN(create_digital_output, "line_name"_a);
        FXN(create_digital_output, "line_name"_a);
        FXN(create_analog_voltage_output, "port_name"_a, "min"_a, "max"_a);
        FXN(create_analog_voltage_input, "port_name"_a, "min"_a, "max"_a, "terminal"_a = vortex::daqmx::terminal_t::referenced);

        FXN(configure_sample_clock, "source"_a, "sample_mode"_a, "samples_per_second"_a, "samples_per_channel"_a, "divisor"_a = 1, "edge"_a = vortex::daqmx::edge_t::rising);

        FXN(set_output_buffer_size, "samples_per_channel"_a);
        FXN(set_input_buffer_size, "samples_per_channel"_a);

        FXN(set_regeneration, "enable"_a);

        c.def("write_analog", [](C& o, size_t samples_per_channel, const vortex::cpu_view_t<float64>& buffer, const vortex::seconds& timeout) {
            o.write_analog(samples_per_channel, buffer.to_xt(), timeout);
        }, "samples_per_channel"_a, "buffer"_a, "timeout"_a, doc(c, "write_analog"));
        c.def("write_digital", [](C& o, size_t samples_per_channel, const vortex::cpu_view_t<uint32_t>& buffer, const vortex::seconds& timeout) {
            o.write_digital(samples_per_channel, buffer.to_xt(), timeout);
        }, "samples_per_channel"_a, "buffer"_a, "timeout"_a, doc(c, "write_digital"));
        // c.def("read_analog", [](C& o, size_t samples_per_channel, vortex::cpu_view_t<float64>& buffer, const vortex::seconds& timeout) {
        //     o.read_analog(samples_per_channel, buffer.to_xt(), timeout);
        // }, "samples_per_channel"_a, "buffer"_a, "timeout"_a, doc(c, "read_analog"));
        // c.def("read_digital", [](C& o, size_t samples_per_channel, vortex::cpu_view_t<uInt32>& buffer, const vortex::seconds& timeout) {
        //     o.read_digital(samples_per_channel, buffer.to_xt(), timeout);
        // }, "samples_per_channel"_a, "buffer"_a, "timeout"_a, doc(c, "read_digital"));

        FXN(start_task);
        FXN(stop_task);
        FXN(clear_task);

        RO_ACC(valid);
        RO_ACC(name);
        RO_ACC(running);
    }
}

PYBIND11_MAKE_OPAQUE(std::vector<vortex::daqmx_io_t::config_t::channel_t>);

// namespace pybind11 {
//     namespace detail {
//         template <typename... Ts>
//         struct type_caster<vortex::io::channel_t<Ts...>> :
//             variant_caster<vortex::io::channel_t<Ts...>> {};
//     }
// }

template<typename C>
static void setup_channel(py::class_<C>& c) {
    c.def(py::init());

    RW_VAR(stream);
    RW_VAR(channel);

    SHALLOW_COPY();
}


template<typename C>
static void setup_digital_channel(py::class_<C>& c) {
    c.def(py::init([](std::string line_name, size_t stream, size_t channel) {
        C o;
        o.stream = stream;
        o.channel = channel;
        o.line_name = std::move(line_name);
        return o;
    }), "line_name"_a, "stream"_a = 0, "channel"_a = 0);

    setup_channel(c);

    RW_VAR(line_name);
    SRO_VAR(max_bits);
}

template<typename C>
static void setup_analog_channel(py::class_<C>& c) {
    c.def(py::init([](std::string port_name, double logical_units_per_physical_unit, size_t stream, size_t channel) {
        C o;
        o.stream = stream;
        o.channel = channel;
        o.port_name = std::move(port_name);
        o.logical_units_per_physical_unit = logical_units_per_physical_unit;
        return o;
    }), "port_name"_a, "logical_units_per_physical_unit"_a = 1.0, "stream"_a = 0, "channel"_a = 0);

    setup_channel(c);

    RW_VAR(port_name);

    RW_VAR(logical_units_per_physical_unit);
    RW_VAR(limits);
}

static void _bind_config(py::module& m) {

    py::bind_vector<std::vector<vortex::daqmx_io_t::config_t::channel_t>>(m, "VectorChannel")
    .def("__repr__", [](const std::vector<vortex::daqmx_io_t::config_t::channel_t>& v){
        return list_repr(v, "VectorChannel");
    });

    {
        using C = vortex::daqmx::channel::digital_output_t;
        CLS_VAL(DigitalOutput);

        setup_digital_channel(c);
    }
    {
        using C = vortex::daqmx::channel::digital_input_t;
        CLS_VAL(DigitalInput);

        setup_digital_channel(c);
    }
    {
        using C = vortex::daqmx::channel::analog_voltage_output_t;
        CLS_VAL(AnalogVoltageOutput);

        setup_analog_channel(c);
    }
    {
        using C = vortex::daqmx::channel::analog_voltage_input_t;
        CLS_VAL(AnalogVoltageInput);

        setup_analog_channel(c);

        RW_VAR(terminal);
    }

}

static void _bind_io(py::module& m) {

    auto daqmx = m.def_submodule("daqmx");
    _bind_driver(daqmx);
    _bind_config(daqmx);

    using T = vortex::daqmx_io_t;
    {
        using C = decltype(T::config_t::clock);
        CLS_VAL(DAQmxConfigClock);

        c.def(py::init());

        RW_VAR(source);
        RW_VAR(edge);
        RW_VAR(divisor);

        SHALLOW_COPY();
    }

    {
        using C = T::config_t;
        CLS_VAL(DAQmxConfig);

        c.def(py::init());

        RW_VAR(name);

        RW_ACC(samples_per_second);
        RW_ACC(samples_per_block);

        RW_VAR(clock);

        RW_VAR(blocks_to_buffer);

        RW_VAR(readwrite_timeout);

        RW_VAR(channels);

        RW_VAR(persistent_task);
        RW_VAR(stop_on_error);

        FXN(validate);

        SHALLOW_COPY();
    }

    {
        using C = T;
        CLS_PTR(DAQmxIO);

        c.def(py::init());
        c.def(py::init<std::shared_ptr<spdlog::logger>>());

        setup_io(c);

        FXN_GIL(stop);
    }
}

static void _bind_engine(py::module& m) {
#if defined(VORTEX_ENABLE_ENGINE)
    m.def("_bind", [](std::shared_ptr<vortex::daqmx_io_t>& a) {
        return vortex::engine::bind::io<block_t>(a);
    });
#endif
}

VORTEX_MODULE(daqmx) {
    VORTEX_BIND(io);
    VORTEX_BIND(engine);
}
