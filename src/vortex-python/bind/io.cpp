#include <vortex-python/bind/common.hpp>

#include <vortex/io.hpp>

#if defined(VORTEX_ENABLE_ASIO)
#  include <vortex/driver/machdsp.hpp>
#endif

#include <vortex-python/module/helper.hpp>
#include <vortex-python/bind/io.hpp>
#include <vortex-python/bind/engine.hpp>
#include <vortex-python/bind/memory.hpp>

static void bind_null(py::module& m) {
    using T = vortex::io::null_io_t<vortex::io::null_config_t>;

    {
        using C = T::config_t;
        CLS_VAL(NullConfig);

        c.def(py::init());

        FXN(validate);

        SHALLOW_COPY();
    }

    {
        using C = T;
        CLS_PTR(NullIO);

        c.def(py::init());

        RO_ACC(config);

        FXN_GIL(initialize);

        FXN_GIL(prepare);
        FXN_GIL(start);
        FXN_GIL(stop);
    }
}

#if defined(VORTEX_ENABLE_ASIO)

static void _bind_machdsp_driver(py::module& m) {
    {
        using C = vortex::machdsp::machdsp_t;
        CLS_VAL(MachDSP);

        c.def(py::init<const std::string&, size_t>(), "port"_a, "baud_rate"_a);

        FXN(ping, "index"_a = 0);

        FXN(stream_start);
        FXN(stream_stop);
        FXN(stream_reset);
        FXN(stream_status);

        FXN(write_setting, "setting"_a, "value0"_a, "value1"_a = 0, "value2"_a = 0);
        FXN(read_settings);
        FXN(reset_settings);

        FXN(set_trigger_edge_rising, "rising"_a = true);
        FXN(set_internal_trigger, "divisor"_a);
        FXN(set_sample_divisor, "divisor"_a);
        FXN(set_aux_divisor, "divisor"_a);
        FXN(set_blocks, "block_count"_a, "block_samples"_a);
        FXN(set_enable_receive, "enable"_a);
        FXN(set_overflow_behavior, "behavior"_a);
        FXN(set_underflow_behavior, "behavior"_a);

        FXN(check_error);
        FXN(clear_error);

        FXN(hard_reset);

        RW_MUT(timeout);

        RO_ACC(valid);
        RO_ACC(info);
    }

    {
        using C = vortex::machdsp::stream_status_t;
        CLS_VAL(StreamStatus);

        RW_VAR(tx_block_idx);
        RW_VAR(rx_block_idx);
        RW_VAR(tx_block_valid);
        RW_VAR(rx_block_valid);
        RW_VAR(state);
    }
    {
        using C = vortex::machdsp::stream_state_t;
        auto c = py::enum_<C>(m, "StreamState")
            .value("StreamContinue", C::STREAM_CONTINUE)
            .value("StreamExit", C::STREAM_EXIT)
            .value("StreamAbort", C::STREAM_ABORT);
        c.export_values();
    }

    {
        using C = vortex::machdsp::settings_t;
        CLS_VAL(Settings);

        RW_VAR(underflow_behavior);
        RW_VAR(overflow_behavior);
        RW_VAR(block_count);
        RW_VAR(block_samples);
        RW_VAR(max_buffer_samples);
        RW_VAR(enable_receive);
        RW_VAR(trigger_edge);
        RW_VAR(internal_trigger);
        RW_VAR(sample_divisor);
        RW_VAR(max_sample_divisor);
        RW_VAR(aux_divisor);
        RW_VAR(max_aux_divisor);
    }
    {
        using C = vortex::machdsp::underflow_t;
        auto c = py::enum_<C>(m, "Underflow")
            .value("Error", C::UNDERFLOW_ERROR)
            .value("Loop", C::UNDERFLOW_LOOP)
            .value("Hold", C::UNDERFLOW_HOLD)
            .value("Ignore", C::UNDERFLOW_IGNORE);
        c.export_values();
    }
    {
        using C = vortex::machdsp::overflow_t;
        auto c = py::enum_<C>(m, "Overflow")
            .value("Error", C::OVERFLOW_ERROR)
            .value("Ignore", C::OVERFLOW_IGNORE);
        c.export_values();
    }
    {
        using C = vortex::machdsp::settings_entry_t;
        auto c = py::enum_<C>(m, "SettingsEntry")
            .value("Underflow", C::SETTINGS_UNDERFLOW)
            .value("Overflow", C::SETTINGS_OVERFLOW)
            .value("Blocks", C::SETTINGS_BLOCKS)
            .value("EnableReceive", C::SETTINGS_ENABLE_RECEIVE)
            .value("TriggerEdge", C::SETTINGS_TRIGGER_EDGE)
            .value("TriggerInternal", C::SETTINGS_TRIGGER_INTERNAL)
            .value("SampleDivisor", C::SETTINGS_SAMPLE_DIVISOR)
            .value("AuxDivisor", C::SETTINGS_AUX_DIVISOR);
        c.export_values();
    }

    {
        using C = vortex::machdsp::info_t;
        CLS_VAL(Info);

        RW_VAR(version);
        RW_VAR(zero_point);
        RW_VAR(per_lsb);
        RW_VAR(max_buffer_samples);
        RW_VAR(max_buffer_samples);
        RW_VAR(max_sample_divisor);
        RW_VAR(max_aux_divisor);
    }
    {
        using C = vortex::machdsp::info_t::version_t;
        CLS_VAL(Version);

        RW_VAR(major);
        RW_VAR(minor);
        RW_VAR(features);
    }
}

static void _bind_machdsp_config(py::module& m) {
    {
        using C = vortex::machdsp::channel_t;
        CLS_VAL(Channel);

        c.def(py::init([](double logical_units_per_physical_unit, size_t stream, size_t channel, double degree_scale) {
            C o;
            o.stream = stream;
            o.channel = channel;
            o.degree_scale = degree_scale;
            o.logical_units_per_physical_unit = logical_units_per_physical_unit;
            return o;
            }), "logical_units_per_physical_unit"_a = 1.0, "stream"_a = 0, "channel"_a = 0, "degree_scale"_a = 15);

        RW_VAR(stream);
        RW_VAR(channel);

        RW_VAR(logical_units_per_physical_unit);
        RW_VAR(degree_scale);
    }
}

static void bind_machdsp(py::module& m) {

    auto machdsp = m.def_submodule("machdsp");
    _bind_machdsp_driver(machdsp);
    _bind_machdsp_config(machdsp);

    using T = vortex::machdsp_io_t;
    {
        using C = T::config_t;
        CLS_VAL(MachDSPIOConfig);

        c.def(py::init());

        RW_VAR(port);
        RW_VAR(baud_rate);
        
        RW_VAR(output_channels);
        RW_VAR(input_channels);

        RW_ACC(samples_per_block);
        RW_VAR(blocks_to_buffer);

        RW_VAR(readwrite_timeout);

        RW_VAR(sample_divisor);
        RW_VAR(aux_divisor);
        RW_VAR(trigger_rising_edge);

        RW_VAR(stop_on_error);

        FXN(validate);

        SHALLOW_COPY();
    }

    {
        using C = T;
        CLS_PTR(MachDSPIO);

        c.def(py::init());
        c.def(py::init<std::shared_ptr<spdlog::logger>>());

        setup_io(c);

        FXN_GIL(stop);
    }
}

#endif

void bind_io(py::module& root) {
    auto m = root.def_submodule("io");

    bind_null(m);
#if defined(VORTEX_ENABLE_ASIO)
    bind_machdsp(m);
#endif
}
