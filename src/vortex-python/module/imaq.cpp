#include <vortex/acquire.hpp>

#include <vortex/driver/imaq.hpp>

#include <vortex-python/module/helper.hpp>
#include <vortex-python/bind/acquire.hpp>
#include <vortex-python/bind/engine.hpp>
#include <vortex-python/bind/exception.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<vortex::imaq::trigger_output_t>);

static void _bind_driver(py::module& m) {

    py::enum_<vortex::imaq::signal_t>(m, "Signal")
        .value("NoSignal", vortex::imaq::signal_t::none)
        .value("External", vortex::imaq::signal_t::external)
        .value("RTSI", vortex::imaq::signal_t::rtsi)
        .value("IsoIn", vortex::imaq::signal_t::iso_in)
        .value("IsoOut", vortex::imaq::signal_t::iso_out)
        .value("Status", vortex::imaq::signal_t::status)
        .value("ScaledEncoder", vortex::imaq::signal_t::scaled_encoder)
        .value("SoftwareTrigger", vortex::imaq::signal_t::software_trigger);

    py::enum_<vortex::imaq::polarity_t>(m, "Polarity")
        .value("Low", vortex::imaq::polarity_t::low)
        .value("High", vortex::imaq::polarity_t::high);

    py::enum_<vortex::imaq::source_t>(m, "Source")
        .value("Disabled", vortex::imaq::source_t::disabled)
        .value("AcquisitionInProgress", vortex::imaq::source_t::acquisition_in_progress)
        .value("AcquisitionDone", vortex::imaq::source_t::acquisition_done)
        .value("PixelClock", vortex::imaq::source_t::pixel_clock)
        .value("Unasserted", vortex::imaq::source_t::unasserted)
        .value("Asserted", vortex::imaq::source_t::asserted)
        .value("Hsync", vortex::imaq::source_t::hsync)
        .value("Vsync", vortex::imaq::source_t::vsync)
        .value("FrameStart", vortex::imaq::source_t::frame_start)
        .value("FrameDone", vortex::imaq::source_t::frame_done)
        .value("ScaledEncoder", vortex::imaq::source_t::scaled_encoder);

    {
        using C = vortex::imaq::imaq_t::roi_t;
        CLS_VAL(RegionOfInterest);

        c.def(py::init<uInt32, uInt32, uInt32, uInt32, uInt32>(),
            "top"_a, "left"_a, "height"_a, "width"_a, "pixels_per_row"_a = 0);

        RW_VAR(top);
        RW_VAR(left);
        RW_VAR(height);
        RW_VAR(width);
        RW_VAR(pixels_per_row);

        SHALLOW_COPY();
    }

    {
        using C = vortex::imaq::imaq_t;
        CLS_PTR(Imaq);

        c.def(py::init());
        c.def(py::init<std::string>());

        FXN(required_buffer_size);

        FXN(configure_region);
        FXN(query_region);

        FXN(configure_line_trigger);
        //FXN(configure_frame_trigger);
        FXN(configure_trigger_output);

        //FXN(start_capture);
        FXN(stop_capture);

        RO_ACC(info);
        RO_ACC(valid);
        RO_ACC(name);
        RO_ACC(running);
    }

    {
        using C = vortex::imaq::imaq_t::info_t;
        CLS_VAL(Info);

        RW_VAR(device);
        RW_VAR(serial);
        RW_VAR(calibration);
        RW_VAR(resolution);
        RW_VAR(acquisition_window);
        RW_VAR(line_scan);
        RW_VAR(bits_per_pixel);
        RW_VAR(bytes_per_pixel);

        SHALLOW_COPY();
    }

    {
        using C = vortex::imaq::imaq_t::info_t::resolution_t;
        CLS_VAL(Resolution);

        RW_VAR(horizontal);
        RW_VAR(vertical);

        SHALLOW_COPY();
    }

    m.def("enumerate", vortex::imaq::enumerate);

}

static void _bind_config(py::module& m) {
    {
        using C = vortex::imaq::line_trigger_t;
        CLS_VAL(LineTrigger);

        c.def(py::init<uInt32, uInt32, vortex::imaq::polarity_t, vortex::imaq::signal_t>(), "line"_a = 0, "skip"_a = 0, "polarity"_a = vortex::imaq::polarity_t::high, "signal"_a = vortex::imaq::signal_t::external);

        RW_VAR(line);
        RW_VAR(skip);
        RW_VAR(polarity);
        RW_VAR(signal);

        SHALLOW_COPY();

        c.def("__repr__", [](const C& v) { return fmt::format("LineTrigger(line={}, skip={}, polarity={}, signal={})", v.line, v.skip, PY_REPR(v.polarity), PY_REPR(v.signal)); });
    }
    {
        using C = vortex::imaq::frame_trigger_t;
        CLS_VAL(FrameTrigger);

        c.def(py::init<uInt32, vortex::imaq::polarity_t, vortex::imaq::signal_t>(), "line"_a = 0, "polarity"_a = vortex::imaq::polarity_t::high, "signal"_a = vortex::imaq::signal_t::external);

        RW_VAR(line);
        RW_VAR(polarity);
        RW_VAR(signal);

        SHALLOW_COPY();

        c.def("__repr__", [](const C& v) { return fmt::format("FrameTrigger(line={}, polarity={}, signal={})", v.line, PY_REPR(v.polarity), PY_REPR(v.signal)); });
    }
    {
        using C = vortex::imaq::trigger_output_t;
        CLS_VAL(TriggerOutput);

        c.def(py::init<uInt32, vortex::imaq::source_t, vortex::imaq::polarity_t, vortex::imaq::signal_t>(), "line"_a = 0, "source"_a = vortex::imaq::source_t::hsync, "polarity"_a = vortex::imaq::polarity_t::high, "signal"_a = vortex::imaq::signal_t::external);

        RW_VAR(line);
        RW_VAR(source);
        RW_VAR(polarity);
        RW_VAR(signal);

        SHALLOW_COPY();

        c.def("__repr__", [](const C& v) { return fmt::format("TriggerOutput(line={}, source={}, polarity={}, signal={})", v.line, PY_REPR(v.source), PY_REPR(v.polarity), PY_REPR(v.signal)); });
    }
    {
        using C = std::vector<vortex::imaq::trigger_output_t>;
        auto c = py::bind_vector<C>(m, "VectorTriggerOutput");

        c.def("__repr__", [](const C& v) { return list_repr(v, "VectorTriggerOutput"); });
    }
}

static void _bind_acquire(py::module& m) {

    auto imaq = m.def_submodule("imaq");
    _bind_driver(imaq);
    _bind_config(imaq);

    {
        using C = vortex::imaq_acquisition_t::config_t;
        CLS_VAL(ImaqAcquisitionConfig);

        setup_acquire_config(c);

        RW_VAR(device_name);

        RW_VAR(offset);
        RW_ACC(sample_offset);
        RW_ACC(record_offset);

        RW_VAR(line_trigger);
        RW_VAR(frame_trigger);
        RW_VAR(trigger_output);

        RW_VAR(ring_size);

        RW_VAR(acquire_timeout);
        RW_VAR(stop_on_error);
        RW_VAR(bypass_region_check);

        RO_ACC(channels_per_sample);

        FXN(validate);
    }
    {
        using C = vortex::imaq_acquisition_t;
        CLS_PTR(ImaqAcquisition);

        setup_acquisition<vortex::cpu_view_t<typename C::output_element_t>>(c);
    }
}

static void _bind_engine(py::module& m) {
#if defined(VORTEX_ENABLE_ENGINE)
    m.def("_bind", [](std::shared_ptr<vortex::imaq_acquisition_t>& a) {
        return vortex::engine::bind::acquisition<block_t>(a);
    });
#endif
}

VORTEX_MODULE(imaq) {
    VORTEX_BIND(acquire);
    VORTEX_BIND(engine);
}
