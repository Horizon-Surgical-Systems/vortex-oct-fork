#include <vortex/scan.hpp>

#include <vortex/simple/engine.hpp>

#include <vortex-python/module/helper.hpp>

static void _bind_simple(py::module& m) {
    {
        using C = vortex::simple::simple_engine_config_t;
        CLS_VAL(SimpleEngineConfig);

        c.def(py::init());

        RW_VAR(input_channel);
        RW_VAR(clock_channel);
        RW_VAR(internal_clock);

        RW_VAR(broct_save_path);
        RW_VAR(sample_target_save_path);
        RW_VAR(sample_actual_save_path);

        RW_VAR(swept_source);

        RW_VAR(galvo_delay);

        RW_VAR(dispersion);

        RW_VAR(samples_per_ascan);
        RW_VAR(ascans_per_bscan);
        RW_VAR(bscans_per_volume);

        RW_VAR(blocks_to_acquire);
        RW_VAR(blocks_to_allocate);
        RW_VAR(ascans_per_block);

        RW_VAR(preload_count);
        RW_VAR(process_slots);

        SHALLOW_COPY();
    }
    {
        using C = vortex::simple::simple_engine_t;
        CLS_VAL(SimpleEngine);

        c.def(py::init());
        c.def(py::init<std::string>());

        FXN_GIL(initialize)

        c.def("append_scan", [](C& o, std::shared_ptr<vortex::raster_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("append_scan", [](C& o, std::shared_ptr<vortex::repeated_raster_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("append_scan", [](C& o, std::shared_ptr<vortex::radial_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("append_scan", [](C& o, std::shared_ptr<vortex::repeated_radial_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("append_scan", [](C& o, std::shared_ptr<vortex::spiral_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("append_scan", [](C& o, std::shared_ptr<vortex::freeform_scan_t>& s) { o.append_scan(s); }, py::call_guard<py::gil_scoped_release>());

        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::raster_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::repeated_raster_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::radial_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::repeated_radial_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::spiral_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());
        c.def("interrupt_scan", [](C& o, std::shared_ptr<vortex::freeform_scan_t>& s) { o.interrupt_scan(s); }, py::call_guard<py::gil_scoped_release>());

        c.def("start", [](C& o) { o.start(); });
        c.def("start", [](C& o, C::callback_t&& cb) { o.start(std::forward<C::callback_t>(cb)); });

        FXN_GIL(wait);
        c.def("wait_for", [](C& o, double dt) { return o.wait_for(std::chrono::duration_cast<std::chrono::milliseconds>(vortex::seconds(dt))); }, py::call_guard<py::gil_scoped_release>());

        FXN_GIL(stop);

        c.def("volume", [](const C& o) { return o.volume(); });
    }

}

VORTEX_MODULE(simple) {
    auto m = root.def_submodule("simple");
    _bind_simple(m);
}
