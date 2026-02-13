#include <vortex/acquire.hpp>

#include <vortex-python/bind/acquire.hpp>

static void bind_null(py::module& m) {
    {
        using C = vortex::null_acquisition_t<uint16_t>;
        CLS_PTR(NullAcquisition);

        c.def(py::init(), doc(m, "__init__"));

        FXN_GIL(initialize, "config"_a);

        FXN_GIL(prepare);
        FXN_GIL(start);
        FXN_GIL(stop);

        c.def("next", [](C& o, const vortex::cpu_view_t<C::output_element_t>& buffer, size_t id) {
            py::gil_scoped_release();
            return o.next(id, buffer);
        }, "buffer"_a, "id"_a = 0, doc(c, "next"));

        RO_ACC(config);
    }

    {
        using C = vortex::null_acquisition_t<uint16_t>::config_t;
        CLS_VAL(NullAcquisitionConfig);

        setup_acquire_config(c);

        RW_ACC(channels_per_sample);
    }
}

static void bind_file(py::module& m) {
    {
        using C = vortex::file_acquisition_t<uint16_t>;
        CLS_PTR(FileAcquisition);

        c.def(py::init<std::shared_ptr<spdlog::logger>>(), "logger"_a = nullptr, doc(c, "__init__"));

        FXN_GIL(initialize, "config"_a);

        FXN_GIL(prepare);
        FXN_GIL(start);
        FXN_GIL(stop);

        c.def("next", [](C& o, const vortex::cpu_view_t<C::output_element_t>& buffer, size_t id) {
            py::gil_scoped_release();
            return o.next(id, buffer);
        }, "buffer"_a, "id"_a = 0, doc(c, "next"));

        RO_ACC(config);
    }

    {
        using C = vortex::file_acquisition_t<uint16_t>::config_t;
        CLS_VAL(FileAcquisitionConfig);

        setup_acquire_config(c);

        RW_ACC(channels_per_sample);

        RW_VAR(path);
        RW_VAR(loop);
    }
}

void bind_acquire(py::module& root) {

    auto m = root.def_submodule("acquire");

    bind_null(m);
    bind_file(m);

}
