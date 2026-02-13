#pragma once

#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>

template<typename C>
static void setup_acquire_config(py::class_<C>& c) {
    c.def(py::init(), doc(c, "__init__"));

    RO_ACC(shape);

    RW_ACC(samples_per_record);
    RW_ACC(records_per_block);

    FXN(validate);

    SHALLOW_COPY();
}

template<typename view_t, typename C>
static void setup_acquisition(py::class_<C, std::shared_ptr<C>>& c) {
    c.def(py::init<std::shared_ptr<spdlog::logger>>(), "logger"_a = nullptr, doc(c, "__init__"));

    RO_ACC(config);

    FXN_GIL(initialize, "config"_a);

    FXN_GIL(prepare);
    FXN_GIL(start);
    FXN_GIL(stop);
    RO_ACC(running);

    c.def("next", [](C& o, const view_t& buffer, size_t id) {

        py::gil_scoped_release gil;
        return o.next(id, buffer);

    }, "buffer"_a, "id"_a = 0, doc(c, "next"));

    c.def("next_async", [](C& o, const view_t& buffer, typename C::callback_t callback, size_t id) {

        py::gil_scoped_release gil;
        o.next_async(id, buffer, std::move(callback));

    }, "buffer"_a, "callback"_a, "id"_a = 0, doc(c, "next_async"));

}
