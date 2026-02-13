#pragma once

#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>

template<typename C>
static void setup_processor_config(py::class_<C>& c) {
    c.def(py::init(), doc(c, "__init__"));

    RW_ACC(records_per_block);
    RW_ACC(samples_per_record);
    RW_ACC(channels_per_sample);

    RW_ACC(ascans_per_block);
    RO_ACC(samples_per_ascan);

    RO_ACC(input_shape);
    RO_ACC(output_shape);

    FXN(validate);

    SHALLOW_COPY();
}

template<typename C>
static void setup_processor(py::class_<C, std::shared_ptr<C>>& c) {
    c.def(py::init<std::shared_ptr<spdlog::logger>>(), "logger"_a = nullptr, doc(c, "__init__"));

    RO_ACC(config);

    FXN_GIL(initialize, "config"_a);
    FXN_GIL(change, "config"_a);
}

template<typename input_view_t, typename output_view_t, typename C>
static void setup_oct_processor(py::class_<C, std::shared_ptr<C>>& c) {
    setup_processor(c);

    c.def("next", [](C& o, const input_view_t& input_buffer, const output_view_t& output_buffer, size_t id, bool append_history) {

            py::gil_scoped_release gil;
            o.next(id, input_buffer, output_buffer, append_history);

    }, "input_buffer"_a, "output_buffer"_a, "id"_a = 0, "append_history"_a = true, doc(c, "next"));

    c.def("next_async", [](C& o, const input_view_t& input_buffer, const output_view_t& output_buffer, typename C::callback_t callback, size_t id, bool append_history) {

            py::gil_scoped_release gil;
            o.next_async(id, input_buffer, output_buffer, append_history, std::move(callback));

    }, "input_buffer"_a, "output_buffer"_a, "callback"_a, "id"_a = 0, "append_history"_a = true, doc(c, "next_async"));
}
