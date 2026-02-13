#pragma once

#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>
#include <vortex-python/bind/engine.hpp>

template<typename C>
static void setup_io(py::class_<C, std::shared_ptr<C>>& c) {
    RO_ACC(config);

    FXN_GIL(initialize);

    FXN_GIL(prepare);
    FXN_GIL(start);

#if defined(VORTEX_ENABLE_ENGINE)

    using streams_t = std::tuple<
        vortex::cpu_view_t<vortex::counter_t>,                      // counter
        vortex::cpu_view_t<typename block_t::analog_element_t>,     // galvo target
        vortex::cpu_view_t<typename block_t::analog_element_t>,     // sample target
        vortex::cpu_view_t<typename block_t::analog_element_t>,     // galvo actual
        vortex::cpu_view_t<typename block_t::analog_element_t>,     // sample actual
        vortex::cpu_view_t<typename block_t::digital_element_t>     // strobes
    >;

    c.def("next", [](C& o, size_t count, const streams_t& streams, size_t id) {

        py::gil_scoped_release gil;
        o.next(id, count, streams);

    }, "count"_a, "streams"_a, "id"_a = 0, doc(c, "next"));

    c.def("next_async", [](C& o, size_t count, const streams_t& streams, typename C::callback_t callback, size_t id) {

        py::gil_scoped_release gil;
        o.next_async(id, count, streams, std::move(callback));

    }, "count"_a, "streams"_a, "callback"_a, "id"_a = 0, doc(c, "next_async"));

#endif

    RO_ACC(running);
}
