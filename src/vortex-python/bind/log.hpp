#pragma once

#include <vortex-python/bind/common.hpp>

template<typename C>
static void setup_sink(py::class_<C, spdlog::sinks::sink, std::shared_ptr<C>>& c) {
    c.def_property("level", [](C& o) { return static_cast<int>(o.level()); }, [](C& o, int v) { o.set_level(static_cast<spdlog::level::level_enum>(v)); }, doc(c, "level"));
    FXN(set_pattern);

    FXN(flush)
}
