#include <vortex-python/bind/marker.hpp>

void bind_marker(py::module& root) {
    auto m = root.def_submodule("marker");

    {
        using C = vortex::marker::scan_boundary;
        CLS_VAL(ScanBoundary);

        c.def(py::init());
        c.def(py::init<vortex::counter_t, vortex::counter_t, size_t>(),
            "sample"_a, "sequence"_a, "volume_count_hint"_a = 0);
        c.def(py::init<vortex::counter_t, vortex::counter_t, size_t, vortex::marker::scan_boundary::flags_t>(),
            "sample"_a, "sequence"_a, "volume_count_hint"_a, "flags"_a);
        c.def("__repr__", [](const C& v) {
            return fmt::format("ScanBoundary(sample={}, sequence={}, volume_count_hint={}, flags={})",
                PY_REPR(v.sample), PY_REPR(v.sequence), PY_REPR(v.volume_count_hint), PY_REPR(v.flags));
        });

        RW_VAR(sample);
        RW_VAR(sequence);
        RW_VAR(volume_count_hint);
        RW_VAR(flags);
    }

    {
        using C = vortex::marker::volume_boundary;
        CLS_VAL(VolumeBoundary);

        c.def(py::init());
        c.def(py::init<vortex::counter_t, vortex::counter_t, bool>(), "sample"_a, "sequence"_a, "reversed"_a);
        c.def(py::init<vortex::counter_t, vortex::counter_t, vortex::counter_t, bool, size_t>(),
            "sample"_a, "sequence"_a, "index_in_scan"_a, "reversed"_a, "segment_count_hint"_a = 0);
        c.def(py::init < vortex::counter_t, vortex::counter_t, vortex::counter_t, bool, size_t, vortex::marker::volume_boundary::flags_t>(),
            "sample"_a, "sequence"_a, "index_in_scan"_a, "reversed"_a, "segment_count_hint"_a, "flags"_a);
        c.def("__repr__", [](const C& v) {
            return fmt::format("VolumeBoundary(sample={}, sequence={}, index_in_scan={}, reversed={}, segment_count_hint={}, flags={})",
                PY_REPR(v.sample), PY_REPR(v.sequence), PY_REPR(v.index_in_scan), PY_REPR(v.reversed), PY_REPR(v.segment_count_hint), PY_REPR(v.flags));
        });

        RW_VAR(sample);
        RW_VAR(sequence);
        RW_VAR(index_in_scan);
        RW_VAR(reversed);
        RW_VAR(segment_count_hint);
        RW_VAR(flags);
    }

    {
        using C = vortex::marker::segment_boundary;
        CLS_VAL(SegmentBoundary);

        c.def(py::init());
        c.def(py::init<vortex::counter_t, vortex::counter_t, bool>(), "sample"_a, "sequence"_a, "reversed"_a);
        c.def(py::init<vortex::counter_t, vortex::counter_t, vortex::counter_t, bool, size_t>(),
            "sample"_a, "sequence"_a, "index_in_volume"_a, "reversed"_a, "record_count_hint"_a = 0);
        c.def(py::init<vortex::counter_t, vortex::counter_t, vortex::counter_t, bool, size_t, vortex::marker::segment_boundary::flags_t>(),
            "sample"_a, "sequence"_a, "index_in_volume"_a, "reversed"_a, "record_count_hint"_a, "flags"_a);
        c.def("__repr__", [](const C& v) {
            return fmt::format("SegmentBoundary(sample={}, sequence={}, index_in_volume={}, reversed={}, record_count_hint={}, flags={})",
                PY_REPR(v.sample), PY_REPR(v.sequence), PY_REPR(v.index_in_volume), PY_REPR(v.reversed), PY_REPR(v.record_count_hint), PY_REPR(v.flags));
        });

        RW_VAR(sample);
        RW_VAR(sequence);
        RW_VAR(index_in_volume);
        RW_VAR(reversed);
        RW_VAR(record_count_hint);
        RW_VAR(flags);

    }


    {
        using C = vortex::marker::active_lines;
        CLS_VAL(ActiveLines);
        c.def(py::init());
        c.def("__repr__", [](const C& v) {return fmt::format("ActiveLines(sample={})", v.sample); });

        RW_VAR(sample);
    }

    {
        using C = vortex::marker::inactive_lines;
        CLS_VAL(InactiveLines);
        c.def(py::init());
        c.def("__repr__", [](const C& v) {return fmt::format("InactiveLines(sample={})", v.sample); });

        RW_VAR(sample);
    }

    {
        using C = vortex::marker::event;
        CLS_VAL(Event);
        c.def(py::init());
        c.def("__repr__", [](const C& v) {return fmt::format("Event(sample={}, id={}, flags={})", v.sample, v.id, PY_REPR(v.flags)); });

        RW_VAR(sample);
        RW_VAR(id);
        RW_VAR(flags);
    }

    py::bind_vector<std::vector<vortex::default_marker_t>>(m, "MarkerList")
        .def("__repr__", [](const std::vector<vortex::default_marker_t>& v) {
        return list_repr(v, "MarkerList");
    });

    {
        using C = vortex::default_marker_flags_t;
        CLS_VAL(Flags);
        c.def(py::init());
        c.def(py::init<size_t>());
        c.def(py::self == py::self);
        c.def("__repr__", [](const C& v) {return fmt::format("Flags(value={:#x})", v.value); });

        RW_VAR(value);

        c.def("clear", py::overload_cast<>(&C::clear), "Clears all bits");
        c.def("clear", py::overload_cast<uint8_t>(&C::clear), "Clears indicated bit");

        c.def("set", py::overload_cast<>(&C::set), "Sets all bits");
        c.def("set", py::overload_cast<uint8_t>(&C::set), "Sets indicated bit");

        // TODO: matches is not bound because operator== is already defined and pybind11 can't handle static/object overloads.

        SFXN(all);
        SFXN(none);

        c.def_property_readonly_static("max_unique", &C::max_unique);
        // TODO: max_unique.. Couldn't figure out how to bind it

        SHALLOW_COPY();

        py::bind_vector<std::vector<C>>(m, "FlagsList")
            .def("__repr__", [](const std::vector<C>& v) { return list_repr(v, "FlagsList");
        });
    }

}
