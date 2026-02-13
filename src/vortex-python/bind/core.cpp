#include <vortex-python/bind/common.hpp>

#include <vortex/core.hpp>

template<typename T>
static auto bind_range(py::module& m) {
    using C = vortex::range_t<T>;
    CLS_VAL(Range);

    c.def(py::init());
    c.def(py::init([](T min, T max) -> C { return { {min, max} }; }));
    c.def(py::self == py::self);

    RO_ACC(length);

    RW_ACC(min);
    RW_ACC(max);

    SFXN(symmetric);
    FXN(contains);

    SHALLOW_COPY();

    c.def("__repr__", [](const C& o) {
        return fmt::format("Range({}, {})", o.min(), o.max());
    });

    //c.def("__array__", [](const C & o) {
    //    return std::vector<T>{ o.min(), o.max() };
    //});

    return c;
}


void bind_core(py::module& m) {
    bind_range<double>(m);
}
