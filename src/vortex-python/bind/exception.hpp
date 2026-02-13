#include <pybind11/pybind11.h>

namespace pybind11::detail {

    template<>
    struct type_caster<std::exception_ptr> {
    public:
        PYBIND11_TYPE_CASTER(std::exception_ptr, _("std::exception_ptr"));

        // Python -> C++
        bool load(handle src, bool /* convert */) {
            return false;
        }

        // C++ -> Python
        static handle cast(const std::exception_ptr& src, return_value_policy /* policy */, handle /* parent */) {
            try {
                if (src) {
                    std::rethrow_exception(src);
                }
            } catch (const std::exception& e) {
                // attempt conversion
                // TODO: translate exceptions properly rather than mapping all to RuntimeError (#97)
                return py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(e.what()).release();
            } catch (...) {
                // conversion failed
                return handle();
            }

            // no exception
            return py::none().release();
        }

    };
}
