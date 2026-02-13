#pragma once
#include <vortex-python/bind/common.hpp>

#include <vortex/memory.hpp>
#include <vortex/util/sync.hpp>

namespace pybind11::detail {

    template <typename T, size_t N, typename A, bool Init>
    struct type_caster<xt::svector<T, N, A, Init>>
        : list_caster<xt::svector<T, N, A, Init>, T> { };

    template <typename T>
    struct type_caster<vortex::cpu_view_t<T>> {
    public:
        PYBIND11_TYPE_CASTER(vortex::cpu_view_t<T>, _("numpy.ndarray[numpy.") + _(::detail::dtype<std::decay_t<T>>::name) + _("]"));

        // Python -> C++
        // ref: https://stackoverflow.com/questions/42645228/cast-numpy-array-to-from-custom-c-matrix-class-using-pybind11
        bool load(handle src, bool /*convert*/) {
            // access the data
            auto buf = src.cast<py::buffer>();
            if (!buf) {
                return false;
            }
            auto info = buf.request(!std::is_const_v<T>);

            // check compatible types
            if (info.format != py::format_descriptor<T>::format()) {
                return false;
            }

            // convert from stride in bytes to stride in elements
            auto stride = info.strides;
            ptrdiff_t max_offset = 0;
            for (size_t i = 0; i < stride.size(); i++) {
                stride[i] /= sizeof(T);
                max_offset += stride[i] + info.shape[i];
            }

            // construct the view
            auto ptr = static_cast<T*>(info.ptr);
            auto begin = std::min<size_t>(0, max_offset);
            auto end = std::max<size_t>(0, max_offset);
            value = vortex::cpu_view_t<T>(ptr, { ptr + begin, ptr + end }, info.shape, stride);
            return true;
        }

        // C++ -> Python
        static handle cast(const vortex::cpu_view_t<T>& src, return_value_policy /* policy */, handle parent) {
            auto dst = py::array(src.shape(), src.stride_in_bytes(), src.data(), parent);

            if (std::is_const_v<T>) {
                // make read-only
                array_proxy(dst.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
            }

            return dst.release();
        }

    };

#if defined(VORTEX_ENABLE_CUDA)

    template <typename T>
    struct type_caster<vortex::cuda::cuda_view_t<T>> {
    public:
        PYBIND11_TYPE_CASTER(vortex::cuda::cuda_view_t<T>, _("cupy.ndarray[cupy.") + _(::detail::dtype<std::decay_t<T>>::name) + _("]"));

        // Python -> C++
        bool load(handle src, bool /* convert */) {
            py::object dtype;
            vortex::cuda::device_t device;
            intptr_t ptr;
            std::vector<size_t> shape, stride_in_bytes;
            size_t size;

            // extract required variables from input
            try {
                dtype = src.attr("dtype");

                device = src.attr("device").attr("id").cast<vortex::cuda::device_t>();
                ptr = src.attr("data").attr("ptr").cast<intptr_t>();

                shape = src.attr("shape").cast<std::vector<size_t>>();
                stride_in_bytes = src.attr("strides").cast<std::vector<size_t>>();
                size = src.attr("size").cast<size_t>();
            } catch (const py::error_already_set&) {
                PyErr_Clear();
                return false;
            }

            // verify compatible data types
            const auto& api = detail::npy_api::get();
            if (!api.PyArray_EquivTypes_(dtype.ptr(), py::dtype::of<T>().ptr())) {
                return false;
            }

            // convert from stride in bytes to stride in elements
            auto stride = stride_in_bytes;
            ptrdiff_t max_offset = 0;
            for (size_t i = 0; i < stride.size(); i++) {
                stride[i] /= sizeof(T);
                max_offset += stride[i] * shape[i];
            }

            // construct the view
            auto tptr = reinterpret_cast<T*>(ptr);
            auto begin = std::min<size_t>(0, max_offset);
            auto end = std::max<size_t>(0, max_offset);
            vortex::cuda::cuda_view_t<T> view(tptr, { tptr + begin, tptr + end }, shape, stride, device);

            value = std::move(view);
            return true;
        }

        // C++ -> Python
        static handle cast(const vortex::cuda::cuda_view_t<T>& src, return_value_policy /* policy */, handle parent) {
            auto cupy = py::module::import("cupy");

            // construct support objects
            auto mem = cupy.attr("cuda").attr("UnownedMemory")(intptr_t(src.data()), src.count(), parent, src.device());
            auto ptr = cupy.attr("cuda").attr("MemoryPointer")(mem, 0);

            // construct CuPy array
            auto dst = cupy.attr("ndarray")(src.shape(), ::dtype<T>::name, ptr, src.stride_in_bytes());

            return dst.release();
        }

    };

#endif

}

template<typename C>
static void setup_base_tensor(const std::string& name, py::class_<C, std::shared_ptr<C>>& c) {

    c.def("__repr__", [name](const C& o) {
        return fmt::format("{}(shape=[{}], dtype={})", name, vortex::join(o.shape(), ", "), dtype<typename C::element_t>::name);
    });

    RO_ACC(data);
    c.def_property_readonly("dtype", [](const C&) { return py::dtype::of<typename C::element_t>(); });
    RO_ACC(valid);

    RO_ACC(count);
    RO_ACC(size_in_bytes);
    RO_ACC(stride_in_bytes);
    RO_ACC(underlying_size_in_bytes);

    RO_ACC(dimension);
    RO_ACC(shape);
    RO_ACC(stride);

    c.def("resize", [](C& o, std::vector<size_t> shape) {o.resize(shape); }, "shape"_a);
    FXN(shrink);
    FXN(clear);

    c.def("__exit__", [](C& o, py::object exc_type, py::object exc_value, py::object traceback) {
        o.mutex().unlock_shared();
        return false;
    });

}

#if defined(VORTEX_ENABLE_CUDA)

template<typename C>
static void bind_cuda_tensor(py::class_<C, std::shared_ptr<C>>& c, const std::string& base_name, py::object& cupy) {
    setup_base_tensor(base_name, c);

    c.def(py::init());

    if (cupy) {
        c.def("__enter__", [cupy](C& o) {
            {
                py::gil_scoped_release gil;
                o.mutex().lock_shared();
            }

            return view(o);
        });
    } else {
        c.def("__enter__", [](C& o) {
            throw std::runtime_error("CuPy (https://cupy.dev/) is required for CUDA tensor interoperability");
        });
    }

}

#endif

// NOTE: make inline so that no need to link against vortex-python
inline py::module try_import_cupy() {
    try {
        return py::module::import("cupy");
    } catch (const py::error_already_set& e) {
        if (!e.matches(PyExc_ImportError)) {
            throw;
        }
        return {};
    }
}
