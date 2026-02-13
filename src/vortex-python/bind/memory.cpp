#include <vortex-python/bind/memory.hpp>

#include <vortex/util/sync.hpp>

using namespace std::string_literals;

template<typename T>
static void bind_cpu_tensor(py::module& m) {
    using C = vortex::sync::lockable<vortex::cpu_tensor_t<T>>;
    auto c = py::class_<C, std::shared_ptr<C>>(m, ("CpuTensor"s + dtype<T>::display_name).c_str());

    setup_base_tensor("CpuTensor", c);

    c.def(py::init());

    c.def("__enter__", [](C& o) {
        {
            py::gil_scoped_release gil;
            o.mutex().lock_shared();
        }

        return view(o);
    });

}

#if defined(VORTEX_ENABLE_CUDA)

template<typename T>
static void bind_cuda_host_tensor(py::module& m, py::object& cupy) {
    using C = vortex::sync::lockable<vortex::cuda::cuda_host_tensor_t<T>>;
    auto c = py::class_<C, std::shared_ptr<C>>(m, fmt::format("CudaHostTensor{}", dtype<T>::display_name).c_str());

    bind_cuda_tensor(c, "CudaHostTensor", cupy);
}

template<typename T>
static void bind_cuda_device_tensor(py::module& m, py::object& cupy) {
    using C = vortex::sync::lockable<vortex::cuda::cuda_device_tensor_t<T>>;
    auto c = py::class_<C, std::shared_ptr<C>>(m, fmt::format("CudaDeviceTensor{}", dtype<T>::display_name).c_str());

    bind_cuda_tensor(c, "CudaDeviceTensor", cupy);

    RO_ACC(device);
}

#endif

void bind_memory(py::module& root) {
    auto m = root.def_submodule("memory");

    bind_cpu_tensor<int8_t>(m);
    bind_cpu_tensor<uint16_t>(m);
    bind_cpu_tensor<uint64_t>(m);
    bind_cpu_tensor<float>(m);
    bind_cpu_tensor<double>(m);

#if defined(VORTEX_ENABLE_CUDA)
    auto cupy = try_import_cupy();

    bind_cuda_host_tensor<int8_t>(m, cupy);
    bind_cuda_host_tensor<uint16_t>(m, cupy);
    bind_cuda_host_tensor<uint64_t>(m, cupy);
    bind_cuda_host_tensor<float>(m, cupy);
    bind_cuda_host_tensor<double>(m, cupy);

    bind_cuda_device_tensor<int8_t>(m, cupy);
    bind_cuda_device_tensor<uint16_t>(m, cupy);
    bind_cuda_device_tensor<uint64_t>(m, cupy);
    bind_cuda_device_tensor<float>(m, cupy);
    bind_cuda_device_tensor<double>(m, cupy);
#endif

}
