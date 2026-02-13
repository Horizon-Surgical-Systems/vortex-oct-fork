#define FORCE_IMPORT_ARRAY
#include <vortex-python/bind/common.hpp>
#include <vortex-python/module/loader.hpp>

#include <vortex/version.hpp>

extern void bind_core(py::module& m);
extern void bind_log(py::module& m);

extern void bind_marker(py::module& m);
extern void bind_scan(py::module& m);

extern void bind_acquire(py::module& m);
extern void bind_format(py::module& m);
extern void bind_io(py::module& m);
extern void bind_process(py::module& m);

extern void bind_engine(py::module& m);
extern void bind_storage(py::module& m);

extern void bind_memory(py::module& m);

#if defined(VORTEX_ENABLE_ALAZAR)
extern void bind_alazar(py::module& root);
#endif
static bool try_bind_alazar(py::module& root) {
#if defined(VORTEX_ENABLE_MODULAR_BUILD) && defined(VORTEX_PYTHON_ALAZAR_MODULE)
    return load_and_bind_module(root, "alazar", VORTEX_PYTHON_ALAZAR_MODULE);
#elif defined(VORTEX_ENABLE_ALAZAR)
    bind_alazar(root);
    return true;
#else
    return false;
#endif
}

#if defined(VORTEX_ENABLE_DAQMX)
extern void bind_daqmx(py::module& root);
#endif
static bool try_bind_daqmx(py::module& root) {
#if defined(VORTEX_ENABLE_MODULAR_BUILD) && defined(VORTEX_PYTHON_DAQMX_MODULE)
    return load_and_bind_module(root, "daqmx", VORTEX_PYTHON_DAQMX_MODULE);
#elif defined(VORTEX_ENABLE_DAQMX)
    bind_daqmx(root);
    return true;
#else
    return false;
#endif
}

#if defined(VORTEX_ENABLE_IMAQ)
extern void bind_imaq(py::module& root);
#endif
static bool try_bind_imaq(py::module& root) {
#if defined(VORTEX_ENABLE_MODULAR_BUILD) && defined(VORTEX_PYTHON_IMAQ_MODULE)
    return load_and_bind_module(root, "imaq", VORTEX_PYTHON_IMAQ_MODULE);
#elif defined(VORTEX_ENABLE_IMAQ)
    bind_imaq(root);
    return true;
#else
    return false;
#endif
}

#if defined(VORTEX_ENABLE_TELEDYNE)
extern void bind_teledyne(py::module& root);
#endif
static bool try_bind_teledyne(py::module& root) {
#if defined(VORTEX_ENABLE_MODULAR_BUILD) && defined(VORTEX_PYTHON_TELEDYNE_MODULE)
    return load_and_bind_module(root, "teledyne", VORTEX_PYTHON_TELEDYNE_MODULE);
#elif defined(VORTEX_ENABLE_TELEDYNE)
    bind_teledyne(root);
    return true;
#else
    return false;
#endif
}

#if defined(VORTEX_ENABLE_SIMPLE)
extern bool bind_imaq(py::module& root);
#endif
static bool try_bind_simple(py::module& root) {
#if defined(VORTEX_ENABLE_MODULAR_BUILD) && defined(VORTEX_PYTHON_SIMPLE_MODULE)
    return load_and_bind_module(root, "simple", VORTEX_PYTHON_SIMPLE_MODULE);
#elif defined(VORTEX_ENABLE_SIMPLE)
    bind_simple(root);
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(vortex, m) {
    xt::import_numpy();

    bind_core(m);
    bind_log(m);

    bind_marker(m);
    bind_scan(m);

    bind_acquire(m);
    bind_process(m);
    bind_format(m);
    bind_io(m);

    bind_engine(m);
    bind_storage(m);

    bind_memory(m);

    m.attr("__version__") = VORTEX_VERSION_STRING;

    std::vector<std::string> feature;

    // compile time features
#if defined(VORTEX_ENABLE_ASIO)
    feature.push_back("machdsp");
#endif
#if defined(VORTEX_ENABLE_REFLEXXES)
    feature.push_back("reflexxes");
#endif
#if defined(VORTEX_ENABLE_FFTW)
    feature.push_back("fftw");
#endif
#if defined(VORTEX_ENABLE_CUDA)
    feature.push_back("cuda");
#endif
#if defined(VORTEX_ENABLE_HDF5)
    feature.push_back("hdf5");
#endif
#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
    feature.push_back("cuda_dynamic_resampling");
#endif

    // runtime features
    if (try_bind_alazar(m)) {
        feature.push_back("alazar");
#if defined(VORTEX_ENABLE_ALAZAR_GPU)
        feature.push_back("alazar_gpu");
#endif
#if defined(VORTEX_ENABLE_ALAZAR_DAC)
        feature.push_back("alazar_dac");
#endif
    }
    if (try_bind_daqmx(m)) { feature.push_back("daqmx"); }
    if (try_bind_imaq(m)) { feature.push_back("imaq"); }
    if (try_bind_teledyne(m)) { feature.push_back("teledyne"); }
    if (try_bind_simple(m)) { feature.push_back("simple"); }

    // issue warnings for non-standard debug features
    auto module_name = current_module_path().filename().string();
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS) && defined(VORTEX_ENABLE_CUDA)
    PyErr_WarnExplicit(PyExc_RuntimeWarning, "Vortex is compiled with CUDA kernel serialization enabled. This is a debug feature and may negatively impact performance.", module_name.c_str(), 0, "vortex", NULL);
    feature.push_back("serial_cuda_kernels");
#endif
#if defined(VORTEX_EXCEPTION_GUARDS)
    feature.push_back("exception_guards");
#else
    PyErr_WarnExplicit(PyExc_RuntimeWarning, "Vortex is compiled with exception guards disabled. This is a debug feature. Exceptions raised during callbacks may cause program termination.", module_name.c_str(), 0, "vortex", NULL);
#endif
#if defined(VORTEX_PYBIND11_OPTIMIZATIONS)
    feature.push_back("pybind11_optimizations");
#else
    PyErr_WarnExplicit(PyExc_RuntimeWarning, "Vortex is compiled with pybind11 optimizations disabled. This is a debug feature and may negatively impact performance.", module_name.c_str(), 0, "vortex", NULL);
#endif

    m.attr("__feature__") = feature;

    // warn for non-optimized builds
#if !defined(NDEBUG)
    PyErr_WarnExplicit(PyExc_RuntimeWarning, "Vortex is compiled in debug mode. This may negatively impact performance.", module_name.c_str(), 0, "vortex", NULL);
#endif
}
