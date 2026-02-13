#include <vortex-python/bind/common.hpp>
#include <vortex-python/bind/memory.hpp>

#include <spdlog/logger.h>
#include <fmt/format.h>

#include <vortex/marker.hpp>

#include <vortex/storage.hpp>

template<typename T>
using sp = std::shared_ptr<T>;

static void bind_simple_config(py::module& m) {

    py::enum_<vortex::storage::simple_stream_header_t>(m, "SimpleStreamHeader")
        .value("Empty", vortex::storage::simple_stream_header_t::none);

    py::enum_<vortex::storage::simple_stack_header_t>(m, "SimpleStackHeader")
        .value("Empty", vortex::storage::simple_stack_header_t::none)
        .value("NumPy", vortex::storage::simple_stack_header_t::numpy)
        .value("MATLAB", vortex::storage::simple_stack_header_t::matlab)
        .value("NRRD", vortex::storage::simple_stack_header_t::nrrd)
        .value("NIfTI", vortex::storage::simple_stack_header_t::nifti);

    {
        using C = typename vortex::storage::simple_stack_config_t;
        CLS_PTR(SimpleStackConfig);

        c.def(py::init());
        c.def(py::init(
            [](std::string path,
                std::array<size_t, 4> shape,
                bool buffering )
            -> C { return { path, shape, buffering }; }),
            "path"_a = "",
            "shape"_a = std::array<size_t, 4>({ 0, 0, 0, 1 }),
            "buffering"_a = false);

        c.def("__repr__",
            [](const C& v) {
            return fmt::format("SimpleStackConfig(path={}, shape={}, buffering={})",
                PY_REPR(v.path), PY_REPR(v.shape), PY_REPR(v.buffering));
        });

        RW_VAR(path);
        RW_VAR(shape);

        RW_ACC(samples_per_ascan);
        RW_ACC(ascans_per_bscan);
        RW_ACC(bscans_per_volume);

        RO_ACC(volume_shape);
        RO_ACC(bscan_shape);

        RW_VAR(buffering);
        RW_VAR(header);

        SHALLOW_COPY();
    }

    {
        using C = typename vortex::storage::simple_stream_config_t;
        CLS_PTR(SimpleStreamConfig);

        c.def(py::init());
        c.def(py::init(
            [](std::string path,
                bool buffering = false)
            -> C { return { path, buffering }; }),
            "path"_a = "",
            "buffering"_a = false);

        c.def("__repr__",
            [](const C& v) {
            return fmt::format("SimpleStreamConfig(path={}, buffering={})",
                PY_REPR(v.path), PY_REPR(v.buffering));
        });

        RW_VAR(path);

        RW_VAR(buffering);
        RW_VAR(header);

        SHALLOW_COPY();
    }
}

template<typename T, typename C>
static void bind_write_methods(py::class_<C, std::shared_ptr<C>>& c) {
    c.def("write_partial_bscan", [](C& o, size_t bscan_index, size_t ascan_index, const vortex::cpu_view_t<const T>& data) {

        py::gil_scoped_release gil;
        o.write_partial_bscan(bscan_index, ascan_index, data);

    }, "bscan_index"_a, "ascan_index"_a, "data"_a);

    c.def("write_multi_bscan", [](C& o, size_t index, const vortex::cpu_view_t<const T>& data) {

        py::gil_scoped_release gil;
        o.write_multi_bscan(index, data);

    }, "index"_a, "data"_a);

    c.def("write_volume", [](C& o, const vortex::cpu_view_t<const T>& data) {

        py::gil_scoped_release gil;
        o.write_volume(data);

    }, "data"_a);
}

template<typename T>
static void bind_simple_typed(py::module& m) {
    {
        using C = typename vortex::simple_stack_t<T>;
        auto c = py::class_<C, std::shared_ptr<C>>(m, (std::string("SimpleStack") + detail::dtype<T>::display_name).c_str());

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        RO_ACC(config);
        FXN(open);
        FXN(close);
        RO_ACC(ready);

        FXN(advance_volume, "allocate"_a = true);
        FXN(seek, "volume_index"_a , "bscan_index"_a);

        bind_write_methods<T>(c);

    }
    {
        using C = typename vortex::simple_stream_t<T>;
        auto c = py::class_<C, std::shared_ptr<C>>(m, (std::string("SimpleStream") + detail::dtype<T>::display_name).c_str());

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        RO_ACC(config);
        FXN(open);
        FXN(close);
        RO_ACC(ready);

    }
}

static void bind_simple(py::module& m) {
    bind_simple_config(m);
    bind_simple_typed<int8_t>(m);
    bind_simple_typed<uint16_t>(m);
    bind_simple_typed<uint64_t>(m);
    bind_simple_typed<double>(m);
}

#if defined(VORTEX_ENABLE_HDF5)

static void bind_hdf5_config(py::module& m) {

    py::enum_<vortex::storage::hdf5_stack_header_t>(m, "HDF5StackHeader")
        .value("Empty", vortex::storage::hdf5_stack_header_t::none)
        .value("MATLAB", vortex::storage::hdf5_stack_header_t::matlab);

    {
        using C = typename vortex::storage::hdf5_stack_config_t;
        CLS_PTR(HDF5StackConfig);

        c.def(py::init());

        c.def("__repr__",
            [](const C& v) {
            return fmt::format("HDF5StackConfig(path={}, shape={})",
                PY_REPR(v.path), PY_REPR(v.shape));
        });

        RW_VAR(path);
        RW_VAR(shape);

        RW_ACC(channels_per_sample);
        RW_ACC(samples_per_ascan);
        RW_ACC(ascans_per_bscan);
        RW_ACC(bscans_per_volume);

        RO_ACC(volume_shape);
        RO_ACC(bscan_shape);

        RW_VAR(header);
        RW_VAR(compression_level);

        SHALLOW_COPY();
    }
}

template<typename T>
static void bind_hdf5_typed(py::module& m) {
    {
        using C = typename vortex::hdf5_stack_t<T>;
        auto c = py::class_<C, std::shared_ptr<C>>(m, (std::string("HDF5Stack") + detail::dtype<T>::display_name).c_str());

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        RO_ACC(config);
        FXN(open);
        FXN(close);
        RO_ACC(ready);

        FXN(advance_volume, "allocate"_a = true);

        bind_write_methods<T>(c);

    }
}

static void bind_hdf5(py::module& m) {
    bind_hdf5_config(m);
    bind_hdf5_typed<int8_t>(m);
    bind_hdf5_typed<uint16_t>(m);
    bind_hdf5_typed<uint64_t>(m);
    bind_hdf5_typed<double>(m);
}

#endif

static void bind_stream(py::module& m){
    {
        using C = typename vortex::stream_dump_t;
        CLS_PTR(StreamDump);

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        RO_ACC(config);
        FXN(open);
        FXN(close);
        RO_ACC(ready);

    }

    {
        using C = typename vortex::stream_dump_t::config_t;
        CLS_PTR(StreamDumpConfig);

        c.def(py::init());
        c.def(py::init(
                       [](std::string path, size_t stream=0, size_t divisor=1, bool buffering=false)
                       -> C{ return {path, stream, divisor, buffering};}),
              "path"_a,
              "stream"_a = 0,
              "divisor"_a = 1,
              "buffering"_a = false);
        c.def("__repr__",
              [](const C& v){
		  return fmt::format("StreamDumpConfig(path={}, stream={}, divisor={}, buffering={})",
				     PY_REPR(v.path), PY_REPR(v.stream), PY_REPR(v.divisor), PY_REPR(v.buffering));
              });

        RW_VAR(path);
        RW_VAR(stream);
        RW_VAR(divisor);
        RW_VAR(buffering);

        SHALLOW_COPY();
    }
}

static void bind_marker(py::module& m) {
    {
        using C = typename vortex::marker_log_t;
        CLS_PTR(MarkerLog);

        c.def(py::init<sp<spdlog::logger>>(), "log"_a = nullptr);

        RO_ACC(config);
        RO_ACC(ready);

        FXN(open);
        FXN(close);

        c.def("write", [](C& o, const std::vector<vortex::default_marker_t>& markers) { o.write(markers); });
    }

    {
        using C = typename vortex::marker_log_t::config_t;
        CLS_PTR(MarkerLogConfig);

        c.def(py::init());
        c.def(py::init(
            [](std::string path, bool binary, bool buffering)
            -> C { return { path, binary, buffering }; }),
            "path"_a,
            "binary"_a = false,
            "buffering"_a = false);
        c.def("__repr__",
            [](const C& v) {
            return fmt::format("MarkerLogConfig(path={}, binary={}, buffering={})",
                PY_REPR(v.path), PY_REPR(v.binary), PY_REPR(v.buffering));
        });

        RW_VAR(path);
        RW_VAR(binary);
        RW_VAR(buffering);

        SHALLOW_COPY();
    }
}

static void bind_broct(py::module& m) {
    using broct_scan_t = typename vortex::storage::broct_scan_t;
    using scan_type_t = std::underlying_type<broct_scan_t>::type;

    {
        using C = typename vortex::broct_storage_t;
        CLS_PTR(BroctStorage);

        c.def(py::init<sp<spdlog::logger>>(), "log"_a=nullptr);

        RO_ACC(config);
        FXN(open);
        FXN(close);

        FXN(advance_volume, "allocate"_a = true);
        FXN(seek);

        c.def("write_bscan", [](C& o, size_t index, const vortex::cpu_view_t<const typename vortex::broct_storage_t::element_t>& data) {

            py::gil_scoped_release gil;
            o.write_bscan(index, data);

        }, "index"_a, "data"_a);

        c.def("write_multi_bscan", [](C& o, size_t index, const vortex::cpu_view_t<const typename vortex::broct_storage_t::element_t>& data) {

            py::gil_scoped_release gil;
            o.write_multi_bscan(index, data);

        }, "index"_a, "data"_a);

        c.def("write_volume", [](C& o, const vortex::cpu_view_t<const typename vortex::broct_storage_t::element_t>& data) {

            py::gil_scoped_release gil;
            o.write_volume(data);

        }, "data"_a);

    }
    {
        using C = broct_scan_t;
        auto c = py::enum_<C>(m, "BroctScan")
	    .value("rectangular", broct_scan_t::rectangular)
	    .value("bscan", broct_scan_t::bscan)
	    .value("aiming", broct_scan_t::aiming)
	    .value("mscan", broct_scan_t::mscan)
	    .value("radial", broct_scan_t::radial)
	    .value("ascan", broct_scan_t::ascan)
	    .value("speckle", broct_scan_t::speckle)
	    .value("mixed", broct_scan_t::mixed)
	    .value("xfast_yfast", broct_scan_t::xfast_yfast)
	    .value("xfast_yfast_speckle", broct_scan_t::xfast_yfast_speckle)
	    .value("spiral", broct_scan_t::spiral);

        c.export_values();
    }
    {
        using C = typename vortex::broct_storage_t::config_t;
        CLS_PTR(BroctStorageConfig);

        c.def(py::init());
        c.def(py::init(
                       [](std::string path,
                          std::array<size_t, 3> shape={{0,0,0}},
                          std::array<double, 3> dimensions={{1.0, 1.0, 1.0}},
                          broct_scan_t scan_type= broct_scan_t::rectangular,
                          std::string notes="",
                          bool buffering=false)
                       -> C{ return {path, shape, dimensions, scan_type, notes, buffering};}),
              "path"_a="",
              "shape"_a = std::array<size_t, 3> ({0, 0, 0}),
              "dimensions"_a = std::array<double, 3>({1.0, 1.0, 1.0}),
              "scan_type"_a = broct_scan_t::rectangular,
              "notes"_a = "",
              "buffering"_a = false);

        c.def("__repr__",
              [](const C& v){
		  return fmt::format("BroctStorageConfig(path={}, shape={}, dimensions={}, scan_type={}, notes={}, buffering={})",
				     PY_REPR(v.path), PY_REPR(v.shape), PY_REPR(v.dimensions), PY_REPR(v.scan_type), PY_REPR(v.notes), PY_REPR(v.buffering));
              });

        RW_VAR(path);
        RW_VAR(shape);
        RW_VAR(dimensions);
        RW_VAR(scan_type);
        RW_VAR(notes);

        RW_ACC(samples_per_ascan);
        RW_ACC(ascans_per_bscan);
        RW_ACC(bscans_per_volume);

        RO_ACC(broct_volume_shape);
        RO_ACC(broct_bscan_shape);

        RW_VAR(buffering);

        SHALLOW_COPY();
    }
}

void bind_storage(py::module& root){
    auto m = root.def_submodule("storage");

    bind_simple(m);
#if defined(VORTEX_ENABLE_HDF5)
    bind_hdf5(m);
#endif
    bind_stream(m);
    bind_marker(m);
    bind_broct(m);
}
