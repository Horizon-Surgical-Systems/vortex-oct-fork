#include <vortex/process.hpp>

#include <vortex-python/bind/process.hpp>

static void bind_null(py::module& m) {
    using T = vortex::null_processor_t;
    {
        using C = T::config_t;
        CLS_VAL(NullProcessorConfig);

        setup_processor_config(c);
    }

    {
        using C = T;
        CLS_PTR(NullProcessor);

        c.def(py::init<std::shared_ptr<spdlog::logger>>(), "logger"_a = nullptr, doc(c, "__init__"));

        RO_ACC(config);

        FXN_GIL(initialize, "config"_a);
    }
}

static void bind_copy(py::module& m) {
    using T = vortex::copy_processor_t<uint16_t, int8_t>;
    {
        using C = T::config_t;
        CLS_VAL(CopyProcessorConfig);

        setup_processor_config(c);

        RW_VAR(channel);

        RW_VAR(slots);

        RW_VAR(sample_slice);
        RW_VAR(sample_transform);
    }

    {
        using C = T;
        CLS_PTR(CopyProcessor);

        setup_processor(c);

        c.def("next", [](C& o,
            const vortex::cpu_view_t<const typename T::input_element_t>& input_buffer,
            const vortex::cpu_view_t<typename T::output_element_t>& output_buffer, size_t id) {

                py::gil_scoped_release gil;
                o.next(id, input_buffer, output_buffer);

        }, "input_buffer"_a, "output_buffer"_a, "id"_a = 0, doc(c, "next"));

        c.def("next_async", [](C& o,
            const vortex::cpu_view_t<const typename T::input_element_t>& input_buffer,
            const vortex::cpu_view_t<typename T::output_element_t>& output_buffer,
            typename C::callback_t callback, size_t id) {

                py::gil_scoped_release gil;
                o.next_async(id, input_buffer, output_buffer, std::move(callback));

        }, "input_buffer"_a, "output_buffer"_a, "callback"_a, "id"_a = 0, doc(c, "next_async"));
    }
}


#if defined(VORTEX_ENABLE_FFTW)

static void bind_cpu(py::module& m) {
    using T = vortex::cpu_processor_t<uint16_t, int8_t>;
    {
        using C = T::config_t;
        CLS_VAL(CPUProcessorConfig);

        setup_processor_config(c);

        RW_VAR(channel);

        RW_VAR(average_window);

        RW_VAR_XT(spectral_filter);
        RW_VAR_XT(resampling_samples);

        RW_VAR(levels);

        RW_VAR(enable_ifft);
        RW_VAR(enable_log10);
        RW_VAR(enable_square);
        RW_VAR(enable_magnitude);

        RW_VAR(slots);
    }

    {
        using C = T;
        CLS_PTR(CPUProcessor);

        setup_oct_processor<vortex::cpu_view_t<const typename T::input_element_t>, vortex::cpu_view_t<typename T::output_element_t>>(c);
    }
}

#endif

#if defined(VORTEX_ENABLE_CUDA)

static void bind_cuda(py::module& m) {
    using T = vortex::cuda_processor_t<uint16_t, int8_t>;
    {
        using C = T::config_t;
        CLS_VAL(CUDAProcessorConfig);

        setup_processor_config(c);

        RW_VAR(channel);
        RW_VAR(clock_channel);

        RW_VAR(average_window);

        RW_VAR_XT(spectral_filter);
        RW_VAR_XT(resampling_samples);

        RW_VAR(levels);

        RW_VAR(enable_ifft);
        RW_VAR(enable_log10);
        RW_VAR(enable_square);
        RW_VAR(enable_magnitude);

        RW_VAR(device);
        RW_VAR(slots);

        RW_VAR(interpret_as_signed);
    }

    {
        using C = T;
        CLS_PTR(CUDAProcessor);

        setup_oct_processor<vortex::cuda::cuda_view_t<const typename T::input_element_t>, vortex::cuda::cuda_view_t<typename T::output_element_t>>(c);
    }
}

#endif

void bind_process(py::module& root) {
    auto m = root.def_submodule("process");

    bind_null(m);
    bind_copy(m);

#if defined(VORTEX_ENABLE_FFTW)
    bind_cpu(m);
#endif

#if defined(VORTEX_ENABLE_CUDA)
    bind_cuda(m);
#endif

}
