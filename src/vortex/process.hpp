#pragma once

#include <vortex/process/null.hpp>
#include <vortex/process/copy.hpp>

namespace vortex {

    using null_processor_t = process::null_processor_t<process::processor_config_t<float>>;

    template<typename input_element_t, typename output_element_t>
    using copy_processor_t = process::copy_processor_t<
        input_element_t,
        output_element_t,
        process::copy_processor_config_t
    >;

}

#if defined(VORTEX_ENABLE_FFTW)

#include <vortex/process/cpu.hpp>

namespace vortex {

    template<typename input_element_t, typename output_element_t>
    using cpu_processor_t = process::cpu_processor_t<
        input_element_t,
        output_element_t,
        float,
        uint32_t,
        process::cpu_processor_config_t<float>
    >;

}

#endif

#if defined(VORTEX_ENABLE_CUDA)

#include <vortex/process/cuda.hpp>

namespace vortex {

    template<typename input_element_t, typename output_element_t>
    using cuda_processor_t = process::cuda_processor_t<
        input_element_t,
        output_element_t,
        float,
        uint32_t,
        process::cuda_processor_config_t<float>
    >;

}

#endif
