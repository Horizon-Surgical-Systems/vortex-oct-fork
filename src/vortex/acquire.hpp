#pragma once

#include <vortex/acquire/null.hpp>
#include <vortex/acquire/file.hpp>

namespace vortex {

    using null_acquire_config_t = acquire::null_config_t;
    template<typename T>
    using null_acquisition_t = acquire::null_acquisition_t<T, null_acquire_config_t>;

    using file_config_t = acquire::file_config_t;
    template<typename T>
    using file_acquisition_t = acquire::file_acquisition_t<T, file_config_t>;

}

#if defined(VORTEX_ENABLE_ALAZAR)

#include <vortex/acquire/dso/alazar/host.hpp>
#include <vortex/acquire/dso/alazar/fft.hpp>

namespace vortex {

    using alazar_config_t = acquire::alazar_config_t<
        alazar::default_clock_t,
        alazar::default_trigger_t,
        alazar::default_option_t
    >;
    using alazar_acquisition_t = acquire::alazar_acquisition_t<alazar_config_t>;

    using alazar_fft_config_t = acquire::alazar_fft_config_t<
        alazar::default_clock_t,
        alazar::default_trigger_t,
        alazar::default_option_t
    >;
    using alazar_fft_acquisition_t = acquire::alazar_fft_acquisition_t<alazar_fft_config_t>;

}

#endif

#if defined(VORTEX_ENABLE_ALAZAR_GPU)

#include <vortex/acquire/dso/alazar/cuda.hpp>

namespace vortex {

    using alazar_gpu_config_t = acquire::alazar_gpu_config_t<
        acquire::default_clock_t,
        acquire::default_trigger_t,
        acquire::default_option_t
    >;

    using alazar_gpu_acquisition_t = acquire::alazar_gpu_acquisition_t<alazar_gpu_config_t>;

}

#endif

#if defined(VORTEX_ENABLE_IMAQ)

#include <vortex/acquire/frame_grabber/imaq.hpp>

namespace vortex {

    using imaq_config_t = acquire::imaq_config_t;

    using imaq_acquisition_t = acquire::imaq_acquisition_t<
        imaq_config_t,
        imaq::imaq_input_ts<uint8_t, uint16_t>
    >;

}

#endif

#if defined(VORTEX_ENABLE_TELEDYNE)

#include <vortex/acquire/dso/teledyne.hpp>

namespace vortex {

    using teledyne_config_t = acquire::teledyne_config_t;

    using teledyne_acquisition_t = acquire::teledyne_acquisition_t;

}

#endif
