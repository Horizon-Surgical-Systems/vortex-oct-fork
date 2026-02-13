#pragma once

#include <vortex/io/null.hpp>

namespace vortex {

    using null_io_config_t = io::null_config_t;

    using null_io_t = io::null_io_t<null_io_config_t>;

}

#if defined(VORTEX_ENABLE_DAQMX)

#include <vortex/io/daqmx.hpp>

namespace vortex {

    using daqmx_config_t = io::daqmx_config_t<daqmx::default_channel_t>;

    using daqmx_io_t = io::daqmx_io_t<daqmx_config_t>;

}

#endif

#if defined(VORTEX_ENABLE_ALAZAR_DAC)

#include <vortex/io/alazar.hpp>

namespace vortex {

    using alazar_io_config_t = io::alazar_config_t;

    using alazar_io_t = io::alazar_io_t<alazar_io_config_t>;

}

#endif

#if defined(VORTEX_ENABLE_ASIO)

#include <vortex/io/machdsp.hpp>

namespace vortex {

    using machdsp_config_t = io::machdsp_config_t;

    using machdsp_io_t = io::machdsp_io_t<machdsp_config_t>;

}

#endif
