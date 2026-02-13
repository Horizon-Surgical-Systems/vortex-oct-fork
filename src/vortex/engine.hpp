#pragma once

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/engine.hpp>

namespace vortex {

    template<typename acquire_element_t, typename process_element_t, typename analog_element_t, typename digital_element_t>
    using engine_t = engine::engine_t<
        engine::engine_config_t<
            engine::block_t<acquire_element_t, process_element_t, analog_element_t, digital_element_t>,
            default_warp_t
        >
    >;

}

#endif
