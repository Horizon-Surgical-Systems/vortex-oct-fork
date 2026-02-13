#pragma once

#include <vortex/engine.hpp>

#if defined(VORTEX_ENABLE_ENGINE)
    using engine_t = vortex::engine_t<uint16_t, int8_t, double, uint32_t>;
    using block_t = engine_t::block_t;
#endif
