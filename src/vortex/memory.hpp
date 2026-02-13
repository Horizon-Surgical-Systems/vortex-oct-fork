#pragma once

#include <vortex/memory/view.hpp>

#include <vortex/memory/cpu.hpp>
#if defined(VORTEX_ENABLE_CUDA)
#   include <vortex/memory/cuda.hpp>
#endif
#if defined(VORTEX_ENABLE_ALAZAR_GPU)
#   include <vortex/memory/alazar.hpp>
#endif
#if defined(VORTEX_ENABLE_TELEDYNE)
#   include <vortex/memory/teledyne.hpp>
#endif
