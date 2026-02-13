#pragma once

#include <vortex/marker.hpp>

#include <vortex/format/stack.hpp>
#include <vortex/format/broct.hpp>

#include <vortex/format/position.hpp>

#include <vortex/format/radial.hpp>
#include <vortex/format/spiral.hpp>

namespace vortex {

    using format_planner_t = format::format_planner_t<format::format_planner_config_t>;

    using stack_format_executor_t = format::stack_format_executor_t<format::stack_format_executor_config_t>;
    using broct_format_executor_t = format::broct_format_executor_t<format::stack_format_executor_config_t>;

    using position_format_executor_t = format::position_format_executor_t<format::position_format_executor_config_t>;

    using radial_format_executor_t = format::radial_format_executor_t<format::radial_format_executor_config_t>;
    using spiral_format_executor_t = format::spiral_format_executor_t<format::spiral_format_executor_config_t>;

}