/** \rst

    cube formatter that stacks segments in a rectangular volume

    The cube formatter uses the formatting plan to extract segments from
    the inputs blocks and builds a rectangular volume.  The volume is
    stored in row-major format with dimensions of #segments/volume x
    #records/segment x #samples/record x #channels/sample.  The alternative
    shape naming is #B-scans/volume x #A-scans/B-scan x #samples/A-scan x
    #channels/sample.  The cube formatter can produce a 3D or 4D volume,
    corresponding to 2D or 3D segments.  The number of channels is
    assumed to be one if needed.  This formatter works best for scan
    patterns where each segment has equal length (e.g., raster or radial
    scans).

 \endrst */

#pragma once

#include <vortex/util/copy.hpp>

#include <vortex/format/common.hpp>
#include <vortex/format/plan.hpp>

namespace vortex::format {
    
    struct stack_format_executor_config_t {

        bool erase_after_volume = false;
        copy::slice_t sample_slice = copy::slice::none_t{};
        copy::transform_t sample_transform = copy::transform::none_t{};
    };

    template<typename config_t_>
    class stack_format_executor_t {
    public:

        using config_t = config_t_;

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            std::swap(_config, config);
        }

        template<typename V1, typename V2>
        void execute(const cpu_viewable<V1>& volume_buffer, const cpu_viewable<V2>& block_buffer, const action::copy& action) const {
            copy::options_block2volume_t options{
                action.count,
                action.block_offset,
                { action.buffer_segment, action.buffer_record },
                action.reverse, _config.sample_slice, _config.sample_transform
            };
            copy::copy(block_buffer, volume_buffer, options);
        }

    public:

#if defined(VORTEX_ENABLE_CUDA)

        template<typename V1, typename V2, typename = std::enable_if_t<cuda::is_cuda_viewable<V1> || cuda::is_cuda_viewable<V2>>>
        void execute(const cuda::stream_t& stream, const viewable<V1>& volume_buffer_, const viewable<V2>& block_buffer_, const action::copy& action) const {
            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& block_buffer = block_buffer_.derived_cast();

            // check that buffers are compatible for copy
            auto [dst_shape, dst_stride] = detail::trim_shape_and_stride(volume_buffer, 4);
            auto [src_shape_, src_stride] = detail::trim_shape_and_stride(block_buffer, 3);
            auto& src_shape = src_shape_; // to satisfy clang

            if (dst_shape.size() != 4) {
                throw std::invalid_argument(fmt::format("volume must have at most 4 non-trivial dimensions: {}", shape_to_string(volume_buffer.shape())));
            }
            if(src_shape.size() != 3) {
                throw std::invalid_argument(fmt::format("block must have at most 3 non-trivial dimensions: {}", shape_to_string(block_buffer.shape())));
            }

            // get sample slicing details
            std::visit(overloaded{
                [&](const copy::slice::none_t& s) {},
                [&](const copy::slice::simple_t& s) { src_shape[1] = s.count(); }
            }, _config.sample_slice);

            for (size_t i = 0; i < src_shape.size() - 1; i++) {
                auto dst_dim = dst_shape.size() - i - 1;
                auto src_dim = src_shape.size() - i - 1;

                if (dst_shape[dst_dim] != src_shape[src_dim]) {
                    throw std::invalid_argument(fmt::format("incompatible volume and block shapes for dimensions {}/{}: [{}] vs [{}]", dst_dim, src_dim, shape_to_string(dst_shape), shape_to_string(src_shape)));
                }
            }

            // launch the copy
            copy::options_block2volume_t options{
                action.count,
                action.block_offset,
                { action.buffer_segment, action.buffer_record },
                action.reverse,
                _config.sample_slice, _config.sample_transform
            };
            copy::copy(stream, block_buffer, volume_buffer, options);
        }

#endif

        config_t _config;

    };

}
