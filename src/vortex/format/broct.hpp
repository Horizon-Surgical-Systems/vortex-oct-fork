/** \rst

    cube formatter that produces BROCT-compatible output

    This formatter is essentially a cube formatter that transposes the
    B-scans during formatting so that the resultant volume is in BROCT
    format for easy saving.  It is not intended to be used outside of
    that role.

 \endrst */

 #pragma once

#include <vortex/format/stack.hpp>

#include <vortex/util/copy.hpp>

#if defined(VORTEX_ENABLE_CUDA)
#  include <vortex/memory/cuda.hpp>
#endif

namespace vortex::format {

    template<typename config_t>
    class broct_format_executor_t : public stack_format_executor_t<config_t> {
    public:

        using base_t = stack_format_executor_t<config_t>;

        template<typename V1, typename V2>
        void execute(const cpu_viewable<V1>& volume_buffer_, const cpu_viewable<V2>& block_buffer_, const action::copy& action) const {
            auto volume_buffer = volume_buffer_.derived_cast().to_xt();
            auto block_buffer = block_buffer_.derived_cast().to_xt();

            // handle copy action manually to perform transposing
            auto src = xt::view(block_buffer, xt::range(action.block_offset, action.block_offset + action.count), xt::all(), 0);
            auto dst = xt::view(volume_buffer, action.buffer_segment, xt::all(), xt::range(action.buffer_record, action.buffer_record + action.count));
            auto out = xt::transpose(dst);
            
            copy::options_block2volume_t options{
                action.count,
                action.block_offset,
                { action.buffer_segment, action.buffer_record },
                action.reverse, _config.sample_slice, _config.sample_transform
            };
            copy::detail::copy_by_xtensor(src, out, options);
        }

#if defined(VORTEX_ENABLE_CUDA)

        template<typename V1, typename V2, typename = std::enable_if_t<cuda::is_cuda_viewable<V1> || cuda::is_cuda_viewable<V2>>>
        void execute(const cuda::stream_t& stream, const viewable<V1>& volume_buffer_, const viewable<V2>& block_buffer_, const action::copy& action) const {
            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& block_buffer = block_buffer_.derived_cast();

            // check that buffers are compatible for copy
            // NOTE: enforce strict dimension requirements since BROCT only supports 3D volumes
            if(volume_buffer.dimension() < 3) {
                throw std::invalid_argument("volume buffer must be at least three-dimensional");
            }
            if (std::any_of(volume_buffer.shape().begin() + 3, volume_buffer.shape().end(), [](const auto& v) { return v != 1; })) {
                throw std::invalid_argument("volume buffer dimensions after the first 3 must have lengths of 1");
            }
            if (block_buffer.dimension() < 2) {
                throw std::invalid_argument("block buffer must be at least two-dimensional");
            }
            if (std::any_of(block_buffer.shape().begin() + 2, block_buffer.shape().end(), [](const auto& v) { return v != 1; })) {
                throw std::invalid_argument("block buffer dimensions after the first 2 must have lengths of 1");
            }

            // get sample slicing details
            auto slice = copy::slice::to_simple(_config.sample_slice, block_buffer.shape(1));

            // require A-scan length to match, taking the transpose into account
            if (volume_buffer.shape(1) != slice.count()) {
                throw std::invalid_argument(fmt::format("incompatible volume and block record lengths: {} vs {}", volume_buffer.shape(1), slice.count()));
            }

            // enforce the B-scan width
            if (action.buffer_record + action.count > volume_buffer.shape(2)) {
                throw std::invalid_argument(fmt::format("B-scan width to too large for volume buffer: {} vs {}", action.count, volume_buffer.shape(2)));
            }
            // enforce the volume C-scan width
            if (action.buffer_segment > volume_buffer.shape(0)) {
                throw std::invalid_argument(fmt::format("segment index to too large for volume buffer: {} vs {}", action.buffer_segment, volume_buffer.shape(0)));
            }

            // launch the copy
            _copy_transpose(stream, volume_buffer, block_buffer, action);
        }

    protected:

        template<typename V1, typename V2>
        void _copy_transpose(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer, const cpu_viewable<V2>& block_buffer, const action::copy& a) const {
            _copy_transpose_by_memcpy(stream, volume_buffer, block_buffer, a);
        }
        template<typename V1, typename V2>
        void _copy_transpose(const cuda::stream_t& stream, const cpu_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& block_buffer, const action::copy& a) const {
            _copy_transpose_by_memcpy(stream, volume_buffer, block_buffer, a);
        }
        template<typename V1, typename V2>
        void _copy_transpose(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer_, const cuda::cuda_viewable<V2>& block_buffer_, const action::copy& a) const {
            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& block_buffer = block_buffer_.derived_cast();

            // always prefer kernel copy over memcpy due to transposing speed
            if (volume_buffer.is_accessible() && block_buffer.is_accessible()) {
                _copy_transpose_by_kernel(stream, volume_buffer, block_buffer, a);
            } else {
                _copy_transpose_by_memcpy(stream, volume_buffer, block_buffer, a);
            }
        }

        template<typename V1, typename V2>
        void _copy_transpose_by_memcpy(const cuda::stream_t& stream, const viewable<V1>& volume_buffer_, const viewable<V2>& block_buffer_, const action::copy& a) const {
            if (!std::holds_alternative<copy::transform::none_t>(_config.sample_transform)) {
                throw std::invalid_argument("transformation not supported with memcpy-based formatting");
            }

            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& block_buffer = block_buffer_.derived_cast();

            // get sample slicing details
            auto slice = copy::slice::to_simple(_config.sample_slice, block_buffer.shape(1));

            // copy each record at a time while transposing
            for (size_t record = 0; record < a.count; record++) {
                size_t src_begin, dst_begin;

                if (a.reverse) {
                    // move backwards through records
                    src_begin = block_buffer.offset({ a.block_offset + a.count - record - 1, slice.start });
                    dst_begin = volume_buffer.offset({ a.buffer_segment, 0, a.buffer_record + record });
                } else {
                    // move forwards through records
                    src_begin = block_buffer.offset({ a.block_offset + record, slice.start });
                    dst_begin = volume_buffer.offset({ a.buffer_segment, 0, a.buffer_record + record });
                }

                // copy a single record
                cuda::copy(
                    block_buffer, src_begin, block_buffer.stride(1) * slice.step,  // copy sequential samples in the block
                    volume_buffer, dst_begin, volume_buffer.stride(1),             // space the samples by the record length to transpose
                    slice.count(), 1,                                              // each sample is a "row" so the pitch is the offset between them during copying
                    &stream
                );
            }
        }

        template<typename V1, typename V2>
        void _copy_transpose_by_kernel(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer_, const cuda::cuda_viewable<V2>& block_buffer_, const action::copy& a) const {
            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& block_buffer = block_buffer_.derived_cast();

            // get sample slicing details
            auto slice = copy::slice::to_simple(_config.sample_slice, block_buffer.shape(1));

            // adjust for the slice step
            std::array<ptrdiff_t, 3> src_stride = { block_buffer.stride(0), block_buffer.stride(1) * downcast<ptrdiff_t>(slice.step), 0 };
            std::array<ptrdiff_t, 3> dst_stride = { volume_buffer.stride(2), volume_buffer.stride(1), 0 };

            size_t src_begin;
            // different stride calculation to perform transpose
            auto dst_begin = volume_buffer.offset({ a.buffer_segment }) + strided_offset(dst_stride, a.buffer_record);

            if (a.reverse) {
                // offset to start of last record in block for reverse copy
                src_begin = block_buffer.offset({ a.block_offset + a.count - 1, slice.start });
                // stride backwards through block records
                src_stride[0] *= -1;
            } else {
                // offset to start of first record in block for forward copy
                src_begin = block_buffer.offset({ a.block_offset, slice.start });
            }

            // get sample transform details
            auto transform = copy::transform::to_linear<float>(_config.sample_transform);

            // launch the copy
            copy::detail::linear_transform(
                stream, transform.scale, transform.offset,
                { a.count, slice.count(), 1 },
                block_buffer.data(), src_begin, src_stride,
                volume_buffer.data(), dst_begin, dst_stride
            );
        }

#endif

        using base_t::_config;

    };

}

