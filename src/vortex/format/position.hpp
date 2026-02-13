#pragma once

#include <vortex/format/stack.hpp>


#if defined(VORTEX_ENABLE_CUDA)
#  include <vortex/driver/cuda/types.hpp>
#endif

namespace vortex::format {

#if defined(VORTEX_ENABLE_CUDA)

    namespace detail {

        void position_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const int8_t, 2>& block,
            const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
            const cuda::strided_t<int8_t, 3>& volume,
            const cuda::matrix_t<float, 2, 3>& transform
        );
        void position_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const uint16_t, 2>& block,
            const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
            const cuda::strided_t<uint16_t, 3>& volume,
            const cuda::matrix_t<float, 2, 3>& transform
        );
        void position_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const float, 2>& block,
            const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
            const cuda::strided_t<int8_t, 3>& volume,
            const cuda::matrix_t<float, 2, 3>& transform
        );

    }

#endif

    struct position_format_executor_config_t : stack_format_executor_config_t {

        xt::xtensor_fixed<double, xt::xshape<2, 3>> transform;

        bool use_target_position = true;
        std::array<size_t, 2> channels = { 0, 1 };

        position_format_executor_config_t() {
            set();
        }

        void set() {
            set({ 1, 1 });
        }
        void set(std::array<double, 2> pitch) {
            set(pitch, { 0, 0 });
        }
        void set(std::array<double, 2> pitch, std::array<double, 2> offset) {
            set(pitch, offset, 0);
        }
        void set(std::array<double, 2> pitch, std::array<double, 2> offset, double angle) {
            transform = { {
                { pitch[0] * std::cos(angle), -pitch[0] * std::sin(angle), offset[0]},
                { pitch[1] * std::sin(angle), pitch[1] * std::cos(angle), offset[1]}
            } };
        }

    };

    template<typename config_t_>
    class position_format_executor_t {
    public:

        using config_t = config_t_;

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            std::swap(_config, config);
        }

#if defined(VORTEX_ENABLE_CUDA)

        template<typename V1, typename V2, typename V3>
        void format(const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& sample_target, const cuda::cuda_viewable<V2>& sample_actual, const cuda::cuda_viewable<V3>& segment_buffer, size_t segment_index, bool reverse = false) const {
            format(cuda::stream_t::default_(), segment_buffer, volume_buffer, segment_index, reverse);
        }
        template<typename V1, typename V2, typename V3>
        void format(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& sample_target, const cuda::cuda_viewable<V2>& sample_actual, const cuda::cuda_viewable<V3>& segment_buffer, size_t segment_index, bool reverse = false) const {
            format(stream, volume_buffer, segment_buffer, segment_index, 0, reverse);
        }
        template<typename V1, typename V2, typename V3>
        void format(const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& sample_target, const cuda::cuda_viewable<V2>& sample_actual, const cuda::cuda_viewable<V3>& segment_buffer, size_t segment_index, size_t record_index, bool reverse = false) const {
            format(cuda::stream_t::default_(), volume_buffer, sample_target, sample_actual, segment_buffer, segment_index, record_index, reverse);
        }
        template<typename V1, typename V2, typename V3>
        void format(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer_, const cuda::cuda_viewable<V2>& sample_target_, const cuda::cuda_viewable<V2>& sample_actual_, const cuda::cuda_viewable<V3>& segment_buffer_, size_t segment_index, size_t record_index, bool reverse = false) const {

            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& segment_buffer = segment_buffer_.derived_cast();

            // check that buffers are compatible for copy
            auto [dst_shape, dst_stride] = detail::trim_shape_and_stride(volume_buffer, 3);
            auto [src_shape_, src_stride] = detail::trim_shape_and_stride(segment_buffer, 2);
            auto& src_shape = src_shape_; // to satisfy clang

            if (dst_shape.size() != 3) {
                throw std::invalid_argument(fmt::format("volume must have at most 3 non-trivial dimensions: {}", shape_to_string(volume_buffer.shape())));
            }
            if (src_shape.size() != 2) {
                throw std::invalid_argument(fmt::format("block must have at most 2 non-trivial dimensions: {}", shape_to_string(segment_buffer.shape())));
            }

            // get sample slicing details
            std::visit(overloaded{
                [&](const copy::slice::none_t& s) {},
                [&](const copy::slice::simple_t& s) { src_shape[1] = s.count(); }
                }, _config.sample_slice);

            // only require a match along the A-scan direction
            auto dst_dim = dst_shape.size() - 1;
            auto src_dim = src_shape.size() - 1;
            if (dst_shape[dst_dim] != src_shape[src_dim]) {
                throw std::invalid_argument(fmt::format("incompatible volume and block shapes for dimensions {}/{}: {} vs {}", dst_dim, src_dim, dst_shape[dst_dim], src_shape[src_dim]));
            }

            // check that sample signals are properly sized
            auto& position_buffer = (_config.use_target_position ? sample_target_.derived_cast() : sample_actual_.derived_cast());
            auto [position_shape, _] = detail::trim_shape_and_stride(position_buffer, 2);

            if (position_shape.size() != 2) {
                throw std::invalid_argument(fmt::format("position must have at most 2 non-trivial dimensions: {}", shape_to_string(position_buffer.shape())));
            }
            if (position_shape[0] != src_shape[0]) {
                throw std::invalid_argument(fmt::format("position sample is required for every A-scan: {} != {}", position_shape[0], src_shape[0]));
            }
            for (size_t i = 0; i < _config.channels.size(); i++) {
                if (_config.channels[i] >= position_shape[1]) {
                    throw std::invalid_argument(fmt::format("position channel index {} is out of bounds: {} >= {}", i, _config.channels[i], position_shape[1]));
                }
            }

            // launch the rectification
            _rectify_by_kernel<typename V1::element_t, typename V2::element_t, typename V3::element_t>(stream, volume_buffer, position_buffer, segment_buffer.morph_right(2), segment_index, record_index, reverse);
        }

        template<typename V1, typename V2, typename V3>
        void execute(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& block_buffer_, const action::copy& a, const viewable<V3>& sample_target_, const viewable<V3>& sample_actual_) const {
            auto& block_buffer = block_buffer_.derived_cast();
            auto& sample_target = sample_target_.derived_cast();
            auto& sample_actual = sample_actual_.derived_cast();

            format(stream,
                volume_buffer,
                sample_target.range(a.block_offset, a.block_offset + a.count),
                sample_actual.range(a.block_offset, a.block_offset + a.count),
                block_buffer.range(a.block_offset, a.block_offset + a.count),
                a.buffer_segment, a.buffer_record, a.reverse
            );
        }

#endif

    protected:

        config_t _config;

#if defined(VORTEX_ENABLE_CUDA)

        template<typename T1, typename T2, typename T3>
        void _rectify_by_kernel(const cuda::stream_t& stream, const cuda::fixed_cuda_view_t<T1, 3>& volume_buffer_,
            const cuda::fixed_cuda_view_t<const T2, 2>& position_buffer_, const cuda::fixed_cuda_view_t<const T3, 2>& segment_buffer_,
            size_t segment_index, size_t record_index, bool reverse) const {

            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& segment_buffer = segment_buffer_.derived_cast();
            auto& position_buffer = position_buffer_.derived_cast();

            // handling slicing
            auto slice = copy::slice::to_simple(_config.sample_slice, segment_buffer.shape(1));
            
            auto [dst_shape, dst_stride] = volume_buffer.shape_and_stride();
            auto [src_shape, src_stride] = segment_buffer.shape_and_stride();
            
            // adjust for the slice step
            src_shape[1] = slice.count();
            src_stride[1] *= downcast<ptrdiff_t>(slice.step);

            // NOTE: ignore reverse flag since galvo position encodes that already
            // offset to start of first sample in block
            auto src_begin = segment_buffer.offset({ 0, slice.start });
            
            auto transform = copy::transform::to_linear<float>(_config.sample_transform);

            // convert matrix to float
            cuda::matrix_t<float, 2, 3> matrix;
            xt::adapt(&matrix.v[0][0], matrix.count(), xt::no_ownership(), matrix.shape()) = xt::cast<float>(_config.transform);

            // launch the rectification
            detail::position_rectify(
                stream, transform.scale, transform.offset,
                { segment_buffer.data() + src_begin, src_shape, src_stride },
                position_buffer.to_strided(), _config.channels,
                { volume_buffer.data(), dst_shape, dst_stride },
                matrix
            );
        }

#endif

    };

}