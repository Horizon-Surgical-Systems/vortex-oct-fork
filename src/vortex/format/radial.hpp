/** \rst

    radial rectification

    The radial formatter or rectifier incrementally maps a radial scan
    onto a rectangular grid.  This is done by determine which en face
    records are affected by the segment under consideration.  Those
    records are then updated with the current segment.

    TODO: Linear interpolation is performed in the radial direction only.

    TODO: No effort is made to average records that are updated
          multiple times per scan, which is a waste of data.

 \endrst */

#pragma once

#include <vortex/format/stack.hpp>

namespace vortex::format {

#if defined(VORTEX_ENABLE_CUDA)

    namespace detail {

        void radial_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const int8_t, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
            const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
        );
        void radial_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const float, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
            const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
        );
        void radial_rectify(
            const cuda::stream_t& stream, float scale, float offset,
            const cuda::strided_t<const uint16_t, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
            const cuda::strided_t<uint16_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
        );

    }

#endif

    struct radial_format_executor_config_t : stack_format_executor_config_t {

        std::array<range_t<double>, 2> volume_xy_extent;
        auto& x_extent() { return volume_xy_extent[0]; }
        const auto& x_extent() const { return volume_xy_extent[0]; }
        auto& y_extent() { return volume_xy_extent[1]; }
        const auto& y_extent() const { return volume_xy_extent[1]; }

        std::array<range_t<double>, 2> segment_rt_extent;
        auto& radial_extent() { return segment_rt_extent[0]; }
        const auto& radial_extent() const { return segment_rt_extent[0]; }
        auto& angular_extent() { return segment_rt_extent[1]; }
        const auto& angular_extent() const { return segment_rt_extent[1]; }

        std::array<size_t, 2> radial_shape;
        auto& radial_segments_per_volume() { return radial_shape[0]; }
        const auto& radial_segments_per_volume() const { return radial_shape[0]; }
        auto& radial_records_per_segment() { return radial_shape[1]; }
        const auto& radial_records_per_segment() const { return radial_shape[1]; }

        //std::array<size_t, 2> rectangular_shape;
        //auto& rectangular_segments_per_volume() { return rectangular_shape[0]; }
        //const auto& rectangular_segments_per_volume() const { return rectangular_shape[0]; }
        //auto& rectangular_records_per_segment() { return rectangular_shape[1]; }
        //const auto& rectangular_records_per_segment() const { return rectangular_shape[1]; }

    };

    template<typename config_t_>
    class radial_format_executor_t {
    public:

        using config_t = config_t_;

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            std::swap(_config, config);
        }

#if defined(VORTEX_ENABLE_CUDA)

        template<typename V1, typename V2>
        void format(const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& segment_buffer, size_t segment_index, bool reverse = false) const {
            format(cuda::stream_t::default_(), segment_buffer, volume_buffer, segment_index, reverse);
        }
        template<typename V1, typename V2>
        void format(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& segment_buffer, size_t segment_index, bool reverse = false) const {
            format(stream, volume_buffer, segment_buffer, segment_index, 0, reverse);
        }
        template<typename V1, typename V2>
        void format(const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& segment_buffer, size_t segment_index, size_t record_index, bool reverse = false) const {
            format(cuda::stream_t::default_(), volume_buffer, segment_buffer, segment_index, record_index, reverse);
        }
        template<typename V1, typename V2>
        void format(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer_, const cuda::cuda_viewable<V2>& segment_buffer_, size_t segment_index, size_t record_index, bool reverse = false) const {

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

            // launch the rectification
            _rectify_by_kernel<typename V1::element_t, typename V2::element_t>(stream, volume_buffer, segment_buffer.morph_right(2), segment_index, record_index, reverse);
        }

        template<typename V1, typename V2>
        void execute(const cuda::stream_t& stream, const cuda::cuda_viewable<V1>& volume_buffer, const cuda::cuda_viewable<V2>& block_buffer_, const action::copy& a) const {
            auto& block_buffer = block_buffer_.derived_cast();

            format(stream, volume_buffer, block_buffer.range(a.block_offset, a.block_offset + a.count), a.buffer_segment, a.buffer_record, a.reverse);
        }

#endif

    protected:

        config_t _config;

#if defined(VORTEX_ENABLE_CUDA)

        template<typename T1, typename T2>
        void _rectify_by_kernel(const cuda::stream_t& stream, const cuda::fixed_cuda_view_t<T1, 3>& volume_buffer_, const cuda::fixed_cuda_view_t<const T2, 2>& segment_buffer_, size_t segment_index, size_t record_index, bool reverse) const {
            auto& volume_buffer = volume_buffer_.derived_cast();
            auto& segment_buffer = segment_buffer_.derived_cast();

            // get sample slicing details
            auto slice = copy::slice::to_simple(_config.sample_slice, segment_buffer.shape(1));

            size_t src_begin;

            auto [dst_shape, dst_stride] = volume_buffer.shape_and_stride();
            auto [src_shape, src_stride] = segment_buffer.shape_and_stride();

            // adjust for the slice step
            src_shape[1] = slice.count();
            src_stride[1] *= downcast<ptrdiff_t>(slice.step);

            if (reverse) {
                // offset to start of last record in block for reverse copy
                src_begin = segment_buffer.offset({ segment_buffer.shape(0) - 1, slice.start });
                // stride backwards through block records
                src_stride[0] *= -1;
            } else {
                // offset to start of first record in block for forward copy
                src_begin = segment_buffer.offset({ 0, slice.start });
            }

            // work out pitches and offsets
            std::array<float, 2> segment_rt_pitch = {
                float(_config.radial_extent().length() / _config.radial_records_per_segment()),
                float(_config.angular_extent().length() / _config.radial_segments_per_volume())
            };
            std::array<float, 2> segment_rt_offset = {
                float(_config.radial_extent().min() + segment_rt_pitch[0] * record_index),
                float(_config.angular_extent().min() + segment_rt_pitch[1] * segment_index)
            };
            std::array<float, 2> volume_xy_pitch = {
                float(_config.x_extent().length() / dst_shape[0]),
                float(_config.y_extent().length() / dst_shape[1])
            };
            std::array<float, 2> volume_xy_offset = {
                float(_config.x_extent().min()),
                float(_config.y_extent().min())
            };

            // get sample transform details
            auto transform = copy::transform::to_linear<float>(_config.sample_transform);

            // launch the rectification
            detail::radial_rectify(
                stream, transform.scale, transform.offset,
                { segment_buffer.data() + src_begin, src_shape, src_stride }, segment_rt_pitch, segment_rt_offset,
                { volume_buffer.data(),              dst_shape, dst_stride }, volume_xy_pitch, volume_xy_offset
            );
        }

#endif

    };

}