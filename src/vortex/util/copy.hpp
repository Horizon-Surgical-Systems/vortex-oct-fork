#pragma once

#include <vortex/memory/view.hpp>
#include <vortex/memory/cpu.hpp>

// XXX: reformat to not require this header
#include <vortex/format/common.hpp>

#include <vortex/util/cast.hpp>

namespace vortex::copy {

    namespace slice {
        struct none_t {};
        struct simple_t {
            size_t start, stop, step;
            simple_t() {}
            simple_t(size_t stop) : simple_t(0, stop) {}
            simple_t(size_t start, size_t stop) : simple_t(start, stop, 1) {}
            simple_t(size_t start_, size_t stop_, size_t step_) { start = start_; stop = stop_; step = step_; }
            size_t count() const { return (stop - start) / step; }
        };
    }
    using slice_t = std::variant<copy::slice::none_t, copy::slice::simple_t>;
    namespace slice {
        inline auto to_simple(slice_t slice, size_t max) {
            return std::visit(overloaded{
                [&](const none_t& s) { return simple_t(max); },
                [&](const simple_t& s) { return s; }
            }, slice);
        }
    }

    namespace transform {
        struct none_t {};
        template<typename T>
        struct linear_t {
            T scale, offset;
            linear_t() : linear_t(1, 0) {}
            linear_t(T scale_, T offset_) : scale(scale_), offset(offset_) {}
        };
    }
    using transform_t = std::variant<transform::none_t, transform::linear_t<double>>;
    namespace transform {
        template<typename T>
        inline auto to_linear(transform_t transform) {
            return std::visit(overloaded{
                [&](const none_t& s) { return linear_t<T>(); },
                [&](const linear_t<double>& s) { return linear_t<T>(T(s.scale), T(s.offset)); }
            }, transform);
        }
    }

    struct options_block2volume_t {
        size_t count = 0;
        size_t src_offset = 0;
        std::array<size_t, 2> dst_offset = { {0,0} };

        bool reverse = false;

        slice_t sample_slice = slice::none_t{};
        transform_t sample_transform = transform::none_t{};
    };

    struct options_block2block_t {
        size_t count = 0;

        size_t src_offset = 0, dst_offset = 0;

        slice_t sample_slice = slice::none_t{};
        transform_t sample_transform = transform::none_t{};
    };

    namespace detail {

        template<typename E1, typename E2>
        void copy_by_xtensor(const xt::xexpression<E1>& viewed_src_, xt::xexpression<E2>& viewed_dst_, const options_block2volume_t& o) {
            auto& viewed_src = viewed_src_.derived_cast();
            auto& viewed_dst = viewed_dst_.derived_cast();

            // copy helper
            auto do_copy = [&](auto&& expr) {
                if (o.reverse) {
                    // NOTE: flip the destination to avoid compile issues with the input intermediate expression not having a data offset
                    xt::flip(viewed_dst, 0) = expr;
                } else {
                    viewed_dst = expr;
                }
            };

            // transform helper
            auto do_transform = [&](auto&& expr) {
                std::visit(overloaded{
                    [&](const transform::none_t& t) { do_copy(expr); },
                    [&](const transform::linear_t<double>& t) { do_copy(xt::cast<typename E2::value_type>(t.scale * expr + t.offset)); }
                    }, o.sample_transform);
            };

            // slice helper
            auto do_slice = [&](auto&& expr) {
                std::visit(overloaded{
                    [&](const slice::none_t& s) { do_transform(expr); },
                    [&](const slice::simple_t& s) { do_transform(xt::strided_view(expr, { xt::all(), xt::range(s.start, s.stop, s.step), xt::ellipsis() })); }
                    }, o.sample_slice);
            };

            // execute the processing
            do_slice(viewed_src);
        }

        template<typename E1, typename E2>
        void copy_by_xtensor(const xt::xexpression<E1>& viewed_src_, xt::xexpression<E2>& viewed_dst_, const options_block2block_t& o) {
            auto& viewed_src = viewed_src_.derived_cast();
            auto& viewed_dst = viewed_dst_.derived_cast();

            // copy helper
            auto do_copy = [&](auto&& expr) {
                viewed_dst = expr;
            };

            // transform helper
            auto do_transform = [&](auto&& expr) {
                std::visit(overloaded{
                    [&](const transform::none_t& t) { do_copy(expr); },
                    [&](const transform::linear_t<double>& t) { do_copy(xt::cast<typename E2::value_type>(t.scale * expr + t.offset)); }
                }, o.sample_transform);
            };

            // slice helper
            auto do_slice = [&](auto&& expr) {
                std::visit(overloaded{
                    [&](const slice::none_t& s) { do_transform(expr); },
                    [&](const slice::simple_t& s) { do_transform(xt::strided_view(expr, { xt::all(), xt::range(s.start, s.stop, s.step), xt::ellipsis() })); }
                }, o.sample_slice);
            };

            // execute the processing
            do_slice(viewed_src);
        }

    }

    template<typename V1, typename V2>
    void copy(const cpu_viewable<V1>& src_, const cpu_viewable<V2>& dst_, const options_block2volume_t& options) {
        auto src = src_.derived_cast().to_xt();
        auto dst = dst_.derived_cast().to_xt();

        // perform the copy
        auto viewed_src = xt::strided_view(xt::atleast_Nd<3>(src), { xt::range(options.src_offset, options.src_offset + options.count), xt::ellipsis() });
        // NOTE: static_cast required since xtensor uses signed integers for slice indices
        auto viewed_dst = xt::strided_view(xt::atleast_Nd<4>(dst), { static_cast<long long>(options.dst_offset[0]), xt::range(options.dst_offset[1], options.dst_offset[1] + options.count), xt::ellipsis() });
        detail::copy_by_xtensor(viewed_src, viewed_dst, options);
    }

    template<typename V1, typename V2>
    void copy(const cpu_viewable<V1>& src_, const cpu_viewable<V2>& dst_, const options_block2block_t& options) {
        auto src = src_.derived_cast().to_xt();
        auto dst = dst_.derived_cast().to_xt();

        // perform the copy
        auto viewed_src = xt::strided_view(src, { xt::range(options.src_offset, options.src_offset + options.count), xt::ellipsis() });
        auto viewed_dst = xt::strided_view(dst, { xt::range(options.dst_offset, options.dst_offset + options.count), xt::ellipsis() });
        detail::copy_by_xtensor(viewed_src, viewed_dst, options);
    }

}

#if defined(VORTEX_ENABLE_CUDA)

#include <vortex/driver/cuda/copy.hpp>
#include <vortex/memory/cuda.hpp>

namespace vortex::copy {

    namespace detail {

        template<typename V1, typename V2>
        void copy_by_memcpy(const cuda::stream_t& stream, const viewable<V2>& src_, const viewable<V1>& dst_, const options_block2volume_t& o) {
            if (!std::holds_alternative<transform::none_t>(o.sample_transform)) {
                throw std::invalid_argument("transformation not supported with memcpy-based formatting");
            }

            auto& src = src_.derived_cast();
            auto& dst = dst_.derived_cast();

            auto [src_shape, src_stride] = format::detail::trim_shape_and_stride(src, 3);
            auto [dst_shape, dst_stride] = format::detail::trim_shape_and_stride(dst, 4);

            // get sample slicing details
            bool slicing;
            slice::simple_t slice = std::visit(overloaded{
                [&](const slice::none_t& s) { slicing = false; return slice::simple_t(src.shape(1)); }, // retain all samples
                [&](const slice::simple_t& s) { slicing = true; return s; }
            }, o.sample_slice);

            if (!o.reverse && !slicing) {
                auto src_begin = strided_offset(src_stride, o.src_offset);
                auto src_end = strided_offset(src_stride, o.src_offset + o.count);
                auto dst_begin = strided_offset(dst_stride, o.dst_offset[0], o.dst_offset[1]);

                // forward copy entire segment
                cuda::copy(src, src_begin, dst, dst_begin, src_end - src_begin, &stream);
            } else if (!o.reverse && slice.step == 1) {
                auto src_begin = strided_offset(src_stride, o.src_offset, slice.start);
                auto dst_begin = strided_offset(dst_stride, o.dst_offset[0], o.dst_offset[1]);

                // forward copy all records with stride
                cuda::copy(
                    src, src_begin, src_stride[0],         // no stepping over samples
                    dst, dst_begin, dst_stride[1],         // contiguous samples in the volume
                    o.count, slice.count() * src_shape[2], // treat [slice.start, slice.end} as a single row
                    &stream
                );
            } else {
                // dispatch multiple copies since CUDA memcpy strides cannot be negative
                for (size_t dst_record = 0; dst_record < o.count; dst_record++) {
                    auto src_record = o.reverse ? o.count - dst_record - 1 : dst_record;

                    auto src_begin = strided_offset(src_stride, o.src_offset + src_record, slice.start);
                    auto src_end = strided_offset(src_stride, o.src_offset + src_record, slice.stop);
                    auto dst_begin = strided_offset(dst_stride, o.dst_offset[0], o.dst_offset[1] + dst_record);

                    if (slice.step == 1) {
                        // forward copy single record
                        cuda::copy(src, src_begin, dst, dst_begin, src_end - src_begin, &stream);
                    } else {
                        // forward copy single record with stride
                        cuda::copy(
                            src, src_begin, src_stride[1] * slice.step, // use the stride to step over samples
                            dst, dst_begin, dst_stride[2],              // contiguous samples in the volume
                            slice.count(), src_shape[2],                // each sample is a "row" so the pitch is the offset between them during copying
                            &stream
                        );
                    }
                }
            }
        }

#define _DECLARE(src_type, dst_type) void linear_transform(const cuda::stream_t& stream, float scale, float offset, const std::array<size_t, 3> shape, const src_type* src, size_t src_offset, const std::array<ptrdiff_t, 3>& src_strides, dst_type* dst, size_t dst_offset, const std::array<ptrdiff_t, 3>& dst_strides)

        _DECLARE(int8_t, int8_t);
        _DECLARE(uint8_t, uint8_t);
        _DECLARE(uint16_t, uint16_t);
        _DECLARE(float, int8_t);
        _DECLARE(float, float);
        _DECLARE(uint16_t, float);
        _DECLARE(int8_t, float);

#undef _DECLARE

        template<typename V1, typename V2>
        void copy_by_kernel(const cuda::stream_t& stream, const cuda::cuda_viewable<V2>& src_, const cuda::cuda_viewable<V1>& dst_, const options_block2volume_t& o) {
            auto& dst = dst_.derived_cast();
            auto& src = src_.derived_cast();

            // get sample slicing details
            auto slice = slice::to_simple(o.sample_slice, src.shape(1));

            auto dst_begin = dst.offset({ o.dst_offset[0], o.dst_offset[1] });
            size_t src_begin;

            // XXX: refactor to not require the format namespace
            auto [dst_shape, dst_full_stride] = format::detail::trim_shape_and_stride(dst, 4);
            auto [src_shape, src_full_stride] = format::detail::trim_shape_and_stride(src, 3);

            auto dst_stride = tail<3>(dst_full_stride);
            auto src_stride = tail<3>(src_full_stride);
            // adjust for the slice step
            src_stride[1] *= downcast<ptrdiff_t>(slice.step);

            if (o.reverse) {
                // offset to start of last record in block for reverse copy
                src_begin = src.offset({ o.src_offset + o.count - 1, slice.start });
                // stride backwards through block records
                src_stride[0] *= -1;
            } else {
                // offset to start of first record in block for forward copy
                src_begin = src.offset({ o.src_offset, slice.start });
            }

            auto transform = transform::to_linear<float>(o.sample_transform);

            // launch the copy
            linear_transform(
                stream, transform.scale, transform.offset,
                { o.count, slice.count(), src_shape[2] },
                src.data(), src_begin, src_stride,
                dst.data(), dst_begin, dst_stride
            );
        }

    }

    template<typename V1, typename V2, typename options_t>
    void copy(const cuda::stream_t& stream, const cpu_viewable<V2>& src, const cuda::cuda_viewable<V1>& dst, const options_t& options) {
        detail::copy_by_memcpy(stream, dst, src, options);
    }
    template<typename V1, typename V2, typename options_t>
    void copy(const cuda::stream_t& stream, const cuda::cuda_viewable<V2>& src, const cpu_viewable<V1>& dst, const options_t& options) {
        detail::copy_by_memcpy(stream, dst, src, options);
    }
    template<typename V1, typename V2, typename options_t>
    void copy(const cuda::stream_t& stream, const cuda::cuda_viewable<V2>& src_, const cuda::cuda_viewable<V1>& dst_, const options_t& options) {
        auto& src = src_.derived_cast();
        auto& dst = dst_.derived_cast();

        // a single device-to-device memcpy is faster than a copy kernel
        if ((options.reverse || !stride_is_compatible(src.stride_with_zeros(), dst.stride_with_zeros()) || !std::holds_alternative<transform::none_t>(options.sample_transform)) && dst.is_accessible() && src.is_accessible()) {
            detail::copy_by_kernel(stream, src, dst, options);
        } else {
            detail::copy_by_memcpy(stream, src, dst, options);
        }
    }

}

#endif
