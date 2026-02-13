#pragma once

#include <xtensor/core/xstrides.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xstorage.hpp>

#include <fmt/format.h>

#include <vortex/core.hpp>
#include <vortex/util/variant.hpp>
#include <vortex/util/exception.hpp>

namespace vortex {

    namespace strategy {
        inline struct left_t {} left;
        inline struct right_t {} right;
        inline struct balance_t {} balance;
    }

    class incompatible_shape : public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
    };

    template<typename T>
    struct bounds_t : range_t<T*> {
        operator bounds_t<const T>() const {
            return { min(), max() };
        }

        using range_t<T*>::min, range_t<T*>::max;
    };

    namespace detail {

        template<typename T, typename shape_t, typename stride_t>
        class base_view_t {
        public:

            using element_t = std::decay_t<T>;

        protected:

            base_view_t(T* data, const bounds_t<T>& bounds)
                : _data(data), _bounds(bounds) {}

        public:

            auto valid() const { return data() != nullptr; }

            auto data() const {
                return _data;
            }
            template<typename List>
            auto data(const List& idxs) const {
                return data(idxs.begin(), idxs.end());
            }
            auto data(const std::initializer_list<size_t>& idxs) const {
                return data(idxs.begin(), idxs.end());
            }
            template<typename Iterator>
            auto data(const Iterator& begin, const Iterator& end) const {
                return data() + offset(begin, end);
            }

            template<typename List>
            auto offset(const List& idxs) const {
                return offset(idxs.begin(), idxs.end());
            }
            auto offset(const std::initializer_list<size_t>& idxs) const {
                return offset(idxs.begin(), idxs.end());
            }
            template<typename Iterator>
            auto offset(const Iterator& begin, const Iterator& end) const {
                auto n = std::distance(begin, end);
                if (n > dimension()) {
                    throw traced<std::invalid_argument>(fmt::format("too many indices for shape [{}]: {}", shape_to_string(shape()), n));
                }

                ptrdiff_t o = 0;
                auto it_idx = begin;
                auto it_stride = _stride.begin();

                while(it_idx != end) {
                    o += *it_stride++ * *it_idx++;
                }

                return o;
            }

            auto count() const {
                if (_shape.size() == 0) {
                    return 0ULL;
                } else {
                    return std::accumulate(_shape.begin(), _shape.end(), 1ULL, std::multiplies());
                }
            }
            auto& bounds() const {
                return _bounds;
            }

            auto size_in_bytes() const {
                return count() * sizeof(T);
            }

            const auto& shape() const { return _shape; };
            auto shape(size_t idx) const { return _shape[idx]; }
            const auto& stride() const { return _stride; };
            auto stride(size_t idx) const { return _stride[idx]; }
            auto stride_in_bytes(size_t idx) const { return stride(idx) * sizeof(T); }
            auto stride_in_bytes() const {
                std::vector<size_t> s(dimension());
                for (size_t i = 0; i < dimension(); i++) {
                    s[i] = stride_in_bytes(i);
                }
                return s;
            }
            auto shape_and_stride() const {
                return std::make_tuple(shape(), stride());
            }
            auto stride_with_zeros() const {
                auto swz = _stride;
                for (size_t i = 0; i < dimension(); i++) {
                    if (_stride[i] == 1 && _shape[i] == 1) {
                        swz[i] = 0;
                    }
                }
                return swz;
            }
            auto stride_with_zeros(size_t idx) const {
                return (_stride[idx] == 1 && _shape[idx] == 1) ? 0 : _stride[idx];
            }

            auto dimension() const { return _shape.size(); }

        protected:

            // template<typename other_t>
            // auto _morph(other_t& dst, const strategy::left_t&) const {
            //     auto& src = *this;
            //     auto offset = abs_diff(src.dimension(), dst.dimension());

            //     if (dst.dimension() < src.dimension()) {

            //         // source is larger than destination so ensure left has only trivial dimensions
            //         for (size_t i = 0; i < offset; i++) {
            //              if (src._shape[i] != 1) {
            //                  throw traced<std::invalid_argument>(fmt::format("source has too many non-trivial dimension on left: {} -> {}", shape_to_string(src._shape), shape_to_string(dst._shape)));
            //              }
            //         }

            //         // skip the trivial dimensions when copying
            //         std::copy(src._shape.begin() + offset, src._shape.end(), dst._shape.begin());
            //         std::copy(src._stride.begin() + offset, src._stride.end(), dst._stride.begin());

            //     } else if (dst.dimension() == src.dimension()) {

            //         // copy the actual shape and stride
            //         std::copy(src._shape.begin(), src._shape.end(), dst._shape.begin()());
            //         std::copy(src._stride.begin(), src._stride.end(), dst._stride.begin());

            //     } else {

            //         // pad on the left with trivial dimensions
            //         std::fill_n(dst._shape.begin(), offset, 1);
            //         std::fill_n(dst._stride.begin(), offset, 0);

            //         // copy the actual shape and stride after the new trivial dimensions
            //         std::copy(src._shape.begin(), src._shape.end(), dst._shape.begin() + offset);
            //         std::copy(src._stride.begin(), src._stride.end(), dst._stride.begin() + offset);
            //     }

            //     return dst;
            // }

            T* _data;
            bounds_t<T> _bounds;
            shape_t _shape;
            stride_t _stride;

        };

        template<typename T, size_t N, typename static_t_, typename dynamic_t>
        class static_view_t : public base_view_t<T, std::array<size_t, N>, std::array<ptrdiff_t, N>> {
        protected:

            using base_t = base_view_t<T, std::array<size_t, N>, std::array<ptrdiff_t, N>>;
            template<size_t N2>
            using static_t = typename static_t_::template of_dimension<N2>;

        public:

            static_view_t()
                : base_t(nullptr, { nullptr, nullptr }) {}

            template<typename S1, typename S2>
            static_view_t(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride)
                : base_t(data, bounds) {
                if (shape.size() != N || stride.size() != N) {
                    throw traced<std::invalid_argument>(fmt::format("shape or stride dimension does not match view dimension: {} != {} || {} != {}", shape.size(), N, stride.size(), N));
                }

                std::copy(shape.begin(), shape.end(), _shape.begin());
                std::copy(stride.begin(), stride.end(), _stride.begin());
            }

            // convert from identically-sized dynamic view
            static_view_t(const dynamic_t& other)
                : static_view_t(other.data(), other.bounds(), other.shape(), other.stride()) { }

            template<size_t N2, typename = std::enable_if_t<N2 < N>>
            auto index(std::array<size_t, N2> idx) const {
                return static_t<N - N2>(*static_cast<const static_t<N>*>(this), data(idx), tail<N - N2>(shape()), tail<N - N2>(stride()));
            }

            template<size_t N2, typename = std::enable_if_t<N2 < N>>
            auto index_right(std::array<size_t, N2> idx) const {
                std::array<size_t, N> full_idx = { 0 };
                std::copy(idx.begin(), idx.end(), full_idx.end() - idx.size());
                auto new_data = data(full_idx);

                auto new_shape = head<N - N2>(shape());

                auto new_stride = head<N - N2>(stride());
                // for (size_t i = N - N2; i < N2; i++) {
                //     // NOTE: do not zero-out strides when indexing trivial dimensions
                //     new_stride.back() *= _stride[i] != 0 ? _stride[i] : 1;
                // }

                return static_t<N - N2>(*static_cast<const static_t<N>*>(this), bounds(), new_data, new_shape, new_stride);
            }

            auto range(size_t end) const {
                return range(0, end);
            }
            auto range(size_t start, size_t end) const {
                if (dimension() == 0 || start >= shape(0) || end > shape(0) || start >= end) {
                    throw traced<incompatible_shape>(fmt::format("cannot apply range [{}, {}) to first dimension with shape [{}]", start, end, shape_to_string(shape())));
                }

                auto shape = _shape;
                shape[0] = end - start;

                return static_t<N>(*static_cast<const static_t<N>*>(this), data({ start }), shape, stride());
            }

            template<size_t N2>
            auto morph() const {
                return morph<N2>(strategy::left);
            }
            template<size_t N2, typename Strategy>
            auto morph(const Strategy& strategy) const {
                static_t<N2> dst;
                return _morph(dst, strategy);
            }

            bool is_contiguous() const {
                return equal(_stride, dense_stride(_shape));
            }

            using base_t::bounds, base_t::shape, base_t::stride, base_t::dimension, base_t::data;

        protected:

            using base_t::_data, base_t::_shape, base_t::_stride;

        };

        template<typename T, typename static_t_, typename dynamic_t>
        class dynamic_view_t : public base_view_t<T, xt::svector<size_t>, xt::svector<ptrdiff_t>> {
        protected:

            using base_t = base_view_t<T, xt::svector<size_t>, xt::svector<ptrdiff_t>>;
            template<size_t N2>
            using static_t = typename static_t_::template of_dimension<N2>;

        public:

            dynamic_view_t()
                : base_t(nullptr, { nullptr, nullptr }) {}

            template<typename S1, typename S2>
            dynamic_view_t(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride)
                : base_t(data, bounds) {
                if (shape.size() != stride.size()) {
                    throw traced<incompatible_shape>("shape and stride dimensions do not match");
                }

                _shape.assign(shape.begin(), shape.end());
                _stride.assign(stride.begin(), stride.end());
            }

            // convert from static view to dynamic view
            template<size_t N>
            dynamic_view_t(const static_t<N>& other)
                : dynamic_view_t(other.data(), other.bounds(), other.shape(), other.stride()) { }

            auto index(std::initializer_list<size_t> idx) const {
                if (idx.size() > dimension()) {
                    throw traced<incompatible_shape>(fmt::format("cannot index to ({}) with shape [{}]", shape_to_string(idx, ", "), shape_to_string(shape())));
                }

                auto new_shape = _shape;
                new_shape.erase(new_shape.begin(), new_shape.begin() + idx.size());
                auto new_stride = _stride;
                new_stride.erase(new_stride.begin(), new_stride.begin() + idx.size());

                // allow scalar views
                if (new_shape.size() == 0) {
                    new_shape.push_back(1);
                    new_stride.push_back(0);
                }

                return dynamic_t(*static_cast<const dynamic_t*>(this), data(idx), new_shape, new_stride);
            }

            auto index_right(std::initializer_list<size_t> idx) const {
                if (idx.size() > dimension()) {
                    throw traced<incompatible_shape>(fmt::format("cannot index to ({}) with shape [{}]", shape_to_string(idx, ", "), shape_to_string(shape())));
                }

                std::vector<size_t> full_idx(dimension(), 0);
                std::copy(idx.begin(), idx.end(), full_idx.end() - idx.size());
                auto new_data = data(full_idx);

                auto new_shape = _shape;
                new_shape.erase(new_shape.end() - idx.size(), new_shape.end());

                auto new_stride = _stride;
                new_stride.erase(new_stride.end() - idx.size(), new_stride.end());
                // for (size_t i = new_stride.size(); i < _stride.size(); i++) {
                //     // NOTE: do not zero-out strides when indexing trivial dimensions
                //     new_stride.back() *= _stride[i] != 0 ? _stride[i] : 1;
                // }

                return dynamic_t(*static_cast<const dynamic_t*>(this), new_data, new_shape, new_stride);
            }

            auto range(size_t end) const {
                return range(0, end);
            }
            auto range(size_t start, size_t end) const {
                if (dimension() == 0 || start >= shape(0) || end > shape(0) || start >= end) {
                    throw traced<incompatible_shape>(fmt::format("cannot apply range [{}, {}) to first dimension with shape [{}]", start, end, shape_to_string(shape())));
                }

                auto shape = _shape;
                shape[0] = end - start;

                return dynamic_t(*static_cast<const dynamic_t*>(this), data({ start }), shape, stride());
            }

            auto morph_right(size_t dim) const {
                auto& src = *this;
                auto offset = abs_diff(src.dimension(), dim);

                auto dst_shape = shape();
                dst_shape.resize(dim);
                auto dst_stride = stride();
                dst_stride.resize(dim);

                if (dim < src.dimension()) {

                    // source is larger than destination so ensure right has only trivial dimensions
                    for (size_t i = dimension() - offset; i < dimension(); i++) {
                        if (src.shape(i) != 1) {
                            throw traced<incompatible_shape>(fmt::format("source has too many non-trivial dimensions on right: {} -> {}", shape_to_string(src.shape()), dim));
                        }
                    }

                    // skip the trivial dimensions when copying
                    std::copy(src.shape().begin(), src.shape().end() - offset, dst_shape.begin());
                    std::copy(src.stride().begin(), src.stride().end() - offset, dst_stride.begin());

                } else if (dim == src.dimension()) {

                    // copy the actual shape and stride
                    std::copy(src.shape().begin(), src.shape().end(), dst_shape.begin());
                    std::copy(src.stride().begin(), src.stride().end(), dst_stride.begin());

                } else {

                    // pad on the right with trivial dimensions
                    std::fill_n(dst_shape.end() - offset, offset, 1);
                    std::fill_n(dst_stride.end() - offset, offset, 0);

                    // copy the actual shape and stride after the new trivial dimensions
                    std::copy(src.shape().begin(), src.shape().end(), dst_shape.begin());
                    std::copy(src.stride().begin(), src.stride().end(), dst_stride.begin());
            }

                return dynamic_t(*static_cast<const dynamic_t*>(this), src.data(), dst_shape, dst_stride);
            }

            //void shrink_left(size_t desired_dimension) {
            //    while (dimension() > desired_dimension && _shape.front() == 1) {
            //        _shape.erase(_shape.begin());
            //        _stride.erase(_stride.begin)();
            //    }
            //    if (dimension != desired_dimension) {

            //    }
            //}
            //void shrink_right(size_t desired_dimension) {
            //
            //}
            //void grow_left(size_t desired_dimension) {
            //    while (dimension() < desired_dimension) {
            //        _shape.insert(_shape.begin(), 1);
            //        _stride.insert(_stride.begin(), 1);
            //    }
            //}
            //void grow_right(size_t desired_dimension) {
            //    while (dimension() < desired_dimension) {
            //        _shape.push_back(1);
            //        _stride.push_back(1);
            //    }
            //}

            bool is_contiguous() const {
                return _stride.back() == 1;
            }

            using base_t::bounds, base_t::shape, base_t::stride, base_t::dimension, base_t::data;

        protected:

            using base_t::_data, base_t::_shape, base_t::_stride;

        };

    }

    template<typename derived_t_>
    struct viewable {
        using derived_t = derived_t_;

        auto& derived_cast() {
            return *static_cast<derived_t*>(this);
        }
        const auto& derived_cast() const {
            return *static_cast<const derived_t*>(this);
        }
    };
    template<typename T>
    inline constexpr bool is_viewable = std::is_base_of_v<viewable<std::decay_t<T>>, std::decay_t<T>>;

    class unsupported_view : public std::invalid_argument {
    public:
        using invalid_argument::invalid_argument;
    };

#define VORTEX_VIEW_AS_IMPL(view_check, suffix) \
    template<typename F, typename... O, typename = typename std::enable_if_t<(sizeof...(O) >= 1)>> \
    void view_as_##suffix(F&& func, O&... obj) { \
        std::invoke([&](auto... buffers) { \
            if constexpr ((... && view_check<decltype(buffers)>)) { \
                std::invoke(func, std::move(buffers)...); \
            } else { \
                std::vector<std::string> type_names = { typeid(decltype(buffers)).name() ... }; \
                throw traced<unsupported_view>(fmt::format("({}) do not support " #suffix, join(type_names, ", "))); \
            } \
        }, std::move(view(obj))...); \
    } \
    template<typename F, typename... O> \
    void view_tuple_as_##suffix(F&& func, const std::tuple<O...>& tup) { \
        std::apply([&](const auto&... objs) { \
            view_as_##suffix([&](auto... buffers) { \
                std::invoke(func, std::move(std::make_tuple(buffers...))); \
            }, objs...); \
        }, tup); \
    }

}
