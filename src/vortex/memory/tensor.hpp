#pragma once

#include <algorithm>

#include <xtensor/containers/xstorage.hpp>

#include <fmt/format.h>

namespace vortex {

    namespace detail {

        template<typename T>
        class tensor_impl_t {
        public:

            tensor_impl_t() {};
            virtual ~tensor_impl_t() {};

            tensor_impl_t(const tensor_impl_t&) = delete;
            tensor_impl_t& operator=(const tensor_impl_t&) = delete;

            tensor_impl_t(tensor_impl_t&& o) {
                *this = std::move(o);
            }
            tensor_impl_t& operator=(tensor_impl_t&& o) {
                clear();

                std::swap(_ptr, o._ptr);
                std::swap(_shape, o._shape);
                std::swap(_stride, o._stride);

                return *this;
            }

            template<typename shape_t>
            void resize(const shape_t& shape, bool shrink = true) {
                _resize(shape, shrink);
            }
            template<typename shape_t, typename stride_t>
            void resize(const shape_t& shape, const stride_t& stride, bool shrink = true) {
                _resize(shape, stride, shrink);
            }
            template<typename U>
            void resize(std::initializer_list<U>&& shape, bool shrink = true) {
                _resize(std::forward<std::initializer_list<U>>(shape), shrink);
            }
            template<typename U, typename V>
            void resize(std::initializer_list<U>&& shape, std::initializer_list<V>&& stride, bool shrink = true) {
                _resize(std::forward<std::initializer_list<U>>(shape), std::forward<std::initializer_list<V>>(stride), shrink);
            }

            void shrink() {
                _resize(_shape, true);
            }

            void clear() {
                resize(std::array<size_t, 0>{}, true);
            }

            auto data() {
                return _ptr;
            }
            const auto data() const {
                return _ptr;
            }

            auto count() const {
                if (_shape.empty()) {
                    return 0ULL;
                } else {
                    return std::accumulate(_shape.begin(), _shape.end(), 1ULL, std::multiplies());
                }
            }
            virtual size_t underlying_count() const = 0;

            auto size_in_bytes() const {
                return count() * sizeof(T);
            }
            auto underlying_size_in_bytes() const {
                return underlying_count() * sizeof(T);
            }

            auto dimension() const {
                return _shape.size();
            }

            const auto& shape() const {
                return _shape;
            }
            auto shape(size_t idx) const {
                return _shape[idx];
            }

            const auto& stride() const {
                return _stride;
            }
            auto stride(size_t idx) const {
                return _stride[idx];
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

            auto stride_in_bytes(size_t idx) const {
                return stride(idx) * sizeof(T);
            }
            auto stride_in_bytes() const {
                std::vector<size_t> s(dimension());
                for (size_t i = 0; i < dimension(); i++) {
                    s[i] = stride_in_bytes(i);
                }
                return s;
            }

            auto valid() const {
                return count() > 0;
            }

        protected:

            virtual void _allocate(size_t n) = 0;

            template<typename shape_t>
            void _resize(const shape_t& shape, bool shrink) {
                _shape.assign(shape.begin(), shape.end());
                _stride.resize(_shape.size());

                // NOTE: do not use xt::compute_strides(...) because it sets stride to 0 for singleton dimensions
                auto n = dense_stride(_shape, _stride);
                if (n > underlying_count() || shrink) {
                    _allocate(_shape.empty() ? 0 : n);
                }
            }

            template<typename shape_t, typename stride_t>
            void _resize(const shape_t& shape, const stride_t& stride, bool shrink) {
                if (shape.size() != stride.size()) {
                    throw traced<std::runtime_error>("shape and stride dimensions do not match");
                }

                _shape.assign(shape.begin(), shape.end());
                _stride.assign(stride.begin(), stride.end());

                auto n = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<>());
                if (n > underlying_count() || shrink) {
                    _allocate(_shape.empty() ? 0 : n);
                }
            }

            T* _ptr;

            xt::svector<size_t> _shape;
            xt::svector<ptrdiff_t> _stride;

        };
    }

    template<typename T>
    inline constexpr bool is_tensor = std::is_base_of_v<detail::tensor_impl_t<typename std::decay_t<T>::element_t>, std::decay_t<T>>;

}
