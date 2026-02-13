#pragma once

#include <vortex/memory/view.hpp>
#include <vortex/memory/tensor.hpp>

#include <fmt/format.h>

namespace vortex {

    template<typename derived_t>
    struct cpu_viewable : public viewable<derived_t> { };
    template<typename T>
    inline constexpr bool is_cpu_viewable = std::is_base_of_v<cpu_viewable<std::decay_t<T>>, std::decay_t<T>>;

    VORTEX_VIEW_AS_IMPL(is_cpu_viewable, cpu);

    namespace detail {
        template<typename T>
        class cpu_view_t;

        template<typename T>
        struct fixed_cpu_view_t {
            template<size_t N>
            class of_dimension :
                public cpu_viewable<of_dimension<N>>,
                public detail::static_view_t<T, N, fixed_cpu_view_t<T>, cpu_view_t<T>>
            {
            public:
                using base_t = detail::static_view_t<T, N, fixed_cpu_view_t<T>, cpu_view_t<T>>;
                using base_t::base_t, typename base_t::element_t;

                template<typename O, typename S1, typename S2>
                of_dimension(const O& other, T* data, const S1& shape, const S2& stride)
                    : of_dimension(data, other.bounds(), shape, stride) {}
                template<typename V>
                of_dimension(const cpu_viewable<V>& other)
                    : base_t(other.derived_cast().data(), other.derived_cast().bounds(), other.derived_cast().shape(), other.derived_cast().stride()) {}

                auto to_xt() {
                    // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                    return xt::adapt(this->data(), this->count(), xt::no_ownership(), this->shape(), this->stride());
                }
                auto to_xt() const {
                    // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                    // XXX: something appears to go wrong in xtensor if a const pointer is adapted
                    return xt::adapt(const_cast<std::remove_const_t<T>*>(this->data()), this->count(), xt::no_ownership(), this->shape(), this->stride());
                }
            };
        };

        template<typename T>
        class cpu_view_t :
            public cpu_viewable<cpu_view_t<T>>,
            public detail::dynamic_view_t<T, fixed_cpu_view_t<T>, cpu_view_t<T>>
        {
        public:
            using base_t = detail::dynamic_view_t<T, fixed_cpu_view_t<T>, cpu_view_t<T>>;
            using base_t::base_t, typename base_t::element_t;

            template<typename O, typename S1, typename S2>
            cpu_view_t(const O& other, T* data, const S1& shape, const S2& stride)
                : cpu_view_t(data, other.bounds(), shape, stride) {}
            template<typename V>
            cpu_view_t(const cpu_viewable<V>& other)
                : base_t(other.derived_cast().data(), other.derived_cast().bounds(), other.derived_cast().shape(), other.derived_cast().stride()) {}

            auto to_xt() {
                // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                return xt::adapt(this->data(), this->count(), xt::no_ownership(), this->shape(), this->stride());
            }
            auto to_xt() const {
                // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                // XXX: something appears to go wrong in xtensor if a const pointer is adapted
                return xt::adapt(const_cast<std::remove_const_t<T>*>(this->data()), this->count(), xt::no_ownership(), this->shape(), this->stride());
            }
        };
    }
    using detail::cpu_view_t;
    template<typename T, size_t N>
    using fixed_cpu_view_t = typename detail::fixed_cpu_view_t<T>::template of_dimension<N>;

    template<typename T>
    class cpu_tensor_t : public detail::tensor_impl_t<T> {
    public:

        using element_t = T;

        cpu_tensor_t()
            : detail::tensor_impl_t<T>() {}
        ~cpu_tensor_t() {
            _allocate(0);
        }

        cpu_tensor_t(cpu_tensor_t&& o) {
            *this = std::move(o);
        }
        cpu_tensor_t& operator=(cpu_tensor_t&& o) {
            detail::tensor_impl_t<T>::operator=(std::move(o));

            std::swap(_buffer, o._buffer);

            return *this;
        }

        size_t underlying_count() const override {
            return _buffer.size();
        }

        using detail::tensor_impl_t<T>::size_in_bytes;

    protected:

        void _allocate(size_t count) override {
            if (count == 0) {
                _buffer.clear();
                _ptr = nullptr;
            } else {
                _buffer.resize(count);
                _ptr = _buffer.data();
            }
        }

        std::vector<T> _buffer;

        using detail::tensor_impl_t<T>::_ptr;

    };

    template<typename T>
    auto view(cpu_tensor_t<T>& obj) {
        return cpu_view_t<T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride());
    }
    template<typename T>
    auto view(const cpu_tensor_t<T>& obj) {
        return cpu_view_t<const T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride());
    }

    template<typename T>
    auto view(xt::xarray<T>& obj) {
        return cpu_view_t<T>(obj.data(), { obj.data(), obj.data() + obj.size() }, obj.shape(), obj.strides());
    }
    template<typename T>
    auto view(const xt::xarray<T>& obj) {
        return cpu_view_t<const T>(obj.data(), { obj.data(), obj.data() + obj.size() }, obj.shape(), obj.strides());
    }
    template<typename T, size_t N>
    auto view(xt::xtensor<T, N>& obj) {
        return fixed_cpu_view_t<T, N>(obj.data(), { obj.data(), obj.data() + obj.size() }, obj.shape(), obj.strides());
    }
    template<typename T, size_t N>
    auto view(const xt::xtensor<T, N>& obj) {
        return fixed_cpu_view_t<const T, N>(obj.data(), { obj.data(), obj.data() + obj.size() }, obj.shape(), obj.strides());
    }

}
