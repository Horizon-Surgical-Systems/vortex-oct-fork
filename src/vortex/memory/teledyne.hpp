#pragma once

#include <vortex/memory/tensor.hpp>
#include <vortex/memory/cuda.hpp>

namespace vortex::teledyne {

    template<typename derived_t>
    struct teledyne_cpu_viewable : public cpu_viewable<derived_t> { };
    template<typename T>
    inline constexpr bool is_teledyne_cpu_viewable = std::is_base_of_v<teledyne_cpu_viewable<std::decay_t<T>>, std::decay_t<T>>;

    namespace detail {
        using vortex::detail::tensor_impl_t;

        template<typename T>
        class teledyne_cpu_view_t;

        template<typename T>
        struct fixed_teledyne_cpu_view_t {
            template<size_t N>
            class of_dimension :
                public teledyne_cpu_viewable<of_dimension<N>>,
                public vortex::detail::static_view_t<T, N, fixed_teledyne_cpu_view_t<T>, teledyne_cpu_view_t<T>>
            {
            public:
                using base_t = vortex::detail::static_view_t<T, N, fixed_teledyne_cpu_view_t<T>, teledyne_cpu_view_t<T>>;
                using typename base_t::element_t;

                of_dimension()
                    : base_t(), _buffer_index(-1) {}
                template<typename S1, typename S2>
                of_dimension(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, size_t buffer_index)
                    : base_t(data, bounds, shape, stride), _buffer_index(buffer_index) {}
                template<typename O, typename S1, typename S2>
                of_dimension(const O& other, T* data, const S1& shape, const S2& stride)
                    : of_dimension(data, other.bounds(), shape, stride, other.buffer_index()) {}

                template<typename V>
                of_dimension(const teledyne_cpu_viewable<V>& other)
                    : of_dimension(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

                void bind(T* ptr, size_t buffer_index) {
                    _data = ptr;
                    _bounds = { ptr, ptr + count() };
                    _buffer_index = buffer_index;
                }
                void unbind() {
                    _data = nullptr;
                    _bounds = { nullptr, nullptr };
                    _buffer_index = -1;
                }

                const auto& buffer_index() const { return _buffer_index; }

                auto to_xt() {
                    // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                    return xt::adapt(this->data(), this->count(), xt::no_ownership(), this->shape(), this->stride());
                }
                auto to_xt() const {
                    // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                    // XXX: something appears to go wrong in xtensor if a const pointer is adapted
                    return xt::adapt(const_cast<std::remove_const_t<T>*>(this->data()), this->count(), xt::no_ownership(), this->shape(), this->stride());
                }

                using base_t::count;

            protected:

                size_t _buffer_index;

                using base_t::_data, base_t::_bounds;

            };
        };

        template<typename T>
        class teledyne_cpu_view_t :
            public teledyne_cpu_viewable<teledyne_cpu_view_t<T>>,
            public vortex::detail::dynamic_view_t<T, fixed_teledyne_cpu_view_t<T>, teledyne_cpu_view_t<T>>
        {
        public:
            using base_t = vortex::detail::dynamic_view_t<T, fixed_teledyne_cpu_view_t<T>, teledyne_cpu_view_t<T>>;
            using typename base_t::element_t;

            teledyne_cpu_view_t()
                : base_t(), _buffer_index(-1) {}

            template<typename S1, typename S2>
            teledyne_cpu_view_t(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, size_t buffer_index)
                : base_t(data, bounds, shape, stride), _buffer_index(buffer_index) {}
            template<typename O, typename S1, typename S2>
            teledyne_cpu_view_t(const O& other, T* data, const S1& shape, const S2& stride)
                : teledyne_cpu_view_t(data, other.bounds(), shape, stride, other.buffer_index()) {}

            template<typename V>
            teledyne_cpu_view_t(const teledyne_cpu_viewable<V>& other)
                : teledyne_cpu_view_t(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

            void bind(T* ptr, size_t buffer_index) {
                _data = ptr;
                _bounds = { ptr, ptr + count() };
                _buffer_index = buffer_index;
            }
            void unbind() {
                _data = nullptr;
                _bounds = { nullptr, nullptr };
                _buffer_index = -1;
            }

            const auto& buffer_index() const { return _buffer_index; }

            auto to_xt() {
                // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                return xt::adapt(this->data(), this->count(), xt::no_ownership(), this->shape(), this->stride());
            }
            auto to_xt() const {
                // NOTE: provide a size that matches this shape, regardless of the underlying data size, to prevent xtensor for attempting a resize
                // XXX: something appears to go wrong in xtensor if a const pointer is adapted
                return xt::adapt(const_cast<std::remove_const_t<T>*>(this->data()), this->count(), xt::no_ownership(), this->shape(), this->stride());
            }

            using base_t::count;

        protected:

            size_t _buffer_index;

            using base_t::_data, base_t::_bounds;

        };
    }
    using detail::teledyne_cpu_view_t;
    template<typename T, size_t N>
    using fixed_teledyne_cpu_view_t = typename detail::fixed_teledyne_cpu_view_t<T>::template of_dimension<N>;

    VORTEX_VIEW_AS_IMPL(is_teledyne_cpu_viewable, teledyne_cpu);

    template<typename T>
    auto& view(teledyne_cpu_view_t<T>& obj) {
        return obj;
    }
    template<typename T>
    const auto& view(const teledyne_cpu_view_t<T>& obj) {
        return obj;
    }

    template<typename derived_t>
    class teledyne_cuda_viewable : public cuda::cuda_viewable<derived_t> {};
    template<typename T>
    inline constexpr bool is_teledyne_cuda_viewable = std::is_base_of_v<teledyne_cuda_viewable<std::decay_t<T>>, std::decay_t<T>>;

    namespace detail {
        using vortex::detail::tensor_impl_t;

        template<typename T>
        class teledyne_cuda_view_t;

        template<typename T>
        struct fixed_teledyne_cuda_view_t {
            template<size_t N>
            class of_dimension :
                public teledyne_cuda_viewable<of_dimension<N>>,
                public vortex::detail::static_view_t<T, N, fixed_teledyne_cuda_view_t<T>, teledyne_cuda_view_t<T>>
            {
            public:
                using base_t = vortex::detail::static_view_t<T, N, fixed_teledyne_cuda_view_t<T>, teledyne_cuda_view_t<T>>;
                using typename base_t::element_t;

                of_dimension()
                    : base_t(), _device(cuda::invalid_device), _buffer_index(0) {}
                template<typename S1, typename S2>
                of_dimension(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, const cuda::device_t& device, const size_t& buffer_index)
                    : base_t(data, bounds, shape, stride), _device(device), _buffer_index(buffer_index) {}
                template<typename O, typename S1, typename S2>
                of_dimension(const O& other, T* data, const S1& shape, const S2& stride)
                    : of_dimension(data, other.bounds(), shape, stride, other.device(), other.buffer_index()) {}

                template<typename V>
                of_dimension(const teledyne_cuda_viewable<V>& other)
                    : of_dimension(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

                const auto& device() const { return _device; }
                const auto& buffer_index() const { return _buffer_index; }

            protected:

                cuda::device_t _device;
                size_t _buffer_index;
            };
        };

        template<typename T>
        class teledyne_cuda_view_t :
            public teledyne_cuda_viewable<teledyne_cuda_view_t<T>>,
            public vortex::detail::dynamic_view_t<T, fixed_teledyne_cuda_view_t<T>, teledyne_cuda_view_t<T>>
        {
        public:
            using base_t = vortex::detail::dynamic_view_t<T, fixed_teledyne_cuda_view_t<T>, teledyne_cuda_view_t<T>>;
            using typename base_t::element_t;

            teledyne_cuda_view_t()
                : base_t(), _device(cuda::invalid_device), _buffer_index(0) {}

            template<typename S1, typename S2>
            teledyne_cuda_view_t(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, const cuda::device_t& device, const size_t& buffer_index)
                : base_t(data, bounds, shape, stride), _device(device), _buffer_index(buffer_index) {}
            template<typename O, typename S1, typename S2>
            teledyne_cuda_view_t(const O& other, T* data, const S1& shape, const S2& stride)
                : teledyne_cuda_view_t(data, other.bounds(), shape, stride, other.device(), other.buffer_index()) {}

            template<typename V>
            teledyne_cuda_view_t(const teledyne_cuda_viewable<V>& other)
                : teledyne_cuda_view_t(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

            const auto& device() const { return _device; }
            const auto& buffer_index() const { return _buffer_index; }

            bool is_accessible() const {
                return is_accessible(cuda::device());
            }
            bool is_accessible(cuda::device_t device2) const {
                if (device() == device2) {
                    return true;
                } else {
                    return cuda::peer_access(device(), device2);
                }
            }

        protected:

            cuda::device_t _device;
            size_t _buffer_index;
        };
    }
    using detail::teledyne_cuda_view_t;
    template<typename T, size_t N>
    using fixed_teledyne_cuda_view_t = typename detail::fixed_teledyne_cuda_view_t<T>::template of_dimension<N>;

    VORTEX_VIEW_AS_IMPL(is_teledyne_cuda_viewable, teledyne_cuda);

    template<typename T>
    auto& view(teledyne_cuda_view_t<T>& obj) {
        return obj;
    }
    template<typename T>
    const auto& view(const teledyne_cuda_view_t<T>& obj) {
        return obj;
    }

}
