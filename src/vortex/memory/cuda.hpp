#pragma once

#include <fmt/format.h>

#include <vortex/memory/tensor.hpp>
#include <vortex/memory/view.hpp>
#include <vortex/memory/cpu.hpp>

#include <vortex/driver/cuda/runtime.hpp>

#include <vortex/util/exception.hpp>

namespace vortex::cuda {

    template<typename derived_t>
    class cuda_viewable : public viewable<derived_t> {};
    template<typename T>
    inline constexpr bool is_cuda_viewable = std::is_base_of_v<cuda_viewable<std::decay_t<T>>, std::decay_t<T>>;

    namespace detail {
        using vortex::detail::tensor_impl_t;

        template<typename T>
        class cuda_view_t;

        template<typename T>
        struct fixed_cuda_view_t {
            template<size_t N>
            class of_dimension :
                public cuda_viewable<of_dimension<N>>,
                public vortex::detail::static_view_t<T, N, fixed_cuda_view_t<T>, cuda_view_t<T>>
            {
            public:
                using base_t = vortex::detail::static_view_t<T, N, fixed_cuda_view_t<T>, cuda_view_t<T>>;
                using typename base_t::element_t;

                of_dimension()
                    :base_t(), _device(cuda::invalid_device) {}
                template<typename S1, typename S2>
                of_dimension(T* data, const bounds_t<T> bounds, const S1& shape, const S2& stride, const cuda::device_t& device)
                    : base_t(data, bounds, shape, stride), _device(device) {}
                template<typename O, typename S1, typename S2>
                of_dimension(const O& other, T* data, const S1& shape, const S2& stride)
                    : of_dimension(data, other.bounds(), shape, stride, other.device()) {}

                template<typename V>
                of_dimension(const cuda_viewable<V>& other)
                    : of_dimension(other.derived_cast(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

                const auto& device() const { return _device; }

                template<typename U = T, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
                auto to_strided() const {
                    return cuda::strided_t<U, N>(this->data(), this->shape(), this->stride());
                }
                template<typename U, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
                operator const cuda::strided_t<U, N>() const { return this->to_strided<U>(); }

            protected:
                cuda::device_t _device;
            };
        };

        template<typename T>
        class cuda_view_t :
            public cuda_viewable<cuda_view_t<T>>,
            public vortex::detail::dynamic_view_t<T, fixed_cuda_view_t<T>, cuda_view_t<T>>
        {
        public:
            using base_t = vortex::detail::dynamic_view_t<T, fixed_cuda_view_t<T>, cuda_view_t<T>>;
            using typename base_t::element_t;

            cuda_view_t()
                : base_t(), _device(cuda::invalid_device) {}
            template<typename S1, typename S2>
            cuda_view_t(T* data, const bounds_t<T> bounds, const S1& shape, const S2& stride, const cuda::device_t& device)
                : base_t(data, bounds, shape, stride), _device(device) {}
            template<typename O, typename S1, typename S2>
            cuda_view_t(const O& other, T* data, const S1& shape, const S2& stride)
                : cuda_view_t(data, other.bounds(), shape, stride, other.device()) {}

            template<typename V>
            cuda_view_t(const cuda_viewable<V>& other)
                : cuda_view_t(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

            const auto& device() const { return _device; }

            bool is_accessible() const {
                return is_accessible(cuda::device());
            }
            bool is_accessible(device_t device2) const {
                if (device() == device2) {
                    return true;
                } else {
                    int accessible;
                    auto error = cudaDeviceCanAccessPeer(&accessible, device(), device2);
                    detail::handle_error(error, "could not determine peer accessibility {} <-> {}", device(), device2);
                    return accessible != 0;
                }
            }

            template<size_t N, typename U = T, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
            auto to_strided() const {
                if (this->dimension() != N) {
                    throw traced<std::runtime_error>(fmt::format("dimension mismatch in strided creation: {} != {}", this->dimension(), N));
                }

                return cuda::strided_t<U, N>(this->data(), to_array<N>(this->shape()), to_array<N>(this->stride()));
            }
            template<size_t N, typename U, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
            operator const cuda::strided_t<U, N>() const { return this->to_strided<N, U>(); }


        protected:

            cuda::device_t _device;

        };
    }
    using detail::cuda_view_t;
    template<typename T, size_t N>
    using fixed_cuda_view_t = typename detail::fixed_cuda_view_t<T>::template of_dimension<N>;

    VORTEX_VIEW_AS_IMPL(is_cuda_viewable, cuda);

    template<typename T>
    class cuda_device_tensor_t : public detail::tensor_impl_t<T> {
    public:

        using element_t = T;

        cuda_device_tensor_t()
            : detail::tensor_impl_t<T>() {
            _reset();
        }
        ~cuda_device_tensor_t() {
            _release();
        }

        cuda_device_tensor_t(cuda_device_tensor_t&& o) {
            *this = std::move(o);
        }
        cuda_device_tensor_t& operator=(cuda_device_tensor_t&& o) {
            detail::tensor_impl_t<T>::operator=(std::move(o));

            std::swap(_device, o._device);
            std::swap(_count, o._count);

            return *this;
        }

        bool is_accessible() const {
            return is_accessible(cuda::device());
        }
        bool is_accessible(device_t device2) const {
            if (device() == device2) {
                return true;
            } else {
                return cuda::peer_access(device(), device2);
            }
        }

        const auto& device() const {
            return _device;
        }

        size_t underlying_count() const override {
            return _count;
        }

        template<size_t N, typename U = T, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
        auto to_strided() const {
            if (this->dimension() != N) {
                throw traced<std::runtime_error>(fmt::format("dimension mismatch in strided creation: {} != {}", this->dimension(), N));
            }

            return cuda::strided_t<U, N>(this->data(), to_array<N>(this->shape()), to_array<N>(this->stride()));
        }
        template<size_t N, typename U, typename = std::enable_if_t<std::is_same_v<T, U> || std::is_same_v<std::add_const_t<T>, U>>>
        operator const cuda::strided_t<U, N>() const { return this->to_strided<N, U>(); }

        using detail::tensor_impl_t<T>::size_in_bytes;

    protected:

        void _allocate(size_t count) override {
            // elide equivalent size allocations
            if (count == _count) {
                return;
            }

            // do not allow allocate to change the device of currently allocated memory
            auto allocate_device = _device;
            if (allocate_device < 0) {
                // use the current device if this is a fresh allocation
                allocate_device = cuda::device();
            }

            _release();

            // check for empty allocation
            if (count == 0) {
                return;
            }

            _count = count;
            _device = allocate_device;

            // switch devices for the allocation
            auto original_device = cuda::device(allocate_device);
            auto error = cudaMalloc(reinterpret_cast<void**>(&_ptr), size_in_bytes());
            cuda::device(original_device);

            if (error) {
                // clear state before raising the error
                auto size = _count;
                _release();
                detail::handle_error(error, "unable to allocate device memory of {} bytes on device {}", size, allocate_device);
            }
        }

        void _release() {
            if (_ptr) {
                auto error = cudaFree(_ptr);
                detail::handle_error(error, "unable to free device memory");
            }

            _reset();
        }

        void _reset() {
            _ptr = nullptr;
            _count = 0;
            _device = invalid_device;
        }

        size_t _count;
        device_t _device;

        using detail::tensor_impl_t<T>::_ptr;

    };

    template<typename T>
    class cuda_host_tensor_t : public detail::tensor_impl_t<T> {
    public:

        using element_t = T;

        cuda_host_tensor_t()
            : detail::tensor_impl_t<T>() {
            _reset();
        }
        ~cuda_host_tensor_t() {
            _release();
        }

        cuda_host_tensor_t(cuda_host_tensor_t&& o) {
            *this = std::move(o);
        }
        cuda_host_tensor_t& operator=(cuda_host_tensor_t&& o) {
            detail::tensor_impl_t<T>::operator=(std::move(o));

            std::swap(_count, o._count);

            return *this;
        }

        size_t underlying_count() const override {
            return _count;
        }

        using detail::tensor_impl_t<T>::size_in_bytes;

    protected:

        void _allocate(size_t count) override {
            // elide equivalent size allocations
            if (count == _count) {
                return;
            }

            _release();

            // check for empty allocation
            if (count == 0) {
                return;
            }

            _count = count;

            auto error = cudaMallocHost(reinterpret_cast<void**>(&_ptr), size_in_bytes());
            if (error) {
                // clear state before raising the error
                auto size = _count;
                _release();
                detail::handle_error(error, "unable to allocate pinned memory of {} bytes", size);
            }
        }

        void _release() {
            if (_ptr) {
                auto error = cudaFreeHost(_ptr);
                detail::handle_error(error, "unable to free pinned memory");
            }

            _reset();
        }

        void _reset() {
            _ptr = nullptr;
            _count = 0;
        }

        size_t _count;

        using detail::tensor_impl_t<T>::_ptr;

    };

    template<typename T>
    auto view(cuda_device_tensor_t<T>& obj) {
        return cuda_view_t<T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride(), obj.device());
    }
    template<typename T>
    auto view(const cuda_device_tensor_t<T>& obj) {
        return cuda_view_t<const T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride(), obj.device());
    }
    template<typename T>
    auto view(cuda_host_tensor_t<T>& obj) {
        return cpu_view_t<T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride());
    }
    template<typename T>
    auto view(const cuda_host_tensor_t<T>& obj) {
        return cpu_view_t<const T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride());
    }

}
