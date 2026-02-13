#pragma once

#include <ATS_GPU.h>

#include <vortex/memory/tensor.hpp>
#include <vortex/memory/cuda.hpp>

#include <vortex/driver/alazar/core.hpp>

namespace vortex::alazar {

    template<typename derived_t>
    class alazar_viewable : public cuda::cuda_viewable<derived_t> {};
    template<typename T>
    inline constexpr bool is_alazar_viewable = std::is_base_of_v<alazar_viewable<std::decay_t<T>>, std::decay_t<T>>;

    namespace detail {
        using vortex::detail::tensor_impl_t;

        template<typename T>
        class alazar_view_t;

        template<typename T>
        struct fixed_alazar_view_t {
            template<size_t N>
            class of_dimension :
                public alazar_viewable<of_dimension<N>>,
                public vortex::detail::static_view_t<T, N, fixed_alazar_view_t<T>, alazar_view_t<T>>
            {
            public:
                using base_t = vortex::detail::static_view_t<T, N, fixed_alazar_view_t<T>, alazar_view_t<T>>;
                using typename base_t::element_t;

                of_dimension()
                    : base_t(), _device(cuda::invalid_device), _handle(0) {}
                template<typename S1, typename S2>
                of_dimension(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, const cuda::device_t& device, const HANDLE& handle)
                    : base_t(data, bounds, shape, stride), _device(device), _handle(handle) {}
                template<typename O, typename S1, typename S2>
                of_dimension(const O& other, T* data, const S1& shape, const S2& stride)
                    : of_dimension(data, other.bounds(), shape, stride, other.device(), other.handle()) {}

                template<typename V>
                of_dimension(const alazar_viewable<V>& other)
                    : of_dimension(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

                const auto& device() const { return _device; }

            protected:

                cuda::device_t _device;
                HANDLE _handle;
            };
        };

        template<typename T>
        class alazar_view_t :
            public alazar_viewable<alazar_view_t<T>>,
            public vortex::detail::dynamic_view_t<T, fixed_alazar_view_t<T>, alazar_view_t<T>>
        {
        public:
            using base_t = vortex::detail::dynamic_view_t<T, fixed_alazar_view_t<T>, alazar_view_t<T>>;
            using typename base_t::element_t;

            alazar_view_t()
                : base_t(), _device(cuda::invalid_device), _handle(0) {}

            template<typename S1, typename S2>
            alazar_view_t(T* data, const bounds_t<T>& bounds, const S1& shape, const S2& stride, const cuda::device_t& device, const HANDLE& handle)
                : base_t(data, bounds, shape, stride), _device(device), _handle(handle) {}
            template<typename O, typename S1, typename S2>
            alazar_view_t(const O& other, T* data, const S1& shape, const S2& stride)
                : alazar_view_t(data, other.bounds(), shape, stride, other.device(), other.handle()) {}

            template<typename V>
            alazar_view_t(const alazar_viewable<V>& other)
                : alazar_view_t(other.derived_cast(), other.derived_cast().bounds(), other.derived_cast().data(), other.derived_cast().shape(), other.derived_cast().stride()) {}

            const auto& device() const { return _device; }
            const auto& handle() const { return _handle; }

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
            HANDLE _handle;
        };
    }
    using detail::alazar_view_t;
    template<typename T, size_t N>
    using fixed_alazar_view_t = typename detail::fixed_alazar_view_t<T>::template of_dimension<N>;

    VORTEX_VIEW_AS_IMPL(is_alazar_viewable, alazar);

    template<typename T>
    class alazar_device_tensor_t : public detail::tensor_impl_t<T> {
    public:

        using element_t = T;

        alazar_device_tensor_t()
            : detail::tensor_impl_t<T>() {
            _reset();
        }
        alazar_device_tensor_t(HANDLE handle, cuda::device_t device)
            : alazar_device_tensor_t() {
            _handle = handle;
            _device = device;
        }

        ~alazar_device_tensor_t() {
            _release();
        }

        alazar_device_tensor_t(alazar_device_tensor_t&& o)
            : alazar_device_tensor_t(o.handle(), o.device()) {
            *this = std::move(o);
        }
        alazar_device_tensor_t& operator=(alazar_device_tensor_t&& o) {
            detail::tensor_impl_t<T>::operator=(std::move(o));

            std::swap(_handle, o._handle);
            std::swap(_count, o._count);

            return *this;
        }

        const auto& handle() const {
            return _handle;
        }
        const auto& device() const {
            return _device;
        }

        size_t underlying_count() const override {
            return _count;
        }

        using detail::tensor_impl_t<T>::size_in_bytes;

    protected:

        void _allocate(size_t count) override {
            if (count == _count) {
                return;
            }

            if (!_handle) {
                throw exception(fmt::format("attempt to allocate Alazar-accessible device memory without valid board handle"));
            }

            _release();

            // check for empty allocation
            if (count == 0) {
                return;
            }

            _count = count;

            _ptr = reinterpret_cast<T*>(ATS_GPU_AllocBuffer(_handle, size_in_bytes(), nullptr));
            if (!_ptr) {
                throw exception(fmt::format("unable to allocate Alazar-accessible device memory of {} bytes", size_in_bytes()));
            }
        }

        void _release() {
            if (_ptr) {
                auto rc = ATS_GPU_FreeBuffer(_handle, _ptr);
                detail::handle_error(rc, "unable to free Alazar-accessible device memory");
            }

            _reset();
        }

        void _reset() {
            // NOTE: device and handle cannot be changed and therefore are not cleared
            _ptr = nullptr;
            _count = 0;
        }

        size_t _count;

        cuda::device_t _device;
        HANDLE _handle;

        using detail::tensor_impl_t<T>::_ptr;
    };

    template<typename T>
    auto view(alazar_device_tensor_t<T>& obj) {
        return alazar::alazar_view_t<T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride(), obj.device(), obj.handle());
    }
    template<typename T>
    auto view(const alazar_device_tensor_t<T>& obj) {
        return alazar::alazar_view_t<const T>(obj.data(), { obj.data(), obj.data() + obj.count() }, obj.shape(), obj.stride(), obj.device(), obj.handle());
    }

}