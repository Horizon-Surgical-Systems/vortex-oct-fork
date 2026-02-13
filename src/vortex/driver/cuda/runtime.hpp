/** \rst

    CUDA type wrappers

    This file contains several object-oriented wrappers around CUDA
    events, textures, surfaces, and streams.

    NOTE: A cuda::stream_t object is *completely distinct* from
    vortex's concept of "streams" which hold signals
    (cf. block/stream.hpp). In contrast, a CUDA stream is somewhat
    analogous to an additional level of parallelism over that already
    provided by CUDA's parallel threads.

 \endrst */

#pragma once

#include <vector>
#include <numeric>
#include <complex>

#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <fmt/format.h>

#include <vortex/driver/cuda/types.hpp>

// NOTE: no namespace shorthand here so NVCC can include this header
namespace vortex {
    namespace cuda {

        using device_t = int;
        static auto constexpr invalid_device = -1;

        device_t device(device_t index);
        device_t device();

        void peer_access(cuda::device_t device_visible_to, cuda::device_t device_visible_from, bool enable);
        bool peer_access(cuda::device_t device_visible_to, cuda::device_t device_visible_from);

        template<typename T>
        class device_array_t;

        class stream_t;

        class event_t {
        public:
            event_t(unsigned int flags = cudaEventDefault);
            ~event_t();

            void sync() const;
            bool done() const;

            void record();
            void record(const stream_t& stream);

            // compute time in seconds from start to this event
            float elapsed(const event_t& start) const;

            const cudaEvent_t& handle() const;

        protected:
            cudaEvent_t _event;
        };

        class texture_t {
        public:
            texture_t();

            template<typename T>
            texture_t(const device_array_t<T>& array, const cudaTextureDesc& texture_desc) : texture_t() {
                cudaResourceDesc resource_desc;
                std::memset(&resource_desc, 0, sizeof(resource_desc));

                resource_desc.resType = cudaResourceTypeArray;
                resource_desc.res.array.array = array.array();

                auto error = cudaCreateTextureObject(&_texture, &resource_desc, &texture_desc, 0);
                detail::handle_error(error, "unable to create texture object");
            }

            ~texture_t();

            texture_t(const texture_t&) = delete;
            texture_t& operator=(const texture_t&) = delete;

            texture_t(texture_t&& o);
            texture_t& operator=(texture_t&& o);

            bool valid() const;

            void reset();

            const cudaTextureObject_t& handle() const;

        protected:
            cudaTextureObject_t _texture;
        };

        class surface_t {
        public:
            surface_t();

            template<typename T>
            surface_t(const device_array_t<T>& array) : surface_t() {
                cudaResourceDesc resource_desc;
                std::memset(&resource_desc, 0, sizeof(resource_desc));

                resource_desc.resType = cudaResourceTypeArray;
                resource_desc.res.array.array = array.array();

                auto error = cudaCreateSurfaceObject(&_surface, &resource_desc);
                detail::handle_error(error, "unable to create surface object");
            }

            ~surface_t();

            surface_t(const surface_t&) = delete;
            surface_t& operator=(const surface_t&) = delete;

            surface_t(surface_t&& o);
            surface_t& operator=(surface_t&& o);

            bool valid() const;

            const cudaSurfaceObject_t& handle() const;

        protected:
            cudaSurfaceObject_t _surface;
        };

        class stream_t {
        public:
            stream_t(unsigned int flags = cudaStreamNonBlocking);

            ~stream_t();

            stream_t(const stream_t&) = delete;
            stream_t& operator=(const stream_t&) = delete;

            stream_t(stream_t&& o);
            stream_t& operator=(stream_t&& o);

            const cudaStream_t& handle() const;

            void sync() const;

            void wait(const event_t& event) const;

            bool ready() const;

            static stream_t default_();

        protected:

            cudaStream_t _stream;
        };

        namespace detail {

            template<typename T>
            void memset(T* ptr, int value, size_t count, const stream_t* stream) {
                auto sid = stream ? stream->handle() : static_cast<cudaStream_t>(0);

                auto error = cudaMemsetAsync(ptr, value, count * sizeof(T), sid);
                detail::handle_error(error, "unable to memset {} {} at {} on stream {}", count, typeid(T).name(), (const void*)ptr, (void*)sid);
            }

            template<typename T, typename U, typename = typename std::enable_if_t<std::is_same<cuda::device_type<T>, cuda::device_type<U>>::value>>
            void memcpy(T* dst, const U* src, size_t count, cudaMemcpyKind kind, const stream_t* stream) {
                using V = cuda::device_type<T>;
                cudaStream_t sid = stream ? stream->handle() : static_cast<cudaStream_t>(0);

                auto error = cudaMemcpyAsync(dst, src, count * sizeof(V), kind, sid);
                detail::handle_error(error, "unable to memcpy {} {} from {} to {} on stream {}", count, typeid(V).name(), (const void*)src, (const void*)dst, (void*)sid);
            }

            template<typename T, typename U, typename = typename std::enable_if_t<std::is_same<cuda::device_type<T>, cuda::device_type<U>>::value>>
            void memcpy(T* dst, size_t dst_pitch, const U* src, size_t src_pitch, size_t width, size_t height, cudaMemcpyKind kind, const stream_t* stream) {
                using V = cuda::device_type<T>;
                auto sid = stream ? stream->handle() : static_cast<cudaStream_t>(0);

                auto error = cudaMemcpy2DAsync(dst, dst_pitch * sizeof(V), src, src_pitch * sizeof(V), width * sizeof(V), height, kind, sid);
                detail::handle_error(error, "unable to memcpy 2D {} rows x {} cols {} from {} (pitch {}) to {} (pitch {}) on stream {}", height, width, typeid(V).name(), (const void*)src, src_pitch, (const void*)dst, dst_pitch, (void*)sid);
            }

        }

    }
}
