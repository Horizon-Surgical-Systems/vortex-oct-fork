#pragma once

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <vortex/memory/view.hpp>
#include <vortex/memory/cpu.hpp>
#include <vortex/memory/cuda.hpp>

#include <vortex/driver/cuda/runtime.hpp>

namespace vortex::cuda {

    namespace detail {

        template<typename V1, typename V2>
        constexpr auto kind(const cuda_viewable<V1>&, const cuda_viewable<V2>&) { return cudaMemcpyKind::cudaMemcpyDeviceToDevice; };
        template<typename V1, typename V2>
        constexpr auto kind(const cuda_viewable<V1>&, const cpu_viewable<V2>&) { return cudaMemcpyKind::cudaMemcpyDeviceToHost; };
        template<typename V1, typename V2>
        constexpr auto kind(const cpu_viewable<V1>&, const cuda_viewable<V2>&) { return cudaMemcpyKind::cudaMemcpyHostToDevice; };
        template<typename V1, typename V2>
        constexpr auto kind(const cpu_viewable<V1>&, const cpu_viewable<V2>&) { return cudaMemcpyKind::cudaMemcpyHostToHost; };

    }

    template<typename V1, typename V2>
    void copy(const viewable<V1>& src_, const viewable<V2>& dst_, const stream_t* stream = nullptr) {
        auto& src = src_.derived_cast();
        auto& dst = dst_.derived_cast();

        if (src.count() != dst.count()) {
            throw traced<std::invalid_argument>(fmt::format("sizes do not match: {} vs {}", src.count(), dst.count()));
        }
        copy(src, dst, src.count(), stream);
    }
    template<typename V1, typename V2>
    void copy(const viewable<V1>& src, const viewable<V2>& dst, size_t count, const stream_t* stream = nullptr) {
        copy(src, 0, dst, 0, count, stream);
    }
    template<typename V1, typename V2>
    void copy(const viewable<V1>& src_, size_t src_offset, const viewable<V2>& dst_, size_t dst_offset, size_t count, const stream_t* stream = nullptr) {
        auto& src = src_.derived_cast();
        auto& dst = dst_.derived_cast();

        if (!dst.bounds().contains(dst.data() + dst_offset) || !dst.bounds().contains(dst.data() + dst_offset + count)) {
            throw traced<std::invalid_argument>(fmt::format("destination memory is too small for copy: {} < {}", dst.count(), dst_offset + count));
        }
        if (!src.bounds().contains(src.data() + src_offset) || !src.bounds().contains(src.data() + src_offset + count)) {
            throw traced<std::invalid_argument>(fmt::format("source memory is too small for copy: {} < {}", src.count(), src_offset + count));
        }
        detail::memcpy(dst.data() + dst_offset, src.data() + src_offset, count, detail::kind(src, dst), stream);
    }

    template<typename V1, typename V2>
    void copy(const viewable<V1>& src, size_t src_pitch, const viewable<V2>& dst, size_t dst_pitch, size_t rows, size_t cols, const stream_t* stream = nullptr) {
        copy(src, 0, src_pitch, dst, 0, dst_pitch, rows, cols, stream);
    }
    template<typename V1, typename V2>
    void copy(const viewable<V1>& src_, size_t src_offset, size_t src_pitch, const viewable<V2>& dst_, size_t dst_offset, size_t dst_pitch, size_t rows, size_t cols, const stream_t* stream = nullptr) {
        auto& src = src_.derived_cast();
        auto& dst = dst_.derived_cast();

        if (!dst.bounds().contains(dst.data() + dst_offset) || !dst.bounds().contains(dst.data() + dst_offset + (rows - 1) * dst_pitch + cols)) {
            throw traced<std::invalid_argument>(fmt::format("destination memory is too small for copy: {} < {}", dst.count(), dst_offset + (rows - 1) * dst_pitch + cols));
        }
        if (!src.bounds().contains(src.data() + src_offset) || !src.bounds().contains(src.data() + src_offset + (rows - 1) * src_pitch + cols)) {
            throw traced<std::invalid_argument>(fmt::format("source memory is too small for copy: {} < {}", src.count(), src_offset + (rows - 1) * src_pitch + cols));
        }
        detail::memcpy(dst.data() + dst_offset, dst_pitch, src.data() + src_offset, src_pitch, cols, rows, detail::kind(src, dst), stream);
    }

}
