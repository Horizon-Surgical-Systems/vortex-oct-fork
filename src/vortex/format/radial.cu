/** \rst

    CUDA kernels to support radial rectification

 \endrst */

 #define _USE_MATH_DEFINES
#include <math.h>

#include <array>

#include <vortex/driver/cuda/types.hpp>
#include <vortex/driver/cuda/runtime.hpp>
#include <vortex/driver/cuda/kernels.cuh>

using namespace vortex;

template<typename T>
struct vec_xy {
    T x, y;
};
template<typename T>
inline auto _to_vec_xy(const std::array<T, 2>& in) {
    return vec_xy<T>{in[0], in[1]};
}

// 2-vector in polar coordinates
template<typename T>
struct vec_rt {
    T r, t;
};
template<typename T>
inline auto _to_vec_rt(const std::array<T, 2>& in) {
    return vec_rt<T>{in[0], in[1]};
}

template<typename in_t, typename out_t, typename factor_t>
__global__
static void _radial_rectify(
    factor_t scale, factor_t offset,
    const cuda::strided_t<const in_t, 2> block, vec_rt<factor_t> segment_rt_pitch, vec_rt<factor_t> segment_rt_offset,
    const cuda::strided_t<out_t, 3> volume, vec_xy<factor_t> volume_xy_pitch, vec_xy<factor_t> volume_xy_offset
) {

    // 2-d coordinates into rectangular volume
    uint2 idx = {
        blockIdx.x * blockDim.x + threadIdx.x, // moving laterally through volume (dimension 0)
        blockIdx.y * blockDim.y + threadIdx.y, // moving laterally through volume (dimension 1)
    };

    // check valid volume coordinates
    if (idx.x >= volume.shape.x || idx.y >= volume.shape.y) {
        return;
    }

    // find target XY position (some sort of scale + offset)
    // TODO: apply transformation matrix here instead (#69)
    auto x = volume_xy_pitch.x * idx.x + volume_xy_offset.x;
    auto y = volume_xy_pitch.y * idx.y + volume_xy_offset.y;

    // convert target XY position to polar coordinates
    auto r = std::hypot(x, y);
    auto theta = std::atan2(x, y);

    // check both ends of the scan
    ptrdiff_t ridx_before, ridx_after;
    factor_t ratio;
    bool valid = false;
    for (size_t i = 0; i < 2; i++) {
        if (i > 0) {
            r = -r;
            theta += M_PI;
        }

        // check angular range
        auto diff = theta - segment_rt_offset.t;
        diff -= std::round(diff / (2 * M_PI)) * (2 * M_PI);
        if (std::abs(diff) > segment_rt_pitch.t / 2) {
            continue;
        }

        // map linearly into B-scan
        auto ridx = (r - segment_rt_offset.r) / segment_rt_pitch.r;
        ridx_before = cuda::kernel::floor(ridx);
        ridx_after = cuda::kernel::ceil(ridx);

        // check bounds
        if (ridx_after < 0 || ridx_before >= block.shape.x) {
            continue;
        }

        // clamp indices that map straddle the bound
        ridx_before = std::max<ptrdiff_t>(ridx_before, 0);
        ridx_after = std::min<ptrdiff_t>(ridx_after, block.shape.x - 1);
        ratio = ridx - ridx_before;

        valid = true;
        break;
    }
    if (!valid) {
        return;
    }

    // load this A-scan in the volume
    for (int z = 0; z < volume.shape.z; z++) {
        // index into B-scan at the A-scan sample chosen by Z position
        const auto& before = block[ridx_before * block.stride.x + z * block.stride.y];
        const auto& after = block[ridx_after * block.stride.x + z * block.stride.y];

        // index into volume
        auto& out = volume[idx.x * volume.stride.x + idx.y * volume.stride.y + z * volume.stride.z];

        // out-of-place linear transform
        out = cuda::kernel::round_clip_cast<out_t>(scale * (ratio * before + (1 - ratio) * after) + offset);
    }
}

template<typename in_t, typename out_t, typename factor_t>
static void _radial_rectify_internal(
    const cuda::stream_t& stream, factor_t scale, factor_t offset,
    const cuda::strided_t<const in_t, 2>& block, const std::array<factor_t, 2>& segment_rt_pitch, const std::array<factor_t, 2>& segment_rt_offset,
    const cuda::strided_t<out_t, 3> volume, const std::array<factor_t, 2>& volume_xy_pitch, const std::array<factor_t, 2>& volume_xy_offset
) {
    auto threads = cuda::kernel::threads_from_shape(volume.shape.x, volume.shape.y);
    auto blocks = cuda::kernel::blocks_from_threads(threads, volume.shape.x, volume.shape.y);

    _radial_rectify<<<blocks, threads, 0, stream.handle()>>>(
        scale, offset,
        block, _to_vec_rt(segment_rt_pitch), _to_vec_rt(segment_rt_offset),
        volume, _to_vec_xy(volume_xy_pitch), _to_vec_xy(volume_xy_offset)
    );
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
    cudaDeviceSynchronize();
#endif
    auto error = cudaGetLastError();
    cuda::detail::handle_error(error, "radial rectify kernel launch failed: {}", cudaGetErrorName(error));
}

namespace vortex {
    namespace format {
        namespace detail {

            void radial_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const int8_t, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
                const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _radial_rectify_internal(stream, scale, offset, block, segment_rt_pitch, segment_rt_offset, volume, volume_xy_pitch, volume_xy_offset);
            }
            void radial_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const uint16_t, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
                const cuda::strided_t<uint16_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _radial_rectify_internal(stream, scale, offset, block, segment_rt_pitch, segment_rt_offset, volume, volume_xy_pitch, volume_xy_offset);
            }
            void radial_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const float, 2>& block, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
                const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _radial_rectify_internal(stream, scale, offset, block, segment_rt_pitch, segment_rt_offset, volume, volume_xy_pitch, volume_xy_offset);
            }

        }
    }
}
