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
static void _spiral_rectify(
    factor_t scale, factor_t offset,
    const cuda::strided_t<const in_t, 2> block,
    factor_t segment_pitch, factor_t segment_offset, factor_t radial_pitch, factor_t spiral_velocity,
    const cuda::strided_t<out_t, 3> volume, vec_xy<factor_t> volume_xy_pitch, vec_xy<factor_t> volume_xy_offset
) {

    // 2D coordinates into rectangular volume
    uint2 idx = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
    };

    // check valid volume coordinates
    if (idx.x >= volume.shape.x || idx.y >= volume.shape.y) {
        return;
    }

    // find target XY position
    // TODO: apply transformation matrix here instead (#69)
    auto x = volume_xy_pitch.x * idx.x + volume_xy_offset.x;
    auto y = volume_xy_pitch.y * idx.y + volume_xy_offset.y;

    // convert target XY position to polar coordinates
    auto r = std::hypot(x, y);
    auto theta = std::atan2(x, y);

    const auto& dr = radial_pitch;
    const auto& omega = spiral_velocity;

    // map the position to the nearest spiral ring
    auto ideal_t = (r * 2 * M_PI) / (dr * omega);
    auto ring = (omega * ideal_t - theta) / (2 * M_PI);
    if (theta >= 0) {
        // round to prior ring
        ring = cuda::kernel::floor(ring);
    } else {
        // round to next ring
        ring = cuda::kernel::ceil(ring);
    }
    if (ring < 0) {
        ring = 0;
    }
    auto spiral_t = (ring * 2 * M_PI + theta) / omega;

    // map linearly into spiral
    auto sidx = (spiral_t - segment_offset) / segment_pitch;
    ptrdiff_t sidx_before = cuda::kernel::floor(sidx);
    ptrdiff_t sidx_after = cuda::kernel::ceil(sidx);

    // check bounds
    if (sidx_after < 0 || sidx_before >= block.shape.x) {
        return;
    }

    // clamp indices that map straddle the bound
    sidx_before = std::max<ptrdiff_t>(sidx_before, 0);
    sidx_after = std::min<ptrdiff_t>(sidx_after, block.shape.x - 1);
    auto ratio = sidx - sidx_before;

    // load this A-scan in the volume
    for (int z = 0; z < volume.shape.z; z++) {
        // index into B-scan at the A-scan sample chosen by Z position
        const auto& before = block[sidx_before * block.stride.x + z * block.stride.y];
        const auto& after = block[sidx_after * block.stride.x + z * block.stride.y];

        // index into volume
        auto& out = volume[idx.x * volume.stride.x + idx.y * volume.stride.y + z * volume.stride.z];

        // out-of-place linear transform
        out = cuda::kernel::round_clip_cast<out_t>(scale * (ratio * before + (1 - ratio) * after) + offset);
    }
}

//template<typename in_t, typename out_t, typename factor_t typename galvo_t>
//__global__
//static void _spiral_rectify(
//    factor_t scale, factor_t offset, const galvo_t* sample_target, const galvo_t*, sample_actual,
//    const in_t* block, ulonglong2 block_shape, longlong2 block_stride, vec_rt<factor_t> segment_rt_pitch, vec_rt<factor_t> segment_rt_offset,
//    out_t* volume, ulonglong3 volume_shape, longlong3 volume_stride, vec_xy<factor_t> volume_xy_pitch, vec_xy<factor_t> volume_xy_offset
//) {
//
//    uint2 idx = {
//        blockIdx.x * blockDim.x + threadIdx.x, // moving laterally through volume (dimension 0)
//        blockIdx.y * blockDim.y + threadIdx.y, // moving laterally through volume (dimension 1)
//    };
//
//    // check valid volume coordinates
//    if (idx.x >= volume_shape.x || idx.y >= volume_shape.y) {
//        return;
//    }
//
//    // find target XY position
//    // TODO: apply transformation matrix here instead (#69)
//    auto x = volume_xy_pitch.x * idx.x + volume_xy_offset.x;
//    auto y = volume_xy_pitch.y * idx.y + volume_xy_offset.y;
//
//    // convert to polar coordinates
//    auto r = std::hypot(x, y);
//    auto theta = std::atan2(x, y);
//
//    // check both ends of the scan
//    ptrdiff_t ridx_before, ridx_after;
//    factor_t ratio;
//    bool valid = false;
//    for (size_t i = 0; i < 2; i++) {
//        if (i > 0) {
//            r = -r;
//            theta += M_PI;
//        }
//
//        // check angular range
//        auto diff = theta - segment_rt_offset.t;
//        diff -= std::round(diff / (2 * M_PI)) * (2 * M_PI);
//        if (std::abs(diff) > segment_rt_pitch.t / 2) {
//            continue;
//        }
//
//        // map linearly into B-scan
//        auto ridx = (r - segment_rt_offset.r) / segment_rt_pitch.r;
//        ridx_before = std::floor(ridx);
//        ridx_after = std::ceil(ridx);
//
//        // check bounds
//        if (ridx_after < 0 || ridx_before >= block_shape.x) {
//            continue;
//        }
//
//        // clamp indices that map straddle the bound
//        ridx_before = std::max<ptrdiff_t>(ridx_before, 0);
//        ridx_after = std::min<ptrdiff_t>(ridx_after, block_shape.x - 1);
//        ratio = ridx - ridx_before;
//
//        valid = true;
//        break;
//    }
//    if (!valid) {
//        return;
//    }
//
//    // load this A-scan in the volume
//    for (int z = 0; z < volume_shape.z; z++) {
//        // index into B-scan at the A-scan sample chosen by Z position
//        const auto& before = block[ridx_before * block_stride.x + z * block_stride.y];
//        const auto& after = block[ridx_after * block_stride.x + z * block_stride.y];
//
//        // index into volume
//        auto& out = volume[idx.x * volume_stride.x + idx.y * volume_stride.y + z * volume_stride.z];
//
//        // out-of-place linear transform
//        out = cuda::kernel::round_clip_cast<out_t>(scale * (ratio * before + (1 - ratio) * after) + offset);
//    }
//}
//
//template<typename in_t, typename out_t, typename factor_t typename galvo_t>
//static void _spiral_rectify_internal(
//    const cuda::stream_t& stream, factor_t scale, factor_t offset, const cuda::device_memory_t<galvo_t>& sample_actual, const cuda::device_memory_t<galvo_t>& sample_target,
//    const cuda::device_memory_t<in_t>& block, size_t block_offset, std::array<size_t, 2> block_shape, std::array<ptrdiff_t, 2> block_stride, std::array<factor_t, 2> segment_rt_pitch, std::array<factor_t, 2> segment_rt_offset,
//    const cuda::device_memory_t<out_t>& volume, size_t volume_offset, std::array<size_t, 3> volume_shape, std::array<ptrdiff_t, 3> volume_stride, std::array<factor_t, 2> volume_xy_pitch, std::array<factor_t, 2> volume_xy_offset
//) {
//    auto threads = cuda::kernel::threads_from_shape(volume_shape[0], volume_shape[1]);
//    auto blocks = cuda::kernel::blocks_from_threads(threads, volume_shape[0], volume_shape[1]);
//
//    _spiral_rectify << <blocks, threads, 0, stream.handle() >> > (
//        scale, offset, sample_target.ptr(), sample_actual.ptr(),
//        block.ptr() + block_offset, { block_shape[0], block_shape[1] }, { block_stride[0], block_stride[1] }, { segment_rt_pitch[0], segment_rt_pitch[1] }, { segment_rt_offset[0], segment_rt_offset[1] },
//        volume.ptr() + volume_offset, { volume_shape[0], volume_shape[1], volume_shape[2] }, { volume_stride[0], volume_stride[1], volume_stride[2] }, { volume_xy_pitch[0], volume_xy_pitch[1] }, { volume_xy_offset[0], volume_xy_offset[1] }
//    );
//#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
//    cudaDeviceSynchronize();
//#endif
//    auto error = cudaGetLastError();
//    cuda::detail::handle_error(error, "spiral rectify kernel launch failed: {}", cudaGetErrorName(error));
//}

template<typename in_t, typename out_t, typename factor_t>
static void _spiral_rectify_internal(
    const cuda::stream_t& stream, factor_t scale, factor_t offset,
    const cuda::strided_t<const in_t, 2>& block, factor_t segment_pitch, factor_t segment_offset, factor_t radial_pitch, factor_t spiral_velocity,
    const cuda::strided_t<out_t, 3> volume, const std::array<factor_t, 2>& volume_xy_pitch, const std::array<factor_t, 2>& volume_xy_offset
) {
    auto threads = cuda::kernel::threads_from_shape(volume.shape.x, volume.shape.y);
    auto blocks = cuda::kernel::blocks_from_threads(threads, volume.shape.x, volume.shape.y);

    _spiral_rectify<<<blocks, threads, 0, stream.handle()>>> (
        scale, offset,
        block, segment_pitch, segment_offset, radial_pitch, spiral_velocity,
        volume, _to_vec_xy(volume_xy_pitch), _to_vec_xy(volume_xy_offset)
        );
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
    cudaDeviceSynchronize();
#endif
    auto error = cudaGetLastError();
    cuda::detail::handle_error(error, "spiral rectify kernel launch failed: {}", cudaGetErrorName(error));
}

// NVCC does not support nested namespaces yet
namespace vortex {
    namespace format {
        namespace detail {

            //void spiral_rectify(
            //    const cuda::stream_t& stream, float scale, float offset, const cuda::device_memory_t<double>& sample_actual, const cuda::device_memory_t<double>& sample_target,
            //    const cuda::device_memory_t<int8_t>& block, size_t block_offset, std::array<size_t, 2> block_shape, std::array<ptrdiff_t, 2> block_stride, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
            //    const cuda::device_memory_t<int8_t>& volume, size_t volume_offset, std::array<size_t, 3> volume_shape, std::array<ptrdiff_t, 3> volume_stride, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            //) {
            //    _spiral_rectify_internal(
            //        stream, scale, offset, sample_target, sample_actual
            //        block, block_offset, block_shape, block_stride, segment_rt_pitch, segment_rt_offset,
            //        volume, volume_offset, volume_shape, volume_stride, volume_xy_pitch, volume_xy_offset
            //    );
            //}
            //void spiral_rectify(
            //    const cuda::stream_t& stream, float scale, float offset, const cuda::device_memory_t<double>& sample_actual, const cuda::device_memory_t<double>& sample_target,
            //    const cuda::device_memory_t<float>& block, size_t block_offset, std::array<size_t, 2> block_shape, std::array<ptrdiff_t, 2> block_stride, std::array<float, 2> segment_rt_pitch, std::array<float, 2> segment_rt_offset,
            //    const cuda::device_memory_t<int8_t>& volume, size_t volume_offset, std::array<size_t, 3> volume_shape, std::array<ptrdiff_t, 3> volume_stride, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            //) {
            //    _spiral_rectify_internal(
            //        stream, scale, offset, sample_target, sample_actual,
            //        block, block_offset, block_shape, block_stride, segment_rt_pitch, segment_rt_offset,
            //        volume, volume_offset, volume_shape, volume_stride, volume_xy_pitch, volume_xy_offset
            //    );
            //}

            void spiral_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const int8_t, 2>& block, float segment_pitch, float segment_offset, float radial_pitch, float spiral_velocity,
                const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _spiral_rectify_internal(stream, scale, offset, block, segment_pitch, segment_offset, radial_pitch, spiral_velocity, volume, volume_xy_pitch, volume_xy_offset);
            }
            void spiral_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const uint16_t, 2>& block, float segment_pitch, float segment_offset, float radial_pitch, float spiral_velocity,
                const cuda::strided_t<uint16_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _spiral_rectify_internal(stream, scale, offset, block, segment_pitch, segment_offset, radial_pitch, spiral_velocity, volume, volume_xy_pitch, volume_xy_offset);
            }
            void spiral_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const float, 2>& block, float segment_pitch, float segment_offset, float radial_pitch, float spiral_velocity,
                const cuda::strided_t<int8_t, 3> volume, std::array<float, 2> volume_xy_pitch, std::array<float, 2> volume_xy_offset
            ) {
                _spiral_rectify_internal(stream, scale, offset, block, segment_pitch, segment_offset, radial_pitch, spiral_velocity, volume, volume_xy_pitch, volume_xy_offset);
            }

        }
    }
}
