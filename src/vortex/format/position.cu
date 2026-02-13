/** \rst

    CUDA kernels to support position rectification

 \endrst */

#define _USE_MATH_DEFINES
#include <math.h>

#include <array>

#include <vortex/driver/cuda/types.hpp>
#include <vortex/driver/cuda/runtime.hpp>
#include <vortex/driver/cuda/kernels.cuh>

#include <vortex/driver/cuda/types.hpp>

using namespace vortex;

template<typename in_t, typename out_t, typename factor_t, typename position_t>
__global__
static void _position_rectify(
    factor_t scale,
    factor_t offset,
    cuda::strided_t<const in_t, 2> block,
    cuda::strided_t<const position_t, 2> position, ulonglong2 position_channels,
    cuda::strided_t<out_t, 3> volume,
    cuda::matrix_t<factor_t, 2, 3> transform
) {

    // moving through block
    auto bidx = blockIdx.x * blockDim.x + threadIdx.x;

    // check valid block coordinates
    if (bidx >= block.shape.x) {
        return;
    }

    // lookup position for this index
    const auto& xg = position[bidx * position.stride.x + position_channels.x * position.stride.y];
    const auto& yg = position[bidx * position.stride.x + position_channels.y * position.stride.y];

    // map position into volume coordinates
    longlong2 vidx = {
        cuda::kernel::floor_clip_cast<long long int>(transform.v[0][0] * xg + transform.v[0][1] * yg + transform.v[0][2]),
        cuda::kernel::floor_clip_cast<long long int>(transform.v[1][0] * xg + transform.v[1][1] * yg + transform.v[1][2])
    };

    // check valid volume coordinates
    if (vidx.x < 0 || vidx.x >= volume.shape.x || vidx.y < 0 || vidx.y >= volume.shape.y) {
        return;
    }

    // load this A-scan in the volume
    for (int z = 0; z < volume.shape.z; z++) {
        // index into volume
        auto& out = volume[vidx.x * volume.stride.x + vidx.y * volume.stride.y + z * volume.stride.z];

        // out-of-place linear transform
        out = cuda::kernel::round_clip_cast<out_t>(scale * block[bidx * block.stride.x + z * block.stride.y] + offset);
    }
}

template<typename in_t, typename out_t, typename factor_t, typename position_t>
static void _position_rectify_internal(
    const cuda::stream_t& stream,
    factor_t scale, factor_t offset,
    const cuda::strided_t<const in_t, 2>& block,
    const cuda::strided_t<const position_t, 2>& position, const std::array<size_t, 2>& position_channels,
    const cuda::strided_t<out_t, 3>& volume,
    const cuda::matrix_t<factor_t, 2, 3>& transform
) {
    auto threads = cuda::kernel::threads_from_shape(block.shape.x);
    auto blocks = cuda::kernel::blocks_from_threads(threads, block.shape.x);

    _position_rectify<<<blocks, threads, 0, stream.handle()>>>(scale, offset, block, position, cuda::to_vec(position_channels), volume, transform);

#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
    cudaDeviceSynchronize();
#endif
    auto error = cudaGetLastError();
    cuda::detail::handle_error(error, "position rectify kernel launch failed: {}", cudaGetErrorName(error));
}

namespace vortex {
    namespace format {
        namespace detail {

            void position_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const int8_t, 2>& block,
                const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
                const cuda::strided_t<int8_t, 3>& volume,
                const cuda::matrix_t<float, 2, 3>& transform
            ) {
                _position_rectify_internal(stream, scale, offset, block, position, position_channels, volume, transform);
            }
            void position_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const uint16_t, 2>& block,
                const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
                const cuda::strided_t<uint16_t, 3>& volume,
                const cuda::matrix_t<float, 2, 3>& transform
            ) {
                _position_rectify_internal(stream, scale, offset, block, position, position_channels, volume, transform);
            }
            void position_rectify(
                const cuda::stream_t& stream, float scale, float offset,
                const cuda::strided_t<const float, 2>& block,
                const cuda::strided_t<const double, 2>& position, const std::array<size_t, 2>& position_channels,
                const cuda::strided_t<int8_t, 3>& volume,
                const cuda::matrix_t<float, 2, 3>& transform
            ) {
                _position_rectify_internal(stream, scale, offset, block, position, position_channels, volume, transform);
            }
        }
    }
}
