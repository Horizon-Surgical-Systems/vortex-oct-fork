/** \rst

    CUDA kernels to support cube formatter

 \endrst */

 #include <array>

#include <vortex/driver/cuda/types.hpp>
#include <vortex/driver/cuda/runtime.hpp>
#include <vortex/driver/cuda/kernels.cuh>

using namespace vortex;

template<typename in_t, typename out_t, typename factor_t>
__global__
static void _linear_transform(factor_t scale, factor_t offset, size_t s0, size_t s1, size_t s2, const in_t* src, ptrdiff_t ss0, ptrdiff_t ss1, ptrdiff_t ss2, out_t* dst, ptrdiff_t ds0, ptrdiff_t ds1, ptrdiff_t ds2) {
    auto i0 = blockIdx.x * blockDim.x + threadIdx.x;
    auto i1 = blockIdx.y * blockDim.y + threadIdx.y;
    auto i2 = blockIdx.z * blockDim.z + threadIdx.z;

    // check valid source coordinates
    if (i0 >= s0 || i1 >= s1 || i2 >= s2) {
        return;
    }

    // index into array
    const auto& in = src[i0 * ss0 + i1 * ss1 + i2 * ss2];
    auto& out =      dst[i0 * ds0 + i1 * ds1 + i2 * ds2];

    // out-of-place linear transform
    out = cuda::kernel::round_clip_cast<out_t>(scale * in + offset);
}

template<typename in_t, typename out_t, typename factor_t>
static void _linear_transform_internal(const cuda::stream_t& stream, factor_t scale, factor_t offset, const std::array<size_t, 3> shape, const in_t* src, size_t src_offset, const std::array<ptrdiff_t, 3>& src_strides, out_t* dst, size_t dst_offset, const std::array<ptrdiff_t, 3>& dst_strides) {
    auto threads = cuda::kernel::threads_from_shape(shape[0], shape[1], shape[2]);
    auto blocks = cuda::kernel::blocks_from_threads(threads, shape[0], shape[1], shape[2]);

    _linear_transform<<<blocks, threads, 0, stream.handle()>>>(
        scale, offset,
        shape[0], shape[1], shape[2],
        src + src_offset, src_strides[0], src_strides[1], src_strides[2],
        dst + dst_offset, dst_strides[0], dst_strides[1], dst_strides[2]
    );
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
    cudaDeviceSynchronize();
#endif
    auto error = cudaGetLastError();
    cuda::detail::handle_error(error, "linear transform kernel launch failed: {}", cudaGetErrorName(error));
}

namespace vortex {
    namespace copy {
        namespace detail {

#define _DEFINE(src_type, dst_type) \
            void linear_transform(const cuda::stream_t& stream, float scale, float offset, const std::array<size_t, 3> shape, const src_type* src, size_t src_offset, const std::array<ptrdiff_t, 3>& src_strides, dst_type* dst, size_t dst_offset, const std::array<ptrdiff_t, 3>& dst_strides) { \
                _linear_transform_internal(stream, scale, offset, shape, src, src_offset, src_strides, dst, dst_offset, dst_strides); \
            }

            _DEFINE(int8_t, int8_t);
            _DEFINE(uint8_t, uint8_t);
            _DEFINE(uint16_t, uint16_t);
            _DEFINE(float, int8_t);
            _DEFINE(float, float);
            _DEFINE(uint16_t, float);
            _DEFINE(int8_t, float);

#undef _DEFINE

        }
    }
}
