/** \rst

    CUDA kernels to support CUDA OCT processor

 \endrst */

#include <thrust/binary_search.h>

#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
#  include <cub/cub.cuh>

#  include <thrust/functional.h>
#  include <thrust/execution_policy.h>
#  include <thrust/iterator/counting_iterator.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/iterator/permutation_iterator.h>
#endif

#include <vortex/driver/cuda/types.hpp>
#include <vortex/driver/cuda/runtime.hpp>
#include <vortex/driver/cuda/kernels.cuh>

#define _USE_MATH_DEFINES
#include <math.h>
#define M_PI_FLOAT float(M_PI)

#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
 // ref: https://github.com/NVIDIA/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
public:

    typedef typename cuda::thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor
    {
        using argument_type = difference_type;
        using result_type = difference_type;

        difference_type stride;

        __host__ __device__
            stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
            difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename cuda::thrust::counting_iterator<difference_type>                    CountingIterator;
    typedef typename cuda::thrust::transform_iterator<stride_functor, CountingIterator>  TransformIterator;
    typedef typename cuda::thrust::permutation_iterator<Iterator, TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    __host__ __device__
        strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    __host__ __device__
        iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    __host__ __device__
        iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};
#endif

#define INDEX_ENCODE_2D() \
    auto threads = cuda::kernel::threads_from_shape(out.shape.y, out.shape.x); \
    auto blocks = cuda::kernel::blocks_from_threads(threads, out.shape.y, out.shape.x);

#define INDEX_DECODE_2D() \
    auto record_idx = blockIdx.y * blockDim.y + threadIdx.y; \
    auto sample_idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (record_idx >= out.shape.x || sample_idx >= out.shape.y) { \
        return; \
    }

// NOTE: no namespace shorthand here so NVCC can compile this file
namespace vortex {
    namespace process {
        namespace detail {

            //
            // signed conversion
            //
            __global__
                static void _signed_cast(
                    cuda::strided_t<const uint16_t, 2> in,
                    cuda::strided_t<float, 2> out
                ) {
                INDEX_DECODE_2D();

                auto& src = in(record_idx, sample_idx);
                auto& dst = out(record_idx, sample_idx);

                dst = *reinterpret_cast<const int16_t*>(&src);
            }

            void signed_cast(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                INDEX_ENCODE_2D();

                _signed_cast<<<blocks, threads, 0, stream.handle()>>>(in, out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "signed cast kernel launch failed");
            }

            //
            // sum
            //

            size_t prepare_sum(
                const cuda::strided_t<uint32_t, 2>& keys
            ) {
#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
                size_t scratch_size;
                cudaError_t error;

                error = cuda::cub::DeviceReduce::ReduceByKey(
                    nullptr, scratch_size,
                    keys.ptr, reinterpret_cast<uint32_t*>(0),
                    reinterpret_cast<float*>(0), reinterpret_cast<float*>(0), reinterpret_cast<uint32_t*>(0),
                    cuda::cub::Sum(),
                    keys.shape.x * keys.shape.y
                );
                cuda::detail::handle_error(error, "sum planning failed");

                return scratch_size;
#else
                throw std::runtime_error("dynamic resampling is not supported");
#endif
            }

            template<typename T, typename U>
            void _sum_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint32_t, 2>& keys,
                const cuda::strided_t<const T, 2>& in,
                const cuda::strided_t<U, 1>& out_sum,
                const cuda::strided_t<uint32_t, 1>& out_count,
                void* scratch_ptr, size_t scratch_size
            ) {
#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)

                // NOTE: the number_of_runs parameter must be a device pointer
                auto error = cuda::cub::DeviceReduce::ReduceByKey(
                    scratch_ptr, scratch_size,
                    keys.ptr, out_count.ptr,
                    in.ptr, out_sum.ptr, out_count.ptr,
                    cuda::cub::Sum(),
                    in.shape.x * in.shape.y,
                    stream.handle()
                );
                cuda::detail::handle_error(error, "sum kernel(s) failed");

#else
                throw std::runtime_error("dynamic resampling is not supported");
#endif
            }

            void sum(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint32_t, 2>& keys,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<float, 1>& out_sum,
                const cuda::strided_t<uint32_t, 1>& out_count,
                void* scratch_ptr, size_t scratch_size
            ) {
                _sum_internal(stream, keys, in, out_sum, out_count, scratch_ptr, scratch_size);
            }
            void sum(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint32_t, 2>& keys,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<float, 1>& out_sum,
                const cuda::strided_t<uint32_t, 1>& out_count,
                void* scratch_ptr, size_t scratch_size
            ) {
                _sum_internal(stream, keys, in, out_sum, out_count, scratch_ptr, scratch_size);
            }

            //
            // resample
            //

            template<typename in_t, typename out_t, typename index_t, typename float_t>
            __global__
                static void _resample(
                    cuda::strided_t<const index_t, 1> before_index, cuda::strided_t<const index_t, 1> after_index,
                    cuda::strided_t<const float_t, 1> before_weight, cuda::strided_t<const float_t, 1> after_weight,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                // look up bounding samples
                auto offset = before_index.offset(sample_idx);
                auto& before = in(record_idx, before_index[offset]);
                auto& after = in(record_idx, after_index[offset]);

                // perform interpolation
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(before_weight[offset] * before + after_weight[offset] * after);
            }

            template<typename in_t, typename out_t, typename index_t, typename float_t>
            void _resample_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const index_t, 1>& before_index, const cuda::strided_t<const index_t, 1>& after_index,
                const cuda::strided_t<const float_t, 1>& before_weight, const cuda::strided_t<const float_t, 1>& after_weight,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<out_t, 2>& out
            ) {
                INDEX_ENCODE_2D();

                _resample<<<blocks, threads, 0, stream.handle()>>>(before_index, after_index, before_weight, after_weight, in, out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "resample kernel launch failed");
            }

            void resample(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint32_t, 1>& before_index, const cuda::strided_t<const uint32_t, 1>& after_index,
                const cuda::strided_t<const float, 1>& before_weight, const cuda::strided_t<const float, 1>& after_weight,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _resample_internal(stream, before_index, after_index, before_weight, after_weight, in, out);
            }
            void resample(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint32_t, 1>& before_index, const cuda::strided_t<const uint32_t, 1>& after_index,
                const cuda::strided_t<const float, 1>& before_weight, const cuda::strided_t<const float, 1>& after_weight,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _resample_internal(stream, before_index, after_index, before_weight, after_weight, in, out);
            }

            template<typename in_t, typename out_t, typename float_t>
            __global__
                static void _resample_phase(
                    cuda::strided_t<const float_t, 2> phase, cuda::strided_t<const float_t, 1> phase_max,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                // compute linearized phase query
                auto phase_query = phase_max(record_idx) * sample_idx / out.shape.y;

                // map phase to sample index
                //strided_range<const float_t*> phase_record(phase.ptr + phase.offset(record_idx), phase.ptr + phase.offset(record_idx + 1), phase.stride.y);
                //auto phase_before = cuda::thrust::lower_bound(cuda::thrust::seq, phase_record.begin(), phase_record.end(), phase_query);
                auto phase_begin = phase.ptr + phase.offset(record_idx);
                auto phase_end = phase_begin + phase.stride.x;
                auto phase_before = cuda::thrust::lower_bound(cuda::thrust::seq, phase_begin, phase_end, phase_query);

                // calculate and clamp bounding sample indices
                auto before_idx = phase_before - phase_begin;
                auto after_idx = before_idx + 1;
                if (after_idx == phase.shape.y) {
                    after_idx--;
                }

                // look up bounding samples
                auto& before = in(record_idx, before_idx);
                auto& after = in(record_idx, after_idx);

                // determine interpolation ratio
                auto ratio = phase_query - *phase_before;

                // perform interpolation
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>((1 - ratio) * before + ratio * after);
            }

            template<typename in_t, typename out_t, typename float_t>
            void _resample_phase_internal(
                const cuda::stream_t& stream,
                cuda::strided_t<const float_t, 2> phase, cuda::strided_t<const float_t, 1> phase_max,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_ENCODE_2D();

                _resample_phase<<<blocks, threads, 0, stream.handle()>>>(phase, phase_max, in, out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "dynamic resampling kernel launch failed");
            }

            void resample_phase(
                const cuda::stream_t& stream,
                cuda::strided_t<const float, 2> phase, cuda::strided_t<const float, 1> phase_max,
                cuda::strided_t<const uint16_t, 2> in,
                cuda::strided_t<float, 2> out
            ) {
                _resample_phase_internal(stream, phase, phase_max, in, out);
            }
            void resample_phase(
                const cuda::stream_t& stream,
                cuda::strided_t<const float, 2> phase, cuda::strided_t<const float, 1> phase_max,
                cuda::strided_t<const float, 2> in,
                cuda::strided_t<float, 2> out
            ) {
                _resample_phase_internal(stream, phase, phase_max, in, out);
            }

            //
            // remove average
            //

            __global__
                static void _compute_average_record(
                    cuda::strided_t<const float, 2> average_record_buffer,
                    cuda::strided_t<float, 1> average_record
                ) {
                // A-scans run along the rows
                auto sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

                // check valid source coordinates
                if (sample_idx >= average_record_buffer.shape.y) {
                    return;
                }

                // perform averaging
                auto ptr = average_record_buffer.ptr + sample_idx * average_record_buffer.stride.y;
                float out = 0.0f;
                for (auto record_idx = 0; record_idx < average_record_buffer.shape.x; record_idx++) {
                    out += *ptr;
                    ptr += average_record_buffer.stride.x;
                }

                // store result
                average_record(sample_idx) = out / average_record_buffer.shape.x;
            }
            void compute_average_record(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 2>& average_record_buffer,
                const cuda::strided_t<float, 1>& average_record
            ) {
                auto threads = cuda::kernel::threads_from_shape(average_record.shape);
                auto blocks = cuda::kernel::blocks_from_threads(threads, average_record.shape);

                _compute_average_record<<<blocks, threads, 0, stream.handle()>>>(average_record_buffer, average_record);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "compute average kernel launch failed");
            }

            template<typename in_t, typename out_t>
            __global__
                static void _subtract_average_record(
                    cuda::strided_t<const float, 1> sum,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform average subtraction
                out(record_idx, sample_idx) = in(record_idx, sample_idx) - sum(sample_idx);
            }
            template<typename in_t, typename out_t>
            __global__
                static void _subtract_average_record_ss(
                    cuda::strided_t<const float, 1> average,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                auto offset = in.offset(record_idx, sample_idx);

                // perform average subtraction
                out[offset] = in[offset] - average(sample_idx);
            }

            template<typename in_t, typename out_t>
            static void _subtract_average_record_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& average,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<out_t, 2>& out
            ) {
                INDEX_ENCODE_2D();

                if (in.stride == out.stride) {
                    _subtract_average_record_ss<<<blocks, threads, 0, stream.handle()>>>(average, in, out);
                } else {
                    _subtract_average_record<<<blocks, threads, 0, stream.handle()>>>(average, in, out);
                }
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "subtract average record kernel launch failed");
            }
            void subtract_average_record(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& average,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _subtract_average_record_internal(stream, average, in, out);
            }
            void subtract_average_record(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& average,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _subtract_average_record_internal(stream, average, in, out);
            }

            //
            // complex filter
            //

            template<typename in_t>
            __global__
                static void _complex_filter(
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<const cuFloatComplex, 1> filter,
                    cuda::strided_t<cuFloatComplex, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform filtering
                auto input_offset = in.offset(record_idx, sample_idx);
                auto output_offset = out.offset(record_idx, sample_idx);
                out[output_offset].x = filter(sample_idx).x * in[input_offset];
                out[output_offset].y = filter(sample_idx).y * in[input_offset];
            }
            template<typename in_t>
            __global__
                static void _complex_filter_ss(
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<const cuFloatComplex, 1> filter,
                    cuda::strided_t<cuFloatComplex, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform filtering
                auto offset = in.offset(record_idx, sample_idx);
                out[offset].x = filter(sample_idx).x * in[offset];
                out[offset].y = filter(sample_idx).y * in[offset];
            }
            template<typename in_t>
            void complex_filter_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<const cuFloatComplex, 1>& filter,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                INDEX_ENCODE_2D();

                _complex_filter<<<blocks, threads, 0, stream.handle()>>>(in, filter, out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "complex filter kernel launch failed");
            }
            void complex_filter(
                const cuda::stream_t& stream,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<const cuFloatComplex, 1>& filter,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                complex_filter_internal(stream, in, filter, out);
            }
            void complex_filter(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<const cuFloatComplex, 1>& filter,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                complex_filter_internal(stream, in, filter, out);
            }

            //
            // cast
            //

            template<typename in_t>
            __global__
            static void _cast(
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform casting
                auto output_offset = out.offset(record_idx, sample_idx);
                out[output_offset].x = float(in(record_idx, sample_idx));
                out[output_offset].y = 0.0f;
            }
            template<typename in_t>
            __global__
            static void _cast_ss(
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform casting
                auto offset = in.offset(record_idx, sample_idx);
                out[offset].x = float(in[offset]);
                out[offset].y = 0.0f;
            }
            template<>
            __global__
            void _cast(
                cuda::strided_t<const cuFloatComplex, 2> in,
                cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform copying
                out(record_idx, sample_idx) = in(record_idx, sample_idx);
            }
            template<>
            __global__
            void _cast_ss(
                cuda::strided_t<const cuFloatComplex, 2> in,
                cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform copying
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = in[offset];
            }

            template<typename in_t>
            static void _cast_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                INDEX_ENCODE_2D();

                if (in.stride == out.stride) {
                    _cast_ss<<<blocks, threads, 0, stream.handle()>>>(in, out);
                } else {
                    _cast<<<blocks, threads, 0, stream.handle()>>>(in, out);
                }
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "cast kernel launch failed");
            }
            void cast(const cuda::stream_t& stream, const cuda::strided_t<const uint16_t, 2>& in, const cuda::strided_t<cuFloatComplex, 2>& out) {
                _cast_internal(stream, in, out);
            }
            void cast(const cuda::stream_t& stream, const cuda::strided_t<const float, 2>& in, const cuda::strided_t<cuFloatComplex, 2>& out) {
                _cast_internal(stream, in, out);
            }
            void cast(const cuda::stream_t& stream, const cuda::strided_t<const cuFloatComplex, 2>& in, const cuda::strided_t<cuFloatComplex, 2>& out) {
                _cast_internal(stream, in, out);
            }

            //
            // copy
            //

            template<typename T, typename U>
            __global__
                static void _copy(
                    cuda::strided_t<const T, 2> in,
                    cuda::strided_t<U, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform copy
                out(record_idx, sample_idx) = in(record_idx, sample_idx);
            }
            template<typename T, typename U>
            __global__
                static void _copy_ss(
                    cuda::strided_t<const T, 2> in,
                    cuda::strided_t<U, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform copy
                auto offset = out.offset(record_idx, sample_idx);
                out[offset] = in[offset];
            }

            template<typename T, typename U>
            static void _copy_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const T, 2>& in,
                const cuda::strided_t<U, 2>& out
            ) {

                if constexpr (std::is_same<T, U>::value) {
                    if (in.stride == out.stride && in.stride.x == in.shape.y && in.stride.y == 1) {

                        // contiguous memcpy is faster than kernel
                        vortex::cuda::detail::memcpy(out.ptr, in.ptr, in.shape.x * in.shape.y, cudaMemcpyDeviceToDevice, &stream);

                        cudaError_t error = cudaGetLastError();
                        cuda::detail::handle_error(error, "copy D2D failed");

                        return;
                    }
                }

                INDEX_ENCODE_2D();

                if (in.stride == out.stride) {
                    _copy_ss<<<blocks, threads, 0, stream.handle()>>>(in, out);
                } else {
                    _copy<<<blocks, threads, 0, stream.handle()>>>(in, out);
                }
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "copy kernel launch failed");
            }
            void copy(const cuda::stream_t& stream, const cuda::strided_t<const uint16_t, 2>& in, const cuda::strided_t<uint16_t, 2>& out) {
                _copy_internal(stream, in, out);
            }
            void copy(const cuda::stream_t& stream, const cuda::strided_t<const uint16_t, 2>& in, const cuda::strided_t<float, 2>& out) {
                _copy_internal(stream, in, out);
            }
            void copy(const cuda::stream_t& stream, const cuda::strided_t<const float, 2>& in, const cuda::strided_t<float, 2>& out) {
                _copy_internal(stream, in, out);
            }
            void copy(const cuda::stream_t& stream, const cuda::strided_t<const int8_t, 2>& in, const cuda::strided_t<int8_t, 2>& out) {
                _copy_internal(stream, in, out);
            }

            //
            // log abs normalize
            //

            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _log10_normalize_abs(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform abs normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::log10<factor_t>(cuda::kernel::abs(in(record_idx, sample_idx))) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _log10_normalize_abs_ss(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform abs normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::log10<factor_t>(cuda::kernel::abs(in[offset])) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _log10_normalize_real(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform abs normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::log10<factor_t>(cuda::kernel::real(in(record_idx, sample_idx))) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _log10_normalize_real_ss(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform abs normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::log10<factor_t>(cuda::kernel::real(in[offset])) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _sqr_normalize(
                factor_t postfactor,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform square normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::sqr(cuda::kernel::abs(in(record_idx, sample_idx))));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _sqr_normalize(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform square normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::sqr(cuda::kernel::abs(in(record_idx, sample_idx))) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _sqr_normalize_ss(
                factor_t postfactor,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform square normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::sqr(cuda::kernel::abs(in[offset])));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _sqr_normalize_ss(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform square normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::sqr(cuda::kernel::abs(in[offset])) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_abs(
                factor_t postfactor,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::abs(in(record_idx, sample_idx)));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_abs(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::abs(in(record_idx, sample_idx)) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_abs_ss(
                factor_t postfactor,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::abs(in[offset]));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_abs_ss(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::abs(in[offset]) + postoffset);
            }

            template<typename in_t, typename out_t, typename factor_t>
            __global__
                static void _normalize_real(
                    factor_t postfactor,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::real(in(record_idx, sample_idx)));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
                static void _normalize_real(
                    factor_t postfactor, factor_t postoffset,
                    cuda::strided_t<const in_t, 2> in,
                    cuda::strided_t<out_t, 2> out
                ) {
                INDEX_DECODE_2D();

                // perform absolute value normalize
                out(record_idx, sample_idx) = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::real(in(record_idx, sample_idx)) + postoffset);
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_real_ss(
                factor_t postfactor,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform real normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::real(in[offset]));
            }
            template<typename in_t, typename out_t, typename factor_t>
            __global__
            static void _normalize_real_ss(
                factor_t postfactor, factor_t postoffset,
                cuda::strided_t<const in_t, 2> in,
                cuda::strided_t<out_t, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform real normalize
                auto offset = in.offset(record_idx, sample_idx);
                out[offset] = cuda::kernel::round_clip_cast<out_t>(postfactor * cuda::kernel::real(in[offset]) + postoffset);
            }

            template<typename in_t, typename out_t, typename factor_t>
            static void _abs_normalize_internal(
                const cuda::stream_t& stream,
                factor_t factor,
                bool enable_log10, bool enable_square, bool enable_magnitude,
                bool enable_levels, factor_t levels_min, factor_t levels_max,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<out_t, 2>& out
            ) {
                INDEX_ENCODE_2D();

                factor_t sqr_factor = factor_t(enable_square ? 20 : 10);

                if (enable_levels) {
                    factor_t scale, offset;

                    // simplify computations for kernels
                    if constexpr (std::is_integral_v<out_t>) {
                        // map to [levels_min, levels_max]
                        constexpr factor_t min = std::numeric_limits<out_t>::lowest();
                        constexpr factor_t max = std::numeric_limits<out_t>::max();
                        scale = (max - min) / (levels_max - levels_min);
                        offset = -levels_min * scale + min;
                    } else {
                        // scale to [0, 1]
                        scale = factor_t(1) / (levels_max - levels_min);
                        offset = -levels_min * scale;
                    }

                    if (in.stride == out.stride) {
                        if (enable_log10) {
                            if (enable_magnitude) {
                                _log10_normalize_abs_ss<<<blocks, threads, 0, stream.handle()>>>(sqr_factor * scale, sqr_factor * scale * std::log10(factor) + offset, in, out);
                            } else {
                                _log10_normalize_real_ss<<<blocks, threads, 0, stream.handle()>>>(sqr_factor * scale, sqr_factor * scale * std::log10(factor) + offset, in, out);
                            }
                        } else if (enable_square) {
                            _sqr_normalize_ss<<<blocks, threads, 0, stream.handle()>>>(factor * factor * scale, offset, in, out);
                        } else {
                            if (enable_magnitude) {
                                _normalize_abs_ss<<<blocks, threads, 0, stream.handle()>>>(factor * scale, offset, in, out);
                            } else {
                                _normalize_real_ss<<<blocks, threads, 0, stream.handle()>>>(factor * scale, offset, in, out);
                            }
                        }
                    } else {
                        if (enable_log10) {
                            if (enable_magnitude) {
                                _log10_normalize_abs<<<blocks, threads, 0, stream.handle()>>>(sqr_factor * scale, sqr_factor * scale * std::log10(factor) + offset, in, out);
                            } else {
                                _log10_normalize_real<<<blocks, threads, 0, stream.handle()>>>(sqr_factor * scale, sqr_factor * scale * std::log10(factor) + offset, in, out);
                            }
                        } else if (enable_square) {
                            _sqr_normalize<<<blocks, threads, 0, stream.handle()>>>(factor * factor * scale, offset, in, out);
                        } else {
                            if (enable_magnitude) {
                                _normalize_abs<<<blocks, threads, 0, stream.handle()>>>(factor * scale, offset, in, out);
                            } else {
                                _normalize_real<<<blocks, threads, 0, stream.handle()>>>(factor * scale, offset, in, out);
                            }
                        }
                    }
                } else {
                    if (in.stride == out.stride) {
                        if (enable_log10) {
                            if (enable_magnitude) {
                                _log10_normalize_abs_ss<<<blocks, threads, 0, stream.handle()>>>(sqr_factor, sqr_factor * std::log10(factor), in, out);
                            } else {
                                _log10_normalize_real_ss<<<blocks, threads, 0, stream.handle()>>>(sqr_factor, sqr_factor * std::log10(factor), in, out);
                            }
                        } else if (enable_square) {
                            _sqr_normalize_ss<<<blocks, threads, 0, stream.handle()>>>(factor * factor, in, out);
                        } else {
                            if (enable_magnitude) {
                                _normalize_abs_ss<<<blocks, threads, 0, stream.handle()>>>(factor, in, out);
                            } else {
                                _normalize_real_ss<<<blocks, threads, 0, stream.handle()>>>(factor, in, out);
                            }
                        }
                    } else {
                        if (enable_log10) {
                            if (enable_magnitude) {
                                _log10_normalize_abs<<<blocks, threads, 0, stream.handle()>>>(sqr_factor, sqr_factor * std::log10(factor), in, out);
                            } else {
                                _log10_normalize_real<<<blocks, threads, 0, stream.handle()>>>(sqr_factor, sqr_factor * std::log10(factor), in, out);
                            }
                        } else if (enable_square) {
                            _sqr_normalize<<<blocks, threads, 0, stream.handle()>>>(factor * factor, in, out);
                        } else {
                            if (enable_magnitude) {
                                _normalize_abs<<<blocks, threads, 0, stream.handle()>>>(factor, in, out);
                            } else {
                                _normalize_real<<<blocks, threads, 0, stream.handle()>>>(factor, in, out);
                            }
                        }
                    }
                }

#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "abs2 normalize kernel launch failed");
            }

#define _DECLARE(factor_t, in_t, out_t) \
    void abs_normalize( \
        const cuda::stream_t& stream, \
        factor_t factor, \
        bool enable_log10, bool enable_square, bool enable_magnitude, \
        bool enable_levels, factor_t levels_min, factor_t levels_max, \
        const cuda::strided_t<const in_t, 2>& in, \
        const cuda::strided_t<out_t, 2>& out \
    ) { \
        _abs_normalize_internal(stream, factor, enable_log10, enable_square, enable_magnitude, enable_levels, levels_min, levels_max, in, out); \
    }

            _DECLARE(float, uint16_t, float);
            _DECLARE(float, float, float);
            _DECLARE(float, cuFloatComplex, float);
            _DECLARE(float, uint16_t, int8_t);
            _DECLARE(float, float, int8_t);
            _DECLARE(float, cuFloatComplex, int8_t);
#undef _DECLARE

            // NOTE: not yet ready to support double for the other functions but there is no reason it cannot be done
            //_DECLARE(double, uint16_t, double);
            //_DECLARE(double, double, double);
            //_DECLARE(double, cuDoubleComplex, double);
            //_DECLARE(double, uint16_t, int8_t);
            //_DECLARE(double, double, int8_t);
            //_DECLARE(double, cuDoubleComplex, int8_t);

            //
            // demean
            //

            template<typename in_t>
            __global__
            static void _demean_and_cast(
                const cuda::strided_t<const float, 1> sum, float divisor,
                const cuda::strided_t<const in_t, 2> in,
                const cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform average subtraction
                out(record_idx, sample_idx) = { in(record_idx, sample_idx) - sum(record_idx) / divisor, 0.0f };
            }
            template<typename in_t>
            __global__
            static void _demean_and_cast_ss(
                const cuda::strided_t<const float, 1> sum, float divisor,
                const cuda::strided_t<const in_t, 2> in,
                const cuda::strided_t<cuFloatComplex, 2> out
            ) {
                INDEX_DECODE_2D();

                auto offset = in.offset(record_idx, sample_idx);

                // perform average subtraction
                out[offset] = { in[offset] - sum(record_idx) / divisor, 0.0f };
            }
            template<typename in_t>
            __global__
            static void _demean_and_cast(
                const cuda::strided_t<const float, 1> sum, float divisor,
                const cuda::strided_t<const in_t, 2> in,
                const cuda::strided_t<float, 2> out
            ) {
                INDEX_DECODE_2D();

                // perform average subtraction
                out(record_idx, sample_idx) = in(record_idx, sample_idx) - sum(record_idx) / divisor;
            }
            template<typename in_t>
            __global__
            static void _demean_and_cast_ss(
                const cuda::strided_t<const float, 1> sum, float divisor,
                const cuda::strided_t<const in_t, 2> in,
                const cuda::strided_t<float, 2> out
            ) {
                INDEX_DECODE_2D();

                auto offset = in.offset(record_idx, sample_idx);

                // perform average subtraction
                out[offset] = in[offset] - sum(record_idx) / divisor;
            }
            template<typename in_t, typename out_t>
            static void _demean_and_cast_internal(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& sum, float divisor,
                const cuda::strided_t<const in_t, 2>& in,
                const cuda::strided_t<out_t, 2>& out
            ) {
                INDEX_ENCODE_2D();

                if (in.stride == out.stride) {
                    _demean_and_cast_ss<<<blocks, threads, 0, stream.handle()>>>(sum, divisor, in, out);
                } else {
                    _demean_and_cast<<<blocks, threads, 0, stream.handle()>>>(sum, divisor, in, out);
                }
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "demean and cast kernel launch failed");
            }

            void demean_and_cast(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& sum, float divisor,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                _demean_and_cast_internal(stream, sum, divisor, in, out);
            }
            void demean_and_cast(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& sum, float divisor,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<cuFloatComplex, 2>& out
            ) {
                _demean_and_cast_internal(stream, sum, divisor, in, out);
            }
            void demean_and_cast(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& sum, float divisor,
                const cuda::strided_t<const uint16_t, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _demean_and_cast_internal(stream, sum, divisor, in, out);
            }
            void demean_and_cast(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 1>& sum, float divisor,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<float, 2>& out
            ) {
                _demean_and_cast_internal(stream, sum, divisor, in, out);
            }

            //
            // Hilbert windowing
            //

            __global__
            static void _hilbert_window(cuda::strided_t<cuFloatComplex, 2> out) {
                INDEX_DECODE_2D();

                auto& value = out(record_idx, sample_idx);
                auto& samples_per_record = out.shape.y;

                // apply Hilbert window
                if (sample_idx > samples_per_record / 2) {
                    value.x = 0;
                    value.y = 0;
                } else if (sample_idx != 0 && sample_idx != samples_per_record / 2) {
                    value.x *= 2;
                    value.y *= 2;
                }

                // apply normalization for FFT
                value.x /= samples_per_record;
                value.y /= samples_per_record;
            }
            void hilbert_window(const cuda::stream_t& stream, const cuda::strided_t<cuFloatComplex, 2>& out) {
                INDEX_ENCODE_2D();

                _hilbert_window<<<blocks, threads, 0, stream.handle()>>>(out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "Hilbert window kernel launch failed");
            }

            __global__
            static void _phase_differences(
                const cuda::strided_t<const cuFloatComplex, 2> in,
                const cuda::strided_t<float, 2> out
            ) {
                INDEX_DECODE_2D();

                if (sample_idx == 0) {
                    out(record_idx, sample_idx) = 0;
                } else {
                    auto& previous = in(record_idx, sample_idx - 1);
                    auto& current = in(record_idx, sample_idx);

                    // compute and normalize pairwise phase differences between successive samples
                    auto diff = atan2f(current.y, current.x) - atan2f(previous.y, previous.x);
                    if (diff > M_PI_FLOAT) {
                        diff -= 2.0f * M_PI;
                    } else if (diff < -M_PI_FLOAT) {
                        diff += 2.0f * M_PI_FLOAT;
                    }

                    out(record_idx, sample_idx) = diff;
                }
            }
            void phase_differences(
                const cuda::stream_t& stream,
                const cuda::strided_t<const cuFloatComplex, 2>& in,
                const cuda::strided_t<float, 2> out
            ) {
                INDEX_ENCODE_2D();

                _phase_differences<<<blocks, threads, 0, stream.handle()>>>(in, out);
#if defined(VORTEX_SERIALIZE_CUDA_KERNELS)
                cudaDeviceSynchronize();
#endif
                cudaError_t error = cudaGetLastError();
                cuda::detail::handle_error(error, "phase difference kernel launch failed");
            }

            size_t prepare_accumulate_max_phase(
                const cuda::strided_t<uint32_t, 2>& keys
            ) {
#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
                std::array<size_t, 2> scratch_size;
                cudaError_t error;

                error = cuda::cub::DeviceScan::InclusiveScanByKey(
                    nullptr, scratch_size[0],
                    keys.ptr, reinterpret_cast<float*>(0),
                    reinterpret_cast<float*>(0),
                    cuda::cub::Sum(),
                    keys.shape.x * keys.shape.y
                );
                cuda::detail::handle_error(error, "accumulate phase planning failed");

                error = cuda::cub::DeviceReduce::ReduceByKey(
                    nullptr, scratch_size[1],
                    keys.ptr, reinterpret_cast<uint32_t*>(0),
                    reinterpret_cast<float*>(0), reinterpret_cast<float*>(0), reinterpret_cast<uint32_t*>(0),
                    cuda::cub::Max(),
                    keys.shape.x * keys.shape.y
                );
                cuda::detail::handle_error(error, "max phase planning failed");

                return std::max(scratch_size[0], scratch_size[1]);
#else
                throw std::runtime_error("dynamic resampling is not supported");
#endif
            }

            void accumulate_max_phase(
                const cuda::stream_t& stream,
                const cuda::strided_t<const float, 2>& in,
                const cuda::strided_t<const uint32_t, 2>& keys,
                const cuda::strided_t<float, 2>& out_accum,
                const cuda::strided_t<float, 1>& out_max,
                const cuda::strided_t<uint32_t, 1>& out_count,
                void* scratch_ptr, size_t scratch_size
            ) {
#if defined(VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
                cudaError_t error;

                error = cuda::cub::DeviceScan::InclusiveScanByKey(
                    scratch_ptr, scratch_size,
                    keys.ptr, in.ptr, out_accum.ptr,
                    cuda::cub::Sum(),
                    in.shape.x * in.shape.y,
                    cuda::cub::Equality(),
                    stream.handle()
                );
                cuda::detail::handle_error(error, "accumulate records kernel(s) failed");

                // NOTE: the number_of_runs parameter must be a device pointer
                error = cuda::cub::DeviceReduce::ReduceByKey(
                    scratch_ptr, scratch_size,
                    keys.ptr, out_count.ptr,
                    out_accum.ptr, out_max.ptr, out_count.ptr,
                    cuda::cub::Max(),
                    in.shape.x * in.shape.y,
                    stream.handle()
                );
                cuda::detail::handle_error(error, "max phase kernel(s) failed");

#else
                throw std::runtime_error("dynamic resampling is not supported");
#endif
            }

        }
    }
}
