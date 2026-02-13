/** \rst

    Utility functions for use from CUDA kernels

    CUDA code often has repeated idioms to calculate the size of the
    grid and blocks of the execution configuration. We provide some
    sane implementations of those here to avoid code duplication.

 \endrst */

#pragma once

#include <type_traits>
#include <limits>

#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <vortex/util/cast.hpp>

namespace vortex {
    namespace cuda {
        namespace kernel {

            inline dim3 threads_from_shape(size_t x) {
                return {
                    static_cast<unsigned int>(std::min(size_t(1024), x)),
                    1u,
                    1u
                };
            }
            inline dim3 threads_from_shape(const ulonglong1& shape) {
                return threads_from_shape(shape.x);
            }
            inline dim3 threads_from_shape(size_t x, size_t y) {
                return {
                    static_cast<unsigned int>(std::min(size_t(32), x)),
                    static_cast<unsigned int>(std::min(size_t(32), y)),
                    1u
                };
            }
            inline dim3 threads_from_shape(const ulonglong2& shape) {
                return threads_from_shape(shape.x, shape.y);
            }
            inline dim3 threads_from_shape(size_t x, size_t y, size_t z) {
                return {
                    static_cast<unsigned int>(std::min(size_t(16), x)),
                    static_cast<unsigned int>(std::min(size_t(16), y)),
                    static_cast<unsigned int>(std::min(size_t(4),  z)),
                };
            }
            inline dim3 threads_from_shape(const ulonglong3& shape) {
                return threads_from_shape(shape.x, shape.y, shape.z);
            }

            inline dim3 blocks_from_threads(const dim3& threads, size_t x) {
                return {
                    downcast<unsigned int>(std::ceil(x / double(threads.x))),
                    1u,
                    1u
                };
            }
            inline dim3 blocks_from_threads(const dim3& threads, const ulonglong1& shape) {
                return blocks_from_threads(threads, shape.x);
            }
            inline dim3 blocks_from_threads(const dim3& threads, size_t x, size_t y) {
                return {
                    downcast<unsigned int>(std::ceil(x / double(threads.x))),
                    downcast<unsigned int>(std::ceil(y / double(threads.y))),
                    1u
                };
            }
            inline dim3 blocks_from_threads(const dim3& threads, const ulonglong2& shape) {
                return blocks_from_threads(threads, shape.x, shape.y);
            }
            inline dim3 blocks_from_threads(const dim3& threads, size_t x, size_t y, size_t z) {
                return {
                    downcast<unsigned int>(std::ceil(x / double(threads.x))),
                    downcast<unsigned int>(std::ceil(y / double(threads.y))),
                    downcast<unsigned int>(std::ceil(z / double(threads.z))),
                };
            }
            inline dim3 blocks_from_threads(const dim3& threads, const ulonglong3& shape) {
                return blocks_from_threads(threads, shape.x, shape.y, shape.z);
            }

            template<typename in_t>
            __host__ __device__ in_t sqr(const in_t& x) {
                return x * x;
            }

            template<typename out_t, typename in_t>
            __host__ __device__ out_t log10(const in_t& x) {
                return std::log10(out_t(x));
            }

            template<typename in_t, typename = std::enable_if_t<(std::is_integral<in_t>::value || std::is_floating_point<in_t>::value) && std::is_signed<in_t>::value>>
            __host__ __device__
            inline in_t abs(const in_t& in) {
                return std::abs(in);
            }
            template<typename in_t, typename = std::enable_if_t<std::is_integral<in_t>::value || std::is_floating_point<in_t>::value>, typename = std::enable_if_t<!std::is_signed<in_t>::value>>
            __host__ __device__
            inline in_t abs(const in_t& in) {
                return in;
            }
            __host__ __device__
            inline float abs(const cuFloatComplex& in) {
                return cuCabsf(in);
            }
            __host__ __device__
            inline double abs(const cuDoubleComplex& in) {
                return cuCabs(in);
            }

            template<typename in_t, typename = std::enable_if_t<(std::is_integral<in_t>::value || std::is_floating_point<in_t>::value) && std::is_signed<in_t>::value>>
            __host__ __device__
                inline in_t real(const in_t& in) {
                return in;
            }
            template<typename in_t, typename = std::enable_if_t<std::is_integral<in_t>::value || std::is_floating_point<in_t>::value>, typename = std::enable_if_t<!std::is_signed<in_t>::value>>
            __host__ __device__
                inline in_t real(const in_t & in) {
                return in;
            }
            __host__ __device__
                inline float real(const cuFloatComplex& in) {
                return cuCrealf(in);
            }
            __host__ __device__
                inline double real(const cuDoubleComplex& in) {
                return cuCreal(in);
            }

            __host__ __device__
            inline float floor(float x) {
                return ::floorf(x);
            }
            __host__ __device__
            inline double floor(double x) {
                return ::floor(x);
            }
            __host__ __device__
            inline float ceil(float x) {
                return ::ceilf(x);
            }
            __host__ __device__
            inline double ceil(double x) {
                return ::ceil(x);
            }
            __host__ __device__
                inline float round(float x) {
                return ::roundf(x);
            }
            __host__ __device__
                inline double round(double x) {
                return ::round(x);
            }

            template<typename out_t, typename in_t, typename = std::enable_if_t<std::is_same<in_t, out_t>::value>>
            __host__ __device__
            inline out_t round_clip_cast(const in_t& in) {
                // pass through since type is unchanged
                return in;
            }
            template<typename out_t, typename in_t, typename = std::enable_if_t<!std::is_same<in_t, out_t>::value>, typename = std::enable_if_t<std::is_floating_point<in_t>::value && std::is_integral<out_t>::value>>
            __host__ __device__
            inline out_t round_clip_cast(const in_t& in) {
                // round first to avoid bounds issues later
                auto val = round(in);

                // clip to output limits
                // NOTE: do comparison in floating point to avoid underflow/overflow of integer
                if (val > static_cast<in_t>(std::numeric_limits<out_t>::max())) {
                    return std::numeric_limits<out_t>::max();
                } else if (val < static_cast<in_t>(std::numeric_limits<out_t>::lowest())) {
                    return std::numeric_limits<out_t>::lowest();
                } else {
                    return static_cast<out_t>(val);
                }
            }

            template<typename out_t, typename in_t, typename = std::enable_if_t<std::is_same<in_t, out_t>::value>>
            __host__ __device__
            inline out_t floor_clip_cast(const in_t& in) {
                // pass through since type is unchanged
                return in;
            }
            template<typename out_t, typename in_t, typename = std::enable_if_t<!std::is_same<in_t, out_t>::value>, typename = std::enable_if_t<std::is_floating_point<in_t>::value && std::is_integral<out_t>::value>>
            __host__ __device__
            inline out_t floor_clip_cast(const in_t& in) {
                // floor first to avoids bound issues later
                auto val = static_cast<intmax_t>(floor(in));

                // clip to output limits
                // NOTE: do comparison in floating point to avoid underflow/overflow of integer
                if (val > static_cast<in_t>(std::numeric_limits<out_t>::max())) {
                    return std::numeric_limits<out_t>::max();
                } else if (val < static_cast<in_t>(std::numeric_limits<out_t>::lowest())) {
                    return std::numeric_limits<out_t>::lowest();
                } else {
                    return static_cast<out_t>(val);
                }
            }

            //template<typename out_t, typename in_t, typename = std::enable_if_t<!std::is_same_v<in_t, out_t>>, typename = std::enable_if_t<std::is_floating_point_v<out_t> && std::is_integral_v<in_t>>, typename = void>
            //__host__ __device__
            //inline out_t _round_clip_cast(const in_t& in) {
            //    // just cast directly
            //    return static_cast<out_t>(in);
            //}
            //template<typename out_t, typename in_t, typename = std::enable_if_t<std::is_same_v<in_t, double> && std::is_same_v<out_t, float>>>
            //__host__ __device__
            //inline out_t _round_clip_cast(const in_t& in) {
            //    // just cast directly
            //    return static_cast<double>(in);
            //}

        }
    }
}
