#pragma once

#include <numeric>
#include <complex>
#include <array>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <fmt/format.h>

#if defined(__CUDACC__)
#  define DEV_OR_HOST __device__ __host__
# else
#  define DEV_OR_HOST
#endif

inline auto format_as(cudaError_t e) { return static_cast<std::underlying_type_t<decltype(e)>>(e); }

DEV_OR_HOST inline auto operator==(const longlong1& a, const longlong1& b) { return a.x == b.x; }
DEV_OR_HOST inline auto operator==(const longlong2& a, const longlong2& b) { return a.x == b.x && a.y == b.y; }
DEV_OR_HOST inline auto operator==(const longlong3& a, const longlong3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
DEV_OR_HOST inline auto operator==(const longlong4& a, const longlong4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

DEV_OR_HOST inline auto operator==(const ulonglong1& a, const ulonglong1& b) { return a.x == b.x; }
DEV_OR_HOST inline auto operator==(const ulonglong2& a, const ulonglong2& b) { return a.x == b.x && a.y == b.y; }
DEV_OR_HOST inline auto operator==(const ulonglong3& a, const ulonglong3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
DEV_OR_HOST inline auto operator==(const ulonglong4& a, const ulonglong4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }


// NOTE: no namespace shorthand here so NVCC can include this header
namespace vortex {
    namespace cuda {

        class exception : public std::runtime_error {
        public:
            using runtime_error::runtime_error;
        };

        namespace detail {

            template<typename T>
            struct cuda_complex_t {};
            template<> struct cuda_complex_t<float> { using type = cuFloatComplex; };
            template<> struct cuda_complex_t<double> { using type = cuDoubleComplex; };

            template<typename T>
            struct cuda_type_t { using type = T; };
            template<> struct cuda_type_t<std::complex<float>> { using type = cuda_complex_t<float>::type; };
            template<> struct cuda_type_t<std::complex<double>> { using type = cuda_complex_t<double>::type; };

            template<typename... Args>
            void handle_error(cudaError_t error, const std::string& msg, Args... args) {
                if (!error) {
                    return;
                }

                // emit an exception
#if FMT_VERSION >= 80000
                auto user_msg = fmt::format(fmt::runtime(msg), args...);
#else
                auto user_msg = fmt::format(msg, args...);
#endif
                auto error_msg = fmt::format("{}: ({}) {}", user_msg, error, cudaGetErrorName(error));
                throw exception(error_msg);
            }
        }

        template<typename T>
        using complex = typename detail::cuda_complex_t<T>::type;

        template<typename T>
        using device_type = typename detail::cuda_type_t<T>::type;

        template<typename T, size_t N, size_t M>
        struct matrix_t {
            T v[N][M];

            std::array<size_t, 2> shape() const {
                return { N, M };
            }
            auto count() const {
                return N * M;
            }
        };

        namespace detail {

            template<size_t N>
            struct longlong {};
            template<> struct longlong<1> { using type = longlong1; };
            template<> struct longlong<2> { using type = longlong2; };
            template<> struct longlong<3> { using type = longlong3; };
            template<> struct longlong<4> { using type = longlong4; };

            template<size_t N>
            struct ulonglong {};
            template<> struct ulonglong<1> { using type = ulonglong1; };
            template<> struct ulonglong<2> { using type = ulonglong2; };
            template<> struct ulonglong<3> { using type = ulonglong3; };
            template<> struct ulonglong<4> { using type = ulonglong4; };

        }

        inline ulonglong1 to_vec(const std::array<size_t, 1>& src) { return { src[0] }; }
        inline ulonglong2 to_vec(const std::array<size_t, 2>& src) { return { src[0], src[1] }; }
        inline ulonglong3 to_vec(const std::array<size_t, 3>& src) { return { src[0], src[1], src[2] }; }
        inline ulonglong4 to_vec(const std::array<size_t, 4>& src) { return { src[0], src[1], src[2], src[3] }; }

        inline longlong1 to_vec(const std::array<intmax_t, 1>& src) { return { src[0] }; }
        inline longlong2 to_vec(const std::array<intmax_t, 2>& src) { return { src[0], src[1] }; }
        inline longlong3 to_vec(const std::array<intmax_t, 3>& src) { return { src[0], src[1], src[2] }; }
        inline longlong4 to_vec(const std::array<intmax_t, 4>& src) { return { src[0], src[1], src[2], src[3] }; }

        template<typename T, size_t N>
        struct strided_t {
            using shape_t = typename detail::ulonglong<N>::type;
            using stride_t = typename detail::longlong<N>::type;

            T* ptr;
            shape_t shape;
            stride_t stride;

            DEV_OR_HOST inline auto& operator[](ptrdiff_t idx) const { return ptr[idx]; }

            DEV_OR_HOST inline auto offset(size_t idx0) const { return stride.x * idx0; }
            DEV_OR_HOST inline auto offset(size_t idx0, size_t idx1) const { return stride.x * idx0 + stride.y * idx1; }
            DEV_OR_HOST inline auto offset(size_t idx0, size_t idx1, size_t idx2) const { return stride.x * idx0 + stride.y * idx1 + stride.z * idx2; }
            DEV_OR_HOST inline auto offset(size_t idx0, size_t idx1, size_t idx2, size_t idx3) const { return stride.x * idx0 + stride.y * idx1 + stride.z * idx2 + stride.w * idx3; }

            DEV_OR_HOST inline auto& operator()(size_t idx0) const { return (*this)[offset(idx0)]; }
            DEV_OR_HOST inline auto& operator()(size_t idx0, size_t idx1) const { return (*this)[offset(idx0, idx1)]; }
            DEV_OR_HOST inline auto& operator()(size_t idx0, size_t idx1, size_t idx2) const { return (*this)[offset(idx0, idx1, idx2)]; }
            DEV_OR_HOST inline auto& operator()(size_t idx0, size_t idx1, size_t idx2, size_t idx3) const { return (*this)[offset(idx0, idx1, idx2, idx3)]; }

            strided_t(T* ptr_, shape_t&& shape_, stride_t&& stride_)
                : ptr(ptr_), shape(std::forward<shape_t>(shape_)), stride(std::forward<stride_t>(stride_)) {}

            strided_t(T* ptr_, const std::array<size_t, N>& shape_, const std::array<ptrdiff_t, N>& stride_)
                : strided_t(ptr_, to_vec(shape_), to_vec(stride_)) {}

        };

    }
}

#undef DEV_OR_HOST
