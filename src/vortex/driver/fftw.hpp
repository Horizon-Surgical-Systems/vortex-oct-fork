#pragma once

#include <complex>

#include <fftw3.h>

#include <fmt/format.h>

#include <vortex/util/cast.hpp>

namespace vortex::fft {

    namespace detail {

        template<typename T>
        struct fftw_complex_t {};

        template<>
        struct fftw_complex_t<float> { using type = fftwf_complex; };
        template<>
        struct fftw_complex_t<double> { using type = fftw_complex; };
        template<>
        struct fftw_complex_t<long double> { using type = fftwl_complex; };

        template<typename T>
        struct fftw_type_t {};

        template<>
        struct fftw_type_t<std::complex<float>> { using type = fftw_complex_t<float>::type; };
        template<>
        struct fftw_type_t<std::complex<double>> { using type = fftw_complex_t<double>::type; };
        template<>
        struct fftw_type_t<std::complex<long double>> { using type = fftw_complex_t<long double>::type; };

        template<typename T>
        struct fftw_api {};

        template<>
        struct fftw_api<float> {
            using plan = fftwf_plan;
            static constexpr auto plan_many_dft = &fftwf_plan_many_dft;
            static constexpr auto execute_dft = &fftwf_execute_dft;
            static constexpr auto destroy_plan = &fftwf_destroy_plan;
        };
        template<>
        struct fftw_api<double> {
            using plan = fftw_plan;
            static constexpr auto plan_many_dft = &fftw_plan_many_dft;
            static constexpr auto execute_dft = &fftw_execute_dft;
            static constexpr auto destroy_plan = &fftw_destroy_plan;
        };
        template<>
        struct fftw_api<long double> {
            using plan = fftwl_plan;
            static constexpr auto plan_many_dft = &fftwl_plan_many_dft;
            static constexpr auto execute_dft = &fftwl_execute_dft;
            static constexpr auto destroy_plan = &fftwl_destroy_plan;
        };

    }

    template<typename T>
    using complex = typename detail::fftw_complex_t<T>::type;

    template<typename T>
    using fftw_type = typename detail::fftw_type_t<T>::type;

    template<typename T>
    class fftw_plan_t {
    public:
        using real_t = T;
        using complex_t = std::complex<T>;

        fftw_plan_t() {

        }

        ~fftw_plan_t() {
            destroy();
        }

        fftw_plan_t(const fftw_plan_t&) = delete;
        fftw_plan_t& operator=(const fftw_plan_t&) = delete;

        fftw_plan_t(fftw_plan_t&& o) : fftw_plan_t() {
            *this = std::move(o);
        }
        fftw_plan_t& operator=(fftw_plan_t&& o) {
            destroy();
            _reset();

            std::swap(_plan, o._plan);

            return *this;
        }

        void forward(int count, const std::pair<int, int> stride, const std::pair<int, int>& offset, const std::vector<int>& n, complex_t* in, complex_t* out, int flags = FFTW_PRESERVE_INPUT | FFTW_MEASURE) {
            create(true, count, stride, offset, n, in, out, flags);
        }
        void inverse(int count, const std::pair<int, int> stride, const std::pair<int, int>& offset, const std::vector<int>& n, complex_t* in, complex_t* out, int flags = FFTW_PRESERVE_INPUT | FFTW_MEASURE) {
            create(false, count, stride, offset, n, in, out, flags);
        }

        void create(bool forward, int count, const std::pair<int, int> stride, const std::pair<int, int>& offset, const std::vector<int>& n, complex_t* in, complex_t* out, int flags = FFTW_PRESERVE_INPUT | FFTW_MEASURE) {
            destroy();
            _plan = detail::fftw_api<T>::plan_many_dft(downcast<int>(n.size()), n.data(), count, reinterpret_cast<fftw_type<complex_t>*>(in), NULL, stride.first, offset.first, reinterpret_cast<fftw_type<complex_t>*>(out), NULL, stride.second, offset.second, forward ? FFTW_FORWARD : FFTW_BACKWARD, flags);
            if (!_plan) {
                throw std::runtime_error(fmt::format("planner failed for c2c: {}", _shape_to_string(n)));
            }
        }

        void execute() {
            if (!_plan) {
                throw std::runtime_error("attempted to execute invalid plan");
            }

            fftw_execute(_plan);
        }

        void execute(complex_t* in, complex_t* out) {
            if (!_plan) {
                throw std::runtime_error("attempted to execute invalid plan");
            }

            detail::fftw_api<T>::execute_dft(_plan, reinterpret_cast<fftw_type<complex_t>*>(in), reinterpret_cast<fftw_type<complex_t>*>(out));
        }

        void destroy() {
            if (_plan) {
                detail::fftw_api<T>::destroy_plan(_plan);
            }

            _reset();
        }

        bool valid() const {
            return _plan != nullptr;
        }

    protected:

        std::string _shape_to_string(const std::vector<int>& n) {
            return std::accumulate(n.begin(), n.end(), std::string(), [](const std::string& a, int b) {
                if (a.empty()) {
                    return fmt::format("{}", b);
                } else {
                    return fmt::format("{} x {}", a, b);
                }
            });
        }

        void _reset() {
            _plan = nullptr;
        }

        typename detail::fftw_api<T>::plan _plan = nullptr;

    };

}
