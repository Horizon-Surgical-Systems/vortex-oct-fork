/** \rst

    Object-oriented wrapper around cuFFT

 \endrst */

#pragma once

#include <vector>
#include <numeric>
#include <complex>

#include <cufft.h>

#include <fmt/format.h>

#include <vortex/driver/cuda/types.hpp>
#include <vortex/driver/cuda/runtime.hpp>

#include <vortex/memory/cuda.hpp>

#include <vortex/util/cast.hpp>

namespace vortex::cuda {

    template<typename T>
    class fft_plan_t {
    public:

        fft_plan_t() {

        }

        ~fft_plan_t() {
            destroy();
        }

        fft_plan_t(const fft_plan_t&) = delete;
        fft_plan_t& operator=(const fft_plan_t&) = delete;

        fft_plan_t(fft_plan_t&& o) : fft_plan_t() {
            *this = std::move(o);
        }
        fft_plan_t& operator=(fft_plan_t&& o) {
            destroy();
            _reset();

            std::swap(_plan, o._plan);

            return *this;
        }

        void plan_many(int count, std::vector<int>& n, const stream_t* stream = nullptr) {
            destroy();

            cufftResult_t result;

            // TODO: switch to managed work area
            result = cufftPlanMany(&_plan,
                downcast<int>(n.size()),    // rank
                n.data(),                   // shape
                NULL, 0, 0,                 // input is contiguous
                NULL, 0, 0,                 // output is contiguous
                CUFFT_C2C,
                count                       // batch size
            );
            if (result != CUFFT_SUCCESS) {
                throw exception(fmt::format("planner failed for [{}] C2C: 0x{:x}", shape_to_string(n), cast(result)));
            }

            if (stream) {
                result = cufftSetStream(_plan, stream->handle());
                if (result != CUFFT_SUCCESS) {
                    throw exception(fmt::format("FFT set stream failed: 0x{:x}", cast(result)));
                }
            }
        }

        void forward(const cuda_view_t<cuda::complex<T>>& in, const cuda_view_t<cuda::complex<T>>& out) {
            execute(in, out, true);
        }

        void inverse(const cuda_view_t<cuda::complex<T>>& in, const cuda_view_t<cuda::complex<T>>& out) {
            execute(in, out, false);
        }

        void execute(const cuda_view_t<cuda::complex<T>>& in, const cuda_view_t<cuda::complex<T>>& out, bool forward) {
            if (!_plan) {
                throw exception("attempted to execute invalid plan");
            }

            // require contiguous buffers
            if (!in.is_contiguous()) {
                throw exception("FFT input buffer is not contiguous");
            }
            if (!out.is_contiguous()) {
                throw exception("FFT output buffer is not contiguous");
            }

            auto result = cufftExecC2C(_plan, in.data(), out.data(), forward ? CUFFT_FORWARD : CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                throw exception(fmt::format("FFT execution failed: 0x{:x}", cast(result)));
            }
        }

        void destroy() {
            if (_plan) {
                auto result = cufftDestroy(_plan);
                if (result != CUFFT_SUCCESS) {
                    throw exception(fmt::format("FFT destruction failed: 0x{:x}", cast(result)));
                }
            }

            _reset();
        }

        bool valid() const {
            return _plan != 0;
        }

    protected:

        void _reset() {
            _plan = 0;
        }

        cufftHandle _plan = 0;

    };

}