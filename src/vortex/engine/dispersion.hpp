#pragma once

#include <complex>

#include <xtensor/containers/xtensor.hpp>

namespace vortex::engine {

    template<typename T>
    xt::xtensor<std::complex<T>, 1> dispersion_phasor(size_t length, std::array<T, 2> coeff) {
        auto index = xt::cast<T>(xt::arange<ptrdiff_t>(length));
        auto offset = index - T(length / 2.0);
        auto phase = coeff[0] * xt::square(offset) + coeff[1] * xt::cube(offset);
        auto j = std::complex<T>{ 0, 1 };
        return xt::exp(j * phase);
    }

}