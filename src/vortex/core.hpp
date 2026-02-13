#pragma once

#include <chrono>
#include <array>
#include <limits>
#include <complex>
#include <numbers>

#include <xtensor/core/xshape.hpp>
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include <fmt/format.h>

#include <vortex/util/exception.hpp>

namespace vortex {

    using counter_t = size_t;
    using delta_t = ptrdiff_t;
    using time_t = std::chrono::high_resolution_clock::time_point;

    template<typename T>
    struct range_t {
        std::array<T, 2> val = {
            -std::numeric_limits<T>::infinity(),
            std::numeric_limits<T>::infinity()
        };

        T& min() {
            return val[0];
        }
        const T& min() const {
            return val[0];
        }

        T length() const {
            return max() - min();
        }

        T& max() {
            return val[1];
        }
        const T& max() const {
            return val[1];
        }

        static range_t symmetric(const T& o) {
            return { -std::abs<T>(o), std::abs<T>(o) };
        }

        bool contains(const T& o) const {
            return o >= min() && o <= max();
        };

        bool operator==(const range_t& o) const {
            return val == o.val;
        }
    };

    template<typename T, size_t N>
    using xt_point = xt::xtensor_fixed<double, xt::xshape<N>>;

    template<typename T>
    auto xt_row(T& m, ptrdiff_t r) {
        if (r < 0) {
            r += m.shape(0);
        }
        return xt::view(m, r, xt::all());
    }

    template<typename T>
    auto xt_row(const T& m, ptrdiff_t r) {
        if (r < 0) {
            r += m.shape(0);
        }
        return xt::view(m, r, xt::all());
    }

    template<typename T>
    auto xt_col(T& m, ptrdiff_t c) {
        if (c < 0) {
            c += m.shape(1);
        }
        return xt::view(m, xt::all(), c);
    }

    template<typename T>
    auto xt_col(const T& m, ptrdiff_t c) {
        if (c < 0) {
            c += m.shape(1);
        }
        return xt::view(m, xt::all(), c);
    }

    // ref: https://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c/21470301
    template<typename T>
    T modulo(T a, T b) {
        return a >= 0 ? a % b : (b - std::abs(a%b)) % b;
    }

    // ref: https://codereview.stackexchange.com/questions/51179/absolute-difference-function-for-stdsize-t
    template<typename T>
    T abs_diff(T a, T b) {
        return a > b ? a - b : b - a;
    }

    constexpr auto pi = std::numbers::pi;
    constexpr auto I = std::complex<double>(0, 1);

    std::string to_string(const std::exception& e, size_t level = 0);
    std::string to_string(const std::exception_ptr& error);

    template<typename logger_t, typename... args_t>
    [[ noreturn ]] void raise(std::shared_ptr<logger_t> logger, const std::string& msg, args_t... args) {
#if FMT_VERSION >= 80000
        auto error_msg = fmt::format(fmt::runtime(msg), args...);
#else
        auto error_msg = fmt::format(msg, args...);
#endif
        if (logger) {
            logger->error(error_msg);
            logger->dump_backtrace();
            logger->flush();
        }
        throw traced<std::runtime_error>(error_msg);
    }
    template<typename logger_t, typename... args_t>
    [[ noreturn ]] void raise(std::shared_ptr<logger_t> logger, std::exception_ptr e = {}) {
        if (!e) {
            e = std::current_exception();
        }
        if (!e) {
            try {
                throw traced<std::invalid_argument>("raise called with no exception");
            } catch (const std::invalid_argument&) {
                e = std::current_exception();
            }
        }
        if (logger) {
            logger->error(to_string(e));
            logger->dump_backtrace();
            logger->flush();
        }
        std::rethrow_exception(e);
    }

    template<typename container_t>
    std::string join(const container_t& n, const std::string& sep) {
        if (n.size() == 0) {
            return {};
        } else {
            std::ostringstream s;
            for (auto it = n.begin(); it != n.end(); ++it) {
                if (it != n.begin()) {
                    s << sep;
                }
                s << *it;
            }
            return s.str();
        }
    }

    template<typename container_t>
    std::string shape_to_string(const container_t& n, const std::string& sep = " x ") {
        return join(n, sep);
    }

    template<size_t N, typename T>
    auto to_array(std::vector<T> in) {
        if (in.size() != N) {
            throw traced<std::invalid_argument>("input vector size mismatch");
        }
        std::array<T, N> out;
        std::move(in.begin(), in.end(), out.begin());
        return out;
    }
    template<size_t N, typename T>
    auto to_array(xt::svector<T> in) {
        if (in.size() != N) {
            throw traced<std::invalid_argument>("input vector size mismatch");
        }
        std::array<T, N> out;
        std::move(in.begin(), in.end(), out.begin());
        return out;
    }

    template<typename S1, typename S2>
    bool equal(const S1& a, const S2& b) {
        if (a.size() != b.size()) {
            return false;
        }
        return std::equal(a.begin(), a.end(), b.begin());
    }

    template<size_t N, typename Input>
    auto head(const Input& in) {
        using T = std::decay_t<decltype(*in.begin())>;
        static_assert(N > 0, "head() minimum count is 1");
        if (N > in.size()) {
            throw traced<std::invalid_argument>(fmt::format("input is too small for head: {} < {}", in.size(), N));
        }
        std::array<T, N> out;
        std::copy(in.begin(), in.begin() + N, out.begin());
        return out;
    }

    template<size_t N, typename Input>
    auto tail(const Input& in) {
        using T = std::decay_t<decltype(*in.begin())>;
        static_assert(N > 0, "tail() minimum count is 1");
        if (N > in.size()) {
            throw traced<std::invalid_argument>(fmt::format("input is too small for tail: {} < {}", in.size(), N));
        }
        std::array<T, N> out;
        std::copy(in.end() - N, in.end(), out.begin());
        return out;
    }

    namespace detail {
        template<typename stride_t>
        auto strided_offset(const stride_t& stride, size_t level) {
            return 0;
        }

        template<typename stride_t, typename index_t, typename... indices_t>
        auto strided_offset(const stride_t& stride, size_t level, index_t val, indices_t... vals) {
            return stride[level] * val + strided_offset(stride, level + 1, vals...);
        }
    }

    template<typename stride_t, typename... indices_t>
    auto strided_offset(const stride_t& stride, indices_t... vals) {
        return detail::strided_offset(stride, 0, vals...);
    }

    template<typename S1, typename S2>
    bool shape_is_compatible(const S1& s1, const S2& s2) {
        auto it1 = s1.cbegin();
        auto it2 = s2.cbegin();

        while (it1 != s1.cend() && it2 != s2.cend()) {
            if (it1 == s1.cend()) {
                // the remainder of s2 must be 1s
                if (*it2 != 1) {
                    return false;
                }
                it2++;
            } else if (it2 == s2.cend()) {
                // the remainder of s1 must be 1s
                if (*it1 != 1) {
                    return false;
                }
                it1++;
            } else {
                if (*it1 == *it2) {
                    // shapes match
                    it1++;
                    it2++;
                } else if (*it1 == 1) {
                    // skip shapes of 1
                    it1++;
                } else if (*it2 == 1) {
                    // skip shapes of 1
                    it2++;
                } else {
                    // incompatible
                    return false;
                }
            }
        }

        // no problems found
        return true;
    }

    template<typename S1, typename S2>
    bool stride_is_compatible(const S1& s1, const S2& s2) {
        auto it1 = s1.crbegin();
        auto it2 = s2.crbegin();

        while (it1 != s1.crend() && it2 != s2.crend()) {
            if (*it1 == *it2) {
                // strides match
                it1++;
                it2++;
            } else if (*it1 == 0) {
                // skip strides of 0
                it1++;
            } else if (*it2 == 0) {
                // skip strides of 0
                it2++;
            } else {
                // incompatible
                return false;
            }
        }

        // no problems found
        return true;
    }

    template<typename Shape, typename Stride>
    auto dense_stride(const Shape& shape, Stride& stride) {
        ptrdiff_t n = 1;
        for (size_t i = shape.size(); i != 0; i--) {
            stride[i - 1] = n;
            n *= shape[i - 1];
        }
        return n;
    }
    template<typename Shape>
    auto dense_stride(const Shape& shape) {
        auto stride = shape;
        dense_stride(shape, stride);
        return stride;
    }

    using seconds = std::chrono::duration<double>;

}
