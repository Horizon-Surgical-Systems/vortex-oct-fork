#pragma once

#include <fmt/format.h>

#if defined(VORTEX_ENABLE_BACKWARD)
#  include <backward.hpp>
#endif

namespace vortex {

    struct tracer {

        tracer();

        std::string format() const;

    protected:

#if defined(VORTEX_ENABLE_BACKWARD)
        backward::StackTrace _st;
#endif

    };

    template<typename T>
    struct traced : T, tracer {
        using T::T;
    };

    template<typename T>
    std::string check_trace(const T& obj) {
        auto tr = dynamic_cast<const tracer*>(&obj);
        if (tr) {
            return tr->format();
        } else {
            return "--- no stack trace ---";
        }
    }

}