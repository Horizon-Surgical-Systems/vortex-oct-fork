#pragma once

#include <type_traits>

#include <fmt/format.h>

#include <AlazarError.h>

auto inline format_as(RETURN_CODE rc) { return static_cast<std::underlying_type_t<decltype(rc)>>(rc); }

namespace vortex::alazar {

    class exception : public std::runtime_error {
    public:
        using runtime_error::runtime_error;
    };
    class buffer_overflow : public exception {
    public:
        using exception::exception;
    };
    class buffer_not_ready : public exception {
    public:
        using exception::exception;
    };
    class wait_timeout : public exception {
    public:
        using exception::exception;
    };
    class unsupported_operation : public exception {
    public:
        using exception::exception;
    };

    std::string to_string(RETURN_CODE rc);

    namespace detail {
        template<typename... Args> const
        void handle_error(RETURN_CODE rc, const std::string& msg, Args... args) {
            if (rc == ApiSuccess) {
                return;
            }

            // emit an exception
#if FMT_VERSION >= 80000
            auto user_msg = fmt::format(fmt::runtime(msg), args...);
#else
            auto user_msg = fmt::format(msg, args...);
#endif
            auto error_msg = fmt::format("{}: ({}) {}", user_msg, rc, to_string(rc));
            if (rc == ApiWaitTimeout) {
                throw wait_timeout(error_msg);
            } else if (rc == ApiBufferOverflow) {
                throw buffer_overflow(error_msg);
            } else if (rc == ApiUnsupportedFunction || rc == ApiOCTIgnoreBadClockNotSupported) {
                throw unsupported_operation(error_msg);
            } else if (rc == ApiBufferNotReady) {
                throw buffer_not_ready(error_msg);
            } else {
                throw exception(error_msg);
            }
        }
    }

}
