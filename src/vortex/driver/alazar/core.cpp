#include <vortex/driver/alazar/core.hpp>

#include <AlazarApi.h>

#include <vortex/driver/alazar/db.hpp>

using namespace vortex::alazar;

std::string vortex::alazar::to_string(RETURN_CODE rc) {
    auto msg = AlazarErrorToText(rc);
    auto extended_msg = detail::lookup_error_message(rc, nullptr);
    if (extended_msg) {
        return fmt::format("{}\n{}", msg, extended_msg);
    } else {
        return msg;
    }
}
