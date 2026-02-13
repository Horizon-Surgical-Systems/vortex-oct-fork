#include <vortex/core.hpp>

// ref: https://en.cppreference.com/w/cpp/error/rethrow_if_nested
std::string vortex::to_string(const std::exception& e, size_t level) {
    std::string msg = std::string(" ", level) + e.what();
    try {
        std::rethrow_if_nested(e);
    } catch (const std::exception& e) {
        msg += "\n" + to_string(e, level + 1);
    } catch (...) {
        msg += "\nunknown exception";
    }
    return msg;
}

std::string vortex::to_string(const std::exception_ptr& error) {
    if (error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& e) {
            return to_string(e);
        } catch (...) {
            return "unknown exception";
        }
    }

    return "";
}
