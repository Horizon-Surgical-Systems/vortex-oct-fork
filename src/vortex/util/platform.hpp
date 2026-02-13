#pragma once

#include <string>
#include <functional>
#include <optional>

// ref: https://stackoverflow.com/questions/5919996/how-to-detect-reliably-mac-os-x-ios-linux-windows-in-c-preprocessor
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define VORTEX_PLATFORM_WINDOWS
#elif __linux__
#  define VORTEX_PLATFORM_LINUX
//#elif __unix__ // all unices not caught above
//// Unix
//#elif defined(_POSIX_VERSION)
//// POSIX
#else
#  error "Unsupported platform"
#endif

namespace vortex {

    using interrupt_callback_t = std::function<void()>;

    void setup_keyboard_interrupt();
    void setup_keyboard_interrupt(interrupt_callback_t&& callback);
    bool check_keyboard_interrupt();

    void setup_realtime();
    void setup_tcp_low_latency();

    void set_thread_name(const std::string& name);

    std::string error_message_with_number();
    std::string error_message_with_number(unsigned long error);
    std::string error_message();
    std::string error_message(unsigned long error);

    std::optional<std::string> envvar(const std::string& name);

}
