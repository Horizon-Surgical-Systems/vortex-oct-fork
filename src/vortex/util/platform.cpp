#define _CRT_SECURE_NO_WARNINGS

#include <vortex/util/platform.hpp>

#include <stdexcept>

#if defined(VORTEX_PLATFORM_WINDOWS)
#  include <Windows.h>
#endif

#include <signal.h>
#include <string.h>

#include <fmt/format.h>

namespace vortex {

    static volatile sig_atomic_t _keyboard_interrupt = 0;
    static std::function<void()> _keyboard_callback;

    bool check_keyboard_interrupt() {
        bool flag = (_keyboard_interrupt != 0);
        _keyboard_interrupt = 0;
        return flag;
    }

#if defined(VORTEX_PLATFORM_WINDOWS)
    BOOL WINAPI _signal_handler(DWORD dwCtrlType) {
        if(dwCtrlType == CTRL_C_EVENT) {
            _keyboard_interrupt = 1;
            if (_keyboard_callback) {
                _keyboard_callback();
            }
            return true;
        } else {
            return false;
        }
    }

    void setup_keyboard_interrupt() {
        if(!SetConsoleCtrlHandler(&_signal_handler, TRUE)) {
            throw std::runtime_error(fmt::format("SetConsoleCtrlHandler failed: {}", error_message_with_number()));
        }
    }

    void setup_keyboard_interrupt(interrupt_callback_t&& callback) {
        _keyboard_callback = std::forward<interrupt_callback_t>(callback);
        setup_keyboard_interrupt();
    }

    void setup_tcp_low_latency() {
        // not yet supported on Windows
    }

    void setup_realtime() {
        if(!::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
            throw std::runtime_error(fmt::format("SetThreadPriority failed: {}", error_message_with_number()));
        }
        if(!::SetPriorityClass(::GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
            throw std::runtime_error(fmt::format("SetPriorityClass failed: {}", error_message_with_number()));
        }
    }

    // ref: https://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push,8)
    typedef struct tagTHREADNAME_INFO
    {
        DWORD dwType; // Must be 0x1000.
        LPCSTR szName;// Pointer to name (in user addr space).
        DWORD dwThreadID;// Thread ID (-1=caller thread).
        DWORD dwFlags;// Reserved for future use, must be zero.
    }THREADNAME_INFO;
#pragma pack(pop)

    void set_thread_name(const std::string& name) {
        THREADNAME_INFO info;
        info.dwType = 0x1000;
        info.szName = name.c_str();
        info.dwThreadID = -1;
        info.dwFlags = 0;

        __try {
            ::RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
        }__except(EXCEPTION_EXECUTE_HANDLER) {
        }
    }

    std::string error_message_with_number() {
        return error_message_with_number(::GetLastError());
    }
    std::string error_message_with_number(unsigned long error) {
        return fmt::format("{} (0x{:08x})", error_message(error), error);
    }

    std::string error_message() {
        return error_message(::GetLastError());
    }

    // ref: http://stackoverflow.com/questions/1387064/how-to-get-the-error-message-from-the-error-code-returned-by-getlasterror
    std::string error_message(unsigned long error) {
        LPTSTR buffer = NULL;
        size_t size = ::FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            error,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&buffer,
            0,
            NULL
        );

        std::string msg(buffer, size);

        // trim whitespace
        // ref: https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
        msg.erase(std::find_if(msg.rbegin(), msg.rend(), [](char c) {return !std::isspace(c); }).base(), msg.end());

        // free the buffer allocated by FormatMessageA
        ::HeapFree(::GetProcessHeap(), 0, buffer);

        return msg;
    }

#elif defined(VORTEX_PLATFORM_LINUX)

    void _signal_handler(int sig) {
        if(sig == SIGINT) {
            _keyboard_interrupt = 1;
        }
    }

    void setup_keyboard_interrupt() {
        if(signal(SIGINT, _signal_handler) == SIG_ERR) {
            throw std::runtime_error(fmt::format("installing signal handler failed: {} ({})", ::strerror(errno), errno));
        }
    }

    void setup_tcp_low_latency() {
        FILE* f = ::fopen("/proc/sys/net/ipv4/tcp_low_latency", "w");
        if(f == NULL) {
            throw std::runtime_error(fmt::format("failed opening /proc/sys/net/ipv4/tcp_low_latency: {} ({})", ::strerror(errno), errno));
        }

        auto result = ::fputs("1", f);
        auto error = errno;
        ::fclose(f);

        if(result < 0) {
            throw std::runtime_error(fmt::format("failed writing /proc/sys/net/ipv4/tcp_low_latency: {} ({})", ::strerror(error), error));
        }
    }

#define REALTIME_PRIORITY 99

    void setup_realtime() {
        sched_param param;
        param.sched_priority = REALTIME_PRIORITY;
        // ref: http://www.drdobbs.com/soft-real-time-programming-with-linux/184402031
        ::pthread_setschedparam(::pthread_self(), SCHED_FIFO, &param);
    }

    void set_thread_name(const std::string& name) {
        ::pthread_setname_np(::pthread_self(), name.c_str());
    }

#endif

    std::optional<std::string> envvar(const std::string& name) {
        // check variable and immediately copy if present
#pragma warning(suppress : 4996)
        auto var = std::getenv(name.c_str());
        if (var) {
            return { var };
        } else {
            return {};
        }
    }

}
