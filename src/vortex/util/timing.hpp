#pragma once

#include <numeric>
#include <chrono>
#include <functional>
#include <unordered_map>

#include <fmt/printf.h>

namespace vortex::util {

    struct timing_book_t {
        std::unordered_map<std::string, std::chrono::nanoseconds> book;
        size_t count;

        auto total() const {
            return std::accumulate(book.begin(), book.end(), std::chrono::nanoseconds{}, [](const auto val, const auto& it) {return val + it.second; });
        }

        template<typename stream_t>
        void write(stream_t& stream) const {
            auto total_ = total();
            for (auto [k, v] : book) {
                fmt::print(stream, "{:15s} {:10d} {:6.1f}%\n", k, v.count() / count, 100 * double(v.count()) / total_.count());
            }
            fmt::print(stream, "{:15s} {:10d} {:6.1f}%\n", "total", total_.count() / count, 100.0);
        }
    };

    template<typename clock_t = std::chrono::high_resolution_clock>
    struct stopwatch_t {
        using result_callback_t = std::function<void(std::chrono::nanoseconds)>;

        stopwatch_t(bool start_ = true) {
            if (start_) {
                start();
            }
        }
        template<typename duration_t>
        stopwatch_t(duration_t& accumulator)
            : stopwatch_t([&](auto duration_) { accumulator += duration_; }) {}
        stopwatch_t(result_callback_t&& callback_)
            : callback(std::forward<result_callback_t>(callback_)) {
            start();
        }

        ~stopwatch_t() {
            stop();
            if (callback) {
                std::invoke(callback, duration());
            }
        }

        void start() {
            mark_start = clock_t::now();
        }
        void stop() {
            mark_stop = clock_t::now();
        }

        auto duration() const {
            return mark_stop - mark_start;
        }

        typename clock_t::time_point mark_start, mark_stop;
        result_callback_t callback;
    };

}
