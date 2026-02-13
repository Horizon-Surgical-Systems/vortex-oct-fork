#pragma once

#include <functional>
#include <tuple>
#include <exception>

namespace vortex::io {

    struct null_config_t {

        void validate() {}

    };

    template<typename config_t_>
    class null_io_t {
    public:
        using config_t = config_t_;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    public:
 
        void prepare() {}
        void start() {}
        void stop() {}

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            std::swap(_config, config);
        }

        template<typename... Vs>
        size_t next(size_t count, const std::tuple<Vs...>& streams) {
            return next(0, count, streams);
        }
        template<typename... Vs>
        size_t next(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            return count;
        }

        template<typename... Vs>
        void next_async(size_t count, const std::tuple<Vs...>& streams, callback_t&& callback) {
            next_async(0, count, streams, std::forward<callback_t>(callback));
        }
        template<typename... Vs>
        void next_async(size_t id, size_t count, const std::tuple<Vs...>& streams, callback_t&& callback) {
            std::invoke(callback, count, std::exception_ptr{});
        }

    protected:

        config_t _config;

    };

}
