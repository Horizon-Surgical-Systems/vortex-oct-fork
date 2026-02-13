#pragma once

#include <spdlog/spdlog.h>

#include <vortex/process/base.hpp>

namespace vortex::process {

    template<typename config_t_>
    class null_processor_t : public processor_t<config_t_> {
    public:

        using base_t = processor_t<config_t_>;
        using config_t = config_t_;

        null_processor_t(std::shared_ptr<spdlog::logger> log = nullptr) {}

        virtual void initialize(config_t config) {
            std::swap(_config, config);
        }

    protected:

        using base_t::_config;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto processor(std::shared_ptr<vortex::process::null_processor_t<vortex::process::processor_config_t<Args...>>> a) {
        using adapter = adapter<block_t>;
        auto w = typename adapter::processor(a.get());

        w.device = []() -> std::optional<cuda::device_t> { return {}; };
        w.stream_factory = []() {
            return []() -> typename adapter::ascan_stream_t {
                return sync::lockable<cuda::cuda_host_tensor_t<typename block_t::process_element_t>>();
            };
        };

        w.input_shape = [a]() { return a->config().input_shape(); };
        w.output_shape = []() -> typename adapter::shape_t { return { 0 }; };

        w.channel = [a]() { return a->config().channel; };

        w.next_async = [](block_t& block,
            const typename adapter::spectra_stream_t& input_stream_, typename adapter::ascan_stream_t& output_stream,
            cuda::event_t* start, cuda::event_t* done, typename adapter::processor::callback_t&& callback) {
            std::invoke(callback, std::exception_ptr());
        };

        return w;
    }
}

#endif
