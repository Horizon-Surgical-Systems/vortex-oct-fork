#pragma once

#include <thread>
#include <functional>
#include <optional>

#include <spdlog/spdlog.h>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/copy.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/thread.hpp>

namespace vortex::process {

    struct copy_processor_config_t {
        std::array<size_t, 3> input_shape() const { return { ascans_per_block(), samples_per_record(), channels_per_sample() }; }

        size_t& records_per_block() { return _ascans_per_block; }
        const size_t& records_per_block() const { return _ascans_per_block; }

        size_t& samples_per_record() { return _samples_per_record; }
        const size_t& samples_per_record() const { return _samples_per_record; }

        size_t& channels_per_sample() { return _channels_per_sample; }
        const size_t& channels_per_sample() const { return _channels_per_sample; }

        std::array<size_t, 3> output_shape() const { return { ascans_per_block(), samples_per_ascan(), 1 }; }

        size_t& ascans_per_block() { return _ascans_per_block; }
        const size_t& ascans_per_block() const { return _ascans_per_block; }

        size_t samples_per_ascan() const {
            return copy::slice::to_simple(sample_slice, samples_per_record()).count();
        }

        copy::slice_t sample_slice;
        copy::transform_t sample_transform;

        size_t slots = 2;

        size_t channel = 0;

        virtual void validate() const {}

    protected:

        size_t _ascans_per_block = 1000;
        size_t _samples_per_record = 1000;
        size_t _channels_per_sample = 1;

    };

    template<typename input_element_t_, typename output_element_t_, typename config_t_>
    class copy_processor_t {
    public:

        using base_t = processor_t<config_t_>;
        using config_t = config_t_;

        using input_element_t = input_element_t_;
        using output_element_t = output_element_t_;

        using callback_t = std::function<void(std::exception_ptr)>;

    protected:

        using job_t = std::function<void()>;

    public:

        copy_processor_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) {}

        virtual ~copy_processor_t() {
            if (_pool) {
                _pool->wait_finish();
            }
        }

        virtual void initialize(config_t config) {
            if (_log) { _log->debug("initializing CPU copy processor"); }

            config.validate();

            std::swap(_config, config);

            // launch worker pool
            _pool.emplace("CPU Copy", _config.slots, [](size_t) { setup_realtime(); }, _log);
        }

        virtual void change(config_t new_config) {
            new_config.validate();

            std::swap(_config, new_config);
        }

        const auto& config() const {
            return _config;
        }

        template<typename V1, typename V2>
        void next(const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer) {
            next(0, input_buffer, output_buffer);
        }
        template<typename V1, typename V2>
        void next(size_t id, const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer) {
            std::unique_lock<std::mutex> lock(_mutex);

            std::exception_ptr error;
            sync::event_t done;

            // schedule processing
            _next_async(id, input_buffer, output_buffer, [&](std::exception_ptr error_) {
                error = std::move(error_);
                done.set();
            });

            // wait for completion
            done.wait();
            if (error) {
                std::rethrow_exception(error);
            }
        }

        template<typename V1, typename V2>
        void next_async(const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, callback_t&& callback) {
            std::unique_lock<std::mutex> lock(_mutex);
            _next_async(id, input_buffer, output_buffer, std::forward<callback_t>(callback));
        }

    protected:

        template<typename V1, typename V2>
        void _next_async(size_t id, const cpu_viewable<V1>& input_buffer_, const cpu_viewable<V2>& output_buffer_, callback_t&& callback) {

            const auto& input_buffer = input_buffer_.derived_cast();
            const auto& output_buffer = output_buffer_.derived_cast();

            _pool->post([this, id, input_buffer, output_buffer]() {

                std::exception_ptr error;
                try {
                    // perform the processing
                    _process(id, input_buffer, output_buffer);
                } catch (const std::exception&) {
                    error = std::current_exception();
                    if (_log) { _log->error("error during processing for block {}: {}", id, to_string(error)); }
                }
                return std::make_tuple(error);

            }, std::forward<callback_t>(callback));

        }

        template<typename V1, typename V2>
        auto _process(size_t id, const cpu_viewable<V1>& input_buffer_, const cpu_viewable<V2>& output_buffer_) {
            const auto& input_buffer = input_buffer_.derived_cast();
            const auto& output_buffer = output_buffer_.derived_cast();

            if (_log) { _log->trace("processing block {}", id); }

            auto count = std::min(input_buffer.shape(0), output_buffer.shape(0));

            // determine if this block length is acceptable
            if (count > _config.records_per_block()) {
                throw std::runtime_error(fmt::format("block is larger than maximum configured size: {} > {}", count, _config.records_per_block()));
            }

            // check that buffers are appropriate shape
            if (!shape_is_compatible(input_buffer.shape(), _config.input_shape())) {
                throw std::runtime_error(fmt::format("input stream shape is not compatible with configured input shape: {} !~= {}", shape_to_string(input_buffer.shape()), shape_to_string(_config.input_shape())));
            }
            if (!shape_is_compatible(output_buffer.shape(), _config.output_shape())) {
                throw std::runtime_error(fmt::format("output stream shape is not compatible with configured output shape: {} !~= {}", shape_to_string(output_buffer.shape()), shape_to_string(_config.output_shape())));
            }

            // perform the copy
            copy::options_block2block_t options{
                count, 0, 0,
                _config.sample_slice, _config.sample_transform
            };
            copy::copy(input_buffer, output_buffer, options);

            if (_log) { _log->trace("processed block {}", id); }
        }

        config_t _config;

        std::shared_ptr<spdlog::logger> _log;

        std::optional<util::completion_worker_pool_t<std::exception_ptr>> _pool;

        std::mutex _mutex;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto processor(std::shared_ptr<vortex::process::copy_processor_t<Args...>> a) {
        using adapter = adapter<block_t>;
        auto w = typename adapter::processor(a.get());

        w.device = []() -> std::optional<cuda::device_t> { return {}; };
        w.stream_factory = []() {
            return []() -> typename adapter::ascan_stream_t {
                return sync::lockable<cuda::cuda_host_tensor_t<typename block_t::process_element_t>>();
            };
        };

        w.input_shape = [a]() { return a->config().input_shape(); };
        w.output_shape = [a]() { return a->config().output_shape(); };

        w.channel = [a]() { return a->config().channel; };

        w.next_async = [a](block_t& block,
            const typename adapter::spectra_stream_t& input_stream_, typename adapter::ascan_stream_t& output_stream_,
            cuda::event_t* start, cuda::event_t* done, typename adapter::processor::callback_t&& callback) {
            std::visit([&](auto& input_stream, auto& output_stream) {
                try {
                    view_as_cpu([&](auto input_buffer, auto output_buffer) {
                        a->next_async(
                            block.id, input_buffer.range(block.length), output_buffer.range(block.length),
                            std::forward<typename adapter::processor::callback_t>(callback)
                        );
                    }, input_stream, output_stream);
                } catch (const unsupported_view&) {
                    callback(std::current_exception());
                }
            }, input_stream_, output_stream_);
        };

        return w;
    }
}

#endif
