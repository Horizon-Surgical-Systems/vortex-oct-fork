
#pragma once

#include <optional>

#include <spdlog/spdlog.h>

#include <vortex/driver/machdsp.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/sync.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/stream.hpp>

namespace vortex::machdsp {

    struct channel_t {
        size_t stream = 0;
        size_t channel = 0;

        double logical_units_per_physical_unit = 1;
        double degree_scale = 15;
    };

}

namespace vortex::io {

    struct machdsp_config_t {

        std::string port = "COM1";
        size_t baud_rate = 12'000'000;

        static constexpr size_t channel_count = 2;
        std::array<machdsp::channel_t, channel_count> output_channels, input_channels;

        size_t& samples_per_block() { return _samples_per_block; }
        const size_t& samples_per_block() const { return _samples_per_block; }

        size_t blocks_to_buffer = 1;

        seconds readwrite_timeout = seconds(1);

        size_t sample_divisor = 1;
        size_t aux_divisor = 1;

        bool trigger_rising_edge = true;

        bool stop_on_error = true;

        void validate() {
            if (blocks_to_buffer < 1) {
                throw std::invalid_argument(fmt::format("must buffer at least 1 block: {}", blocks_to_buffer));
            }
        }

    protected:

        size_t _samples_per_block = 100;

    };

    template<typename config_t_>
    class machdsp_io_t {
    public:
        using config_t = config_t_;

        using element_t = double;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    protected:

        using sample_t = uint32_t;
        using half_sample_t = uint16_t;

        using job_t = std::function<void()>;

    public:

        machdsp_io_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~machdsp_io_t() {
            stop();

            if (_pool) {
                _pool->wait_finish();
            }
        }

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            if (_log) { _log->debug("initializing MachDSP module"); }

            // validate and accept the configuration
            config.validate();
            std::swap(_config, config);

            // load the board
            _board =  machdsp::machdsp_t(_config.port, _config.baud_rate);
            if (_log) { _log->info("using version {}.{} @ {} baud with features {}", static_cast<size_t>(_board.info().version.major), static_cast<size_t>(_board.info().version.minor), _config.baud_rate, static_cast<size_t>(_board.info().version.features)); }

            // initialize the board
            _board.clear_error();
            _board.stream_stop();
            _board.stream_reset();
            _board.reset_settings();

            auto daq_samples = _config.samples_per_block() / _config.sample_divisor;

            // configure board
            _board.set_internal_trigger(0);
            if (_log) { _log->info("configuring external trigger divisor of {} on {} edge", _config.sample_divisor, _config.trigger_rising_edge ? "rising" : "falling"); }
            _board.set_sample_divisor(_config.sample_divisor);
            _board.set_trigger_edge_rising(_config.trigger_rising_edge);
            if (_log) { _log->info("configuring auxiliary I/O divisor of {}", _config.aux_divisor); }
            _board.set_aux_divisor(_config.aux_divisor);
            if (_log) { _log->info("configuring {} blocks with {} samples/block", _config.blocks_to_buffer, daq_samples); }
            _board.set_blocks(_config.blocks_to_buffer, daq_samples);
            _board.set_enable_receive(false);
            _board.set_overflow_behavior(machdsp::overflow_t::OVERFLOW_ERROR);
            _board.set_underflow_behavior(machdsp::underflow_t::UNDERFLOW_ERROR);

            // allocate staging buffer
            _output_buffer.resize(std::vector<size_t>{ daq_samples, _config.channel_count });

            // launch background worker
            _pool.emplace("MachDSP Worker", 1, [](size_t) { setup_realtime(); }, _log);
        }

    public:

        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        size_t next(size_t count, const std::tuple<Vs...>& streams) {
            return next(0, count, streams);
        }
        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        size_t next(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            std::unique_lock<std::mutex> lock(_mutex);

            auto [n, error] = _dispatch_block(id, count, streams);

            // report result now
            if (error) {
                std::rethrow_exception(error);
            }
            return n;
        }

        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        void next_async(size_t count, const std::tuple<Vs...>& streams, callback_t&& callback) {
            next_async(0, count, streams, std::forward<callback_t>(callback));
        }
        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        void next_async(size_t id, size_t count, const std::tuple<Vs...>& streams, callback_t&& callback_) {
            std::unique_lock<std::mutex> lock(_mutex);

            auto result = _dispatch_block(id, count, streams);

            // report result via callback
            _pool->post([this, result, callback = std::forward<callback_t>(callback_)]() {
                std::apply(callback, result);
            });
        }

        void prepare() {
            std::unique_lock<std::mutex> lock(_mutex);

            _board.stream_reset();
        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (_log) { _log->info("starting I/O"); }
            _board.stream_start();
        }

        void stop() {
            std::unique_lock<std::mutex> lock(_mutex);

            _stop();
        }

    protected:

        void _stop() {

            try {
                if (_log) { _log->info("stopping I/O"); }
                _board.stream_stop();
            } catch (const machdsp::exception& e) {
                if (_log) { _log->warn("exception while stopping: {}", to_string(e)); }
            }
        }

    public:

        bool running() const {
            return _board.stream_status().state == machdsp::stream_state_t::STREAM_CONTINUE;
        }

    protected:

        template<typename... Vs>
        auto _dispatch_block(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            if (_log) { _log->trace("dispatching block {}", id); }

            size_t n = 0;
            std::exception_ptr error;
            try {

                // determine if this block length is acceptable
                if (count != _config.samples_per_block()) {
                    raise(_log, "block is not exactly equal to the configured size: {} != {}", count, _config.samples_per_block());
                }
                if (count % _config.sample_divisor != 0) {
                    raise(_log, "block size ({}) is not evenly divisible by divisor ({}): {} != 0", count, _config.sample_divisor, count % _config.sample_divisor);
                }

                // perform the operation
                n = _process_block(id, count, streams);

            } catch (const machdsp::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error during I/O for block {}: {}", id, to_string(error)); }
            }

            // stop if necessary
            if (error && _config.stop_on_error) {
                // NOTE: do not reacquire lock
                _stop();
            }

            return std::make_tuple(n, error);
        }

        template<typename... Vs>
        auto _process_block(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            if (_log) { _log->trace("processing block {}", id); }

            auto block_samples = std::min(count, _config.samples_per_block());
            auto daq_samples = block_samples / _config.sample_divisor;
            auto block_step = _config.sample_divisor;

            size_t output_index = 0;
            for (auto& o : _config.output_channels) {
                // access the stream
                vortex::template select<element_t>(streams, o.stream, [&](const auto& stream_) {
                    auto& stream = stream_.derived_cast();
                    _check_buffer(stream, block_samples, o.channel);

                    // prepare the copy by downsampling as needed
                    auto src_lu = xt::view(stream.to_xt(), xt::range(0, block_samples, block_step), o.channel);
                    auto dst_pu = xt::view(_output_buffer, xt::range(0, daq_samples), output_index++);

                    // load the data with scaling
                    dst_pu = xt::clip(src_lu / o.logical_units_per_physical_unit, -o.degree_scale, o.degree_scale) / (o.degree_scale * _board.info().per_lsb) + _board.info().zero_point;
                });
            }

            // send the outputs
            if (_output_buffer.size() > 0) {
                if (_log) { _log->trace("writing {} samples of I/O", daq_samples); }
                _board.block_write(_output_buffer.data(), daq_samples * sizeof(sample_t));
            }

            if (_log) { _log->trace("finished block {} with {} samples", id, block_samples); }
            return block_samples;
        }

        template<typename B>
        void _check_buffer(const B& buffer, size_t samples, size_t channel) {
            // check the stream dimension
            if (buffer.dimension() != 2) {
                raise(_log, "stream of dimension 2 is required: {}", buffer.dimension());
            }

            // check the stream size
            if (samples > buffer.shape(0) || channel >= buffer.shape(1)) {
                raise(_log, "stream is incorrectly sized: [{}] vs [:{},{}] ", shape_to_string(buffer.shape()), samples, channel);
            }
        }

        std::shared_ptr<spdlog::logger> _log;

        xt::xtensor<half_sample_t, 2> _output_buffer, _input_buffer;

        mutable machdsp::machdsp_t _board;

        std::optional<util::worker_pool_t> _pool;
        std::mutex _mutex;

        config_t _config;

    };

}
