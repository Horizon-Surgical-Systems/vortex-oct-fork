
#pragma once

#include <optional>

#include <spdlog/spdlog.h>

#include <vortex/driver/alazar/core.hpp>
#include <vortex/driver/alazar/board.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/sync.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/stream.hpp>

namespace vortex::alazar {

    struct analog_channel_t {
        size_t stream = 0;
        size_t channel = 0;

        double park = 0;

        double logical_units_per_physical_unit = 1;
        range_t<double> limits = { {-5, 5} };
    };

}

namespace vortex::io {

    struct alazar_config_t {

        struct device_t {
            U32 system_index = 1;
            U32 board_index = 1;
        };
        device_t device;

        static constexpr size_t analog_channel_count = 2;
        std::array<alazar::analog_channel_t, analog_channel_count> analog_output_channels;

        size_t& samples_per_block() { return _samples_per_block; }
        const size_t& samples_per_block() const { return _samples_per_block; }

        size_t blocks_to_buffer = 1;
        size_t divisor = 1;

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
    class alazar_io_t {
    public:
        using config_t = config_t_;

        using analog_element_t = double;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    protected:

        using sample_t = uint32_t;
        using half_sample_t = uint16_t;

        using job_t = std::function<void()>;

    public:

        alazar_io_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~alazar_io_t() {
            stop();

            if (_pool) {
                _pool->wait_finish();
            }
        }

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            if (_log) { _log->debug("initializing Alazar DAC module"); }

            // validate and accept the configuration
            config.validate();
            std::swap(_config, config);

            // load the board
            _board = alazar::board_t(config.device.system_index, config.device.board_index);
            if (_log) { _log->info("using {} ({}) PCIe x{} @ {} Gbps", _board.info().type.model, _board.info().serial_number, _board.info().pcie.width, _board.info().pcie.speed_gbps); }

            // configure custom pattern mode
            if (_log) { _log->debug("configuring custom pattern mode"); }
            _board.configure_dac_mode(0, false);

            // find DAC slot that is sufficiently large for complete buffer
            _total_buffer_samples = _config.blocks_to_buffer * _config.samples_per_block();
            bool slot_found = false;
            for (auto& [idx, size] : _board.info().dac.slot_sizes) {
                if (size >= _total_buffer_samples) {
                    _slot_idx = idx;
                    slot_found = true;
                    break;
                }
            }
            if (!slot_found) {
                raise(_log, "could not find DAC slot with capacity for {} samples", _total_buffer_samples);
            }
            if (_log) { _log->debug("using pattern slot {} for {} samples", _slot_idx, _total_buffer_samples); }

            // configure first sequences to repeat this slot indefinitely
            if (_log) { _log->debug("configuring sequence {} to repeat pattern slot {}",_sequence_idx, _slot_idx); }
            // NOTE: end index is the index of the last sample, not the index after the last sample
            _board.configure_dac_sequence(_sequence_idx, _slot_idx, 0, 0, _total_buffer_samples - 1);

            // allocate staging buffer
            _analog_output_buffer.resize(std::vector<size_t>{ _config.samples_per_block(), _config.analog_channel_count });

            // set park position
            std::vector<uint16_t> parks;
            for (auto& o : _config.analog_output_channels) {
                parks.push_back(std::min(std::max(o.park / o.logical_units_per_physical_unit, o.limits.min()), o.limits.max()) / _board.info().dac.volts_per_lsb + _board.info().dac.zero_point);
            }
            if (_log) { _log->debug("setting park position to ({}, {})", parks[0], parks[1]); }
            _board.set_dac_park_position(parks[0], parks[1]);

            // launch background worker
            _pool.emplace("Alazar DAC Worker", 1, [](size_t) { setup_realtime(); }, _log);
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
            _buffer_offset_samples = 0;
        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (_log) { _log->info("starting I/O"); }
            _board.start_dac();
        }

        void stop(bool underlying = false) {
            std::unique_lock<std::mutex> lock(_mutex);

            _stop(underlying);
        }

    protected:

        void _stop(bool underlying = false) {

            try {
                if (underlying && running()) {
                    if (_log) { _log->info("stopping I/O (underlying acquisition)"); }
                    _board.stop_capture();
                }

                if (_log) { _log->info("stopping I/O"); }
                // omit running check because DAC is only running when board is
                _board.stop_dac();
            } catch (const alazar::exception& e) {
                if (_log) { _log->warn("exception while stopping: {}", to_string(e)); }
            }
        }

    public:

        bool running() const {
            return _board.running();
        }

    protected:

        template<typename... Vs>
        auto _dispatch_block(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            if (_log) { _log->trace("dispatching block {}", id); }

            size_t n = 0;
            std::exception_ptr error;
            try {

                // determine if this block length is acceptable
                if (count > _config.samples_per_block()) {
                    raise(_log, "block is larger than maximum configured size: {} > {}", count, _config.samples_per_block());
                }
                if (count % _config.divisor != 0) {
                    raise(_log, "block size ({}) is not evenly divisible by divisor ({}): {} != 0", count, _config.divisor, count % _config.divisor);
                }

                // perform the operation
                n = _process_block(id, count, streams);

            } catch (const std::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error during I/O for block {}: {}", id, to_string(error)); }
            }

            // stop if necessary
            if (error && _config.stop_on_error) {
                // NOTE: do not reacquire lock
                _stop(true);
            }

            return std::make_tuple(n, error);
        }

        template<typename... Vs>
        auto _process_block(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            if (_log) { _log->trace("processing block {}", id); }

            auto block_samples = std::min(count, _config.samples_per_block());
            auto daq_samples = block_samples / _config.divisor;
            auto block_step = _config.divisor;

            // populate the output buffers from the block while handling wraparound
            size_t written_samples = 0;
            while (written_samples < daq_samples) {

                // determine number of samples left to write until end of buffer
                auto write_count = std::min(daq_samples - written_samples, _total_buffer_samples - _buffer_offset_samples);

                size_t analog_output_index = 0;
                for (auto& o : _config.analog_output_channels) {
                    // access the stream
                    vortex::template select<analog_element_t>(streams, o.stream, [&](const auto& stream_) {
                        auto& stream = stream_.derived_cast();
                        _check_buffer(stream, block_samples, o.channel);

                        // prepare the copy by downsampling as needed
                        auto src_lu = xt::view(stream.to_xt(), xt::range(0, block_samples, block_step), o.channel);
                        auto dst_pu = xt::view(_analog_output_buffer, xt::range(0, write_count), analog_output_index++);

                        // load the data with scaling
                        dst_pu = xt::clip(src_lu / o.logical_units_per_physical_unit, o.limits.min(), o.limits.max()) / _board.info().dac.volts_per_lsb + _board.info().dac.zero_point;
                    });
                }

                // send the outputs
                if (_analog_output_buffer.size() > 0) {
                    if (_log) { _log->trace("writing {} samples of analog I/O", write_count); }
                    _board.write_dac_slot(_slot_idx, reinterpret_cast<sample_t*>(_analog_output_buffer.data()), write_count, _buffer_offset_samples);
                }

                // update buffer write position
                _buffer_offset_samples = (_buffer_offset_samples + write_count) % _total_buffer_samples;
                written_samples += write_count;
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

        size_t _sequence_idx = 0, _slot_idx;
        size_t _total_buffer_samples, _buffer_offset_samples;

        xt::xtensor<half_sample_t, 2> _analog_output_buffer;

        alazar::board_t _board;

        std::optional<util::worker_pool_t> _pool;
        std::mutex _mutex;

        config_t _config;

    };

}
