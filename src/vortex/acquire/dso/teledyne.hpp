#pragma once

#include <array>
#include <variant>
#include <algorithm>
#include <optional>

#include <spdlog/spdlog.h>

#include <xtensor/core/xnoalias.hpp>

#include <vortex/acquire/dso.hpp>

#include <vortex/driver/teledyne.hpp>

#include <vortex/memory/teledyne.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/tuple.hpp>
#include <vortex/util/variant.hpp>

#include <vortex/core.hpp>

namespace vortex::teledyne {

    struct clock_t {
        size_t sampling_frequency = 2'500'000'000;
        size_t reference_frequency = 10'000'000;
        clock_generator_t clock_generator = clock_generator_t::internal_pll;
        clock_reference_source_t reference_source = clock_reference_source_t::internal;
        size_t delay_adjustment = 0;
        bool low_jitter_mode_enabled = true;
    };

    struct input_t {
        int channel = 0;
    };

}

namespace vortex::acquire {
    struct teledyne_config_t : dso_config_t {

        size_t device = 0;

        teledyne::clock_t clock;
        std::vector<teledyne::input_t> inputs;

        teledyne::trigger_source_t trigger_source = teledyne::trigger_source_t::port_trig;
        int32_t trigger_skip_factor = 1;
        int64_t trigger_offset_samples = 0;
        bool trigger_sync_passthrough = true;

        int64_t sample_skip_factor = 1;

        // FWOCT options
        bool enable_fwoct = false;
        double resampling_factor = 1;
        double clock_delay_samples = 0;
        teledyne::clock_edges_t clock_edges = teledyne::clock_edges_t::rising;
        xt::xtensor<int16_t, 1> background;
        xt::xtensor<std::complex<float>, 1> spectral_filter;
        teledyne::fft_mode_t fft_mode = teledyne::fft_mode_t::disabled;

        std::chrono::milliseconds acquire_timeout = std::chrono::seconds(1);

#if defined(VORTEX_PLATFORM_LINUX)
        bool enable_hugepages = false;
#else
        bool enable_hugepages = false;
#endif

        bool stop_on_error = true;

        double periodic_trigger_frequency = 10'000; // Hz
        bool test_pattern_signal = false;

        void validate() {

            // require at least one channel
            if (inputs.empty()) {
                throw std::runtime_error("no inputs are configured");
            }

            // ensure channels are unique
            for (size_t i = 0; i < inputs.size(); i++) {
                for (size_t j = 0; j < i; j++) {
                    if (inputs[i].channel == inputs[j].channel) {
                        throw std::runtime_error(fmt::format("cannot configure the same channel as two different inputs: inputs[{}] = Channel #{} and inputs[{}] = Channel #{}", i, inputs[i].channel, j, inputs[j].channel));
                    }
                }
            }

            if (sample_skip_factor < 1) {
                throw std::runtime_error("sample skip factor must be positive");
            }
            if (trigger_skip_factor < 1) {
                throw std::runtime_error("trigger skip factor must be positive");
            }

            auto board = teledyne::board_t(device);
            if (board.info().fwoct.detected) {

                // ensure record length meets alignment requirements
                if (samples_per_record() % board.info().fwoct.alignment_divisor != 0) {
                    throw std::invalid_argument(fmt::format("record length of {} is not a multiple of {} (FWOCT requirement)", samples_per_record(), board.info().fwoct.alignment_divisor));
                }

                if (enable_fwoct) {
                    // ensure that FFT size is supported
                    if (fft_mode != teledyne::fft_mode_t::disabled && samples_per_record() > board.info().fwoct.fft_capacity) {
                        throw std::runtime_error(fmt::format("required FFT size of {} exceeds capacity of {}", samples_per_record(), board.info().fwoct.fft_capacity));
                    }

                    // ensure spectral filter matches the record length
                    if (spectral_filter.size() > 0 && spectral_filter.size() != samples_per_record()) {
                        throw std::invalid_argument(fmt::format("spectral filter and record length mismatch: {} vs {}", spectral_filter.size(), samples_per_record()));
                    }

                    // ensure background record matches record length
                    if (background.size() > 0 && background.size() != samples_per_record()) {
                        throw std::invalid_argument(fmt::format("background and record length mismatch: {} vs {}", background.size(), samples_per_record()));
                    }
                }

            } else {

                // check that FWOCT is not requested
                if (enable_fwoct) {
                    throw std::invalid_argument("FWOCT is requested but not detected");
                }

            }
        }

        uint32_t channel_mask() const {
            uint32_t mask = 0;
            for (const auto& input : inputs) {
                mask |= (1 << input.channel);
            }
            return mask;
        }

        size_t channels_per_sample() const override {
            return inputs.size();
        }

        std::array<ptrdiff_t, 3> stride() const override {
            // stride for non-interleaved channels
            return { downcast<ptrdiff_t>(_samples_per_record), 1, downcast<ptrdiff_t>(_records_per_block * _samples_per_record) };
        }

        auto& samples_per_second() { return clock.sampling_frequency; }
        const auto& samples_per_second() const { return clock.sampling_frequency; }
    };

    class teledyne_acquisition_t {
    public:

        using output_element_t = uint16_t;
        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    public:

        teledyne_acquisition_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) {}

        virtual ~teledyne_acquisition_t() {
            stop();

            if (_pool) {
                _pool->wait_finish();
            }
        }

        const teledyne_config_t& config() const {
            return _config;
        }

        const std::optional<teledyne::board_t>& board() const {
            return _board;
        }

        virtual void initialize(teledyne_config_t config) {
            if (_log) { _log->debug("initializing Teledyne ADQ acquisition"); }

            // validate and accept the configuration
            config.validate();
            std::swap(_config, config);

            // create the digitizer board
            _board.emplace(_config.device);
            if(_log) { _log->info("using {}{} ({}) with firmware {}", _board->info().parameters.product_name, _board->info().parameters.product_options, _board->info().parameters.serial_number, _board->info().parameters.firmware.name); }

            // configure sampling clock
            if (_log) { _log->debug("configuring {} sampling clock from {} at {} samples/second with delay of {}", to_string(_config.clock.clock_generator), to_string(_config.clock.reference_source), _config.clock.sampling_frequency, _config.clock.delay_adjustment); }
            _board->configure_sampling_clock(
                _config.clock.sampling_frequency,
                _config.clock.reference_frequency,
                static_cast<ADQClockGenerator>(_config.clock.clock_generator),
                static_cast<ADQReferenceClockSource>(_config.clock.reference_source),
                _config.clock.delay_adjustment,
                _config.clock.low_jitter_mode_enabled
            );

            // configure triggering
            if (_log) { _log->debug("configuring trigger from {} with offset of {} samples and skip factor of {}", to_string(_config.trigger_source), _config.trigger_offset_samples, _config.trigger_skip_factor); }
            _board->configure_trigger_source(
                static_cast<ADQEventSource>(_config.trigger_source),
                _config.periodic_trigger_frequency,
                _config.trigger_offset_samples,
                _config.trigger_skip_factor
            );

            // prepare asynchronous acquisition
            if (_log) { _log->debug("configuring acquisition on channel(s) {} with {} samples/record, {} records/block, and sample skipping of {} (test pattern {})", teledyne::channel_mask_to_string(_config.channel_mask()), _config.samples_per_record(), _config.records_per_block(), _config.sample_skip_factor, _config.test_pattern_signal ? "on": "off"); }
            _board->configure_capture(
                _config.channel_mask(),
                _config.samples_per_record(),
                _config.records_per_block(),
                teledyne::infinite_acquisition,
                _config.test_pattern_signal,
                _config.sample_skip_factor
            );

            // NOTE: must configure sync passthrough after channels are configured since the trigger for the first channel is passed
            if(_config.trigger_sync_passthrough) {
                if (_log) { _log->debug("configuring trigger to sync passthrough"); }
                _board->configure_trigger_sync(true);
            }

            // NOTE: must configure hugepages after channels are configured since record buffer sizes are read from those settings
            if(_config.enable_hugepages) {
                auto bytes_per_buffer = _config.records_per_block() * _config.samples_per_record() * _config.channels_per_sample() * sizeof(output_element_t);
                auto buffer_count = _board->info().buffer_count;
                if (_log) { _log->debug("configuring {} hugepage buffers of size {} bytes each", buffer_count, bytes_per_buffer); }

                _board->configure_hugepages(true);
            }

            // commit configuration
            if (_log) { _log->debug("commiting digitizer board configuration"); }
            _board->commit_configuration();

            // configure FWOCT after committing configuration
            if (_board->info().fwoct.detected) {

                if (_log) {
                    _log->info("detected FWOCT version {} with {} clock channels, {} OCT channels, and FFT capacity of {}", _board->info().fwoct.version, _board->info().fwoct.clock_channel_count, _board->info().fwoct.oct_channel_count, _board->info().fwoct.fft_capacity);
                    _log->debug("configuring resampling by factor of {} using clock {} edges and delay of {} samples", _config.resampling_factor, to_string(_config.clock_edges), _config.clock_delay_samples);
                    if (_config.spectral_filter.size() > 0 || _config.background.size() > 0) {
                        if (_config.spectral_filter.size() > 0) {
                            _log->debug("configuring spectral filter of length {} samples", _config.spectral_filter.size());
                        } else {
                            _log->debug("configuring spectral filter as flat-top");
                        }
                        if (_config.background.size() > 0) {
                            _log->debug("configuring background subtraction of length {} samples", _config.background.size());
                        }
                    }
                    if (_config.fft_mode != teledyne::fft_mode_t::disabled) {
                        _log->debug("configuring FFT in mode {}", to_string(_config.fft_mode));
                    }
                    if (!_config.enable_fwoct) {
                        _log->debug("configuring FWOCT in bypass mode (raw clock on channel A and raw OCT signal on channel B)");
                    }
                }

                // must configure for FWOCT to produce output
                _board->configure_fwoct(
                    _config.samples_per_record(),
                    _config.clock_edges,
                    _config.clock_delay_samples,
                    1 / _config.resampling_factor,
                    _config.background.size() > 0 ? _config.background.data() : nullptr,
                    _config.spectral_filter.size() > 0 ? _config.spectral_filter.data() : nullptr,
                    _config.fft_mode,
                    _config.enable_fwoct ? 0b00 : 0b11 // if disabled, raw k-clock on channel A, raw OCT signal on channel B
                );

            }

            // launch worker pool
            _pool.emplace("Teledyne Worker", 1, [](size_t) { setup_realtime(); }, _log);
        }

    public:

        virtual void prepare() {

        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!running()) {
                if (_log) { _log->info("starting acquisition"); }
                _buffer_index = 0;
                _board->start_capture();
            }
        }

        void stop() {
            _stop(false);
        }

        bool running() const {
            return _board && _board->running();
        }

        template<typename V>
        size_t next(const cpu_viewable<V>& buffer) {
            return next(0, buffer);
        }
        template<typename V>
        size_t next(size_t id, const cpu_viewable<V>& buffer_) {
            const auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            // wait for buffer
            auto [n, error] = _wait_block(id, buffer, true);
            if (error) {
                std::rethrow_exception(error);
            }

            return n;
        }

        template<typename V>
        void next_async(const cpu_viewable<V>& buffer, callback_t&& callback) {
            next_async(0, buffer, std::forward<callback_t>(callback));
        }
        template<typename V>
        void next_async(size_t id, const cpu_viewable<V>& buffer_, callback_t&& callback) {
            const auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            _pool->post([this, id, buffer, callback = std::forward<callback_t>(callback)]() {
                // wait for buffer
                auto [n, error] = _wait_block(id, buffer, false);
                std::invoke(callback, n, error);
            });
        }

        template<typename V>
        void next_async(size_t id, teledyne::teledyne_cpu_viewable<V>& buffer_, callback_t&& callback) {
            auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            _pool->post([this, id, &buffer, callback = std::forward<callback_t>(callback)]() {
                // wait for buffer
                auto [n, error] = _wait_block(id, buffer, false);
                std::invoke(callback, n, error);
            });
        }
        template<typename V>
        void unlock(teledyne::teledyne_cpu_viewable<V>& buffer_) {
            auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            // save values for later
            auto ptr = buffer.data();
            auto index = buffer.buffer_index();

            _board->unlock_buffer(buffer.buffer_index());
            buffer.unbind();

            if (_log) { _log->trace("released zero-copy buffer {} / {}", index, static_cast<void*>(ptr)); }
        }

    protected:

        template<typename V>
        auto _wait_block(size_t id, V& output_buffer, bool lock_is_held) {
            // default to no records acquired
            size_t n = 0;
            std::exception_ptr error;

            // handle early abort
            if (_abort) {
                if (_log) { _log->trace("aborted block {}", id); }
                return std::make_tuple(n, error);
            }

            try {
                // check that buffers are appropriate shape
                if (!shape_is_compatible(output_buffer.shape(), _config.shape())) {
                    throw std::runtime_error(fmt::format("stream shape is not compatible with configured shape: {} !~= {}", shape_to_string(output_buffer.shape()), shape_to_string(_config.shape())));
                }

                // wait until next job is done
                if (_log) { _log->trace("waiting for block {}", id); }

                // zero copy
                // NOTE: the driver fills the buffers in sequence
                void* ptr;
                auto idx = _buffer_index % _board->info().buffer_count;
                n = _board->wait_and_lock_buffer(&ptr, idx, _config.acquire_timeout);
                output_buffer.bind(static_cast<output_element_t*>(ptr), idx);

                // advance buffers
                _buffer_index++;
            } catch (const teledyne::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error while waiting for block {}: {}", id, to_string(error)); }
            }
            if (_log) { _log->trace("acquired block {} with {} records (zero-copy buffer {} / {})", id, n, output_buffer.buffer_index(), static_cast<void*>(output_buffer.data())); }

            // stop if necessary
            if (error && _config.stop_on_error) {
                // NOTE: call the internal _stop() because the caller may have already locked the mutex
                _stop(lock_is_held);
            };

            return std::make_tuple(n, error);
        }

        template<typename V>
        auto _wait_block(size_t id, const V& output_buffer, bool lock_is_held) {
            // default to no records acquired
            size_t n = 0;
            std::exception_ptr error;

            // handle early abort
            if (_abort) {
                if (_log) { _log->trace("aborted block {}", id); }
                return std::make_tuple(n, error);
            }

            try {
                // check that buffers are appropriate shape
                if (!shape_is_compatible(output_buffer.shape(), _config.shape())) {
                    throw std::runtime_error(fmt::format("stream shape is not compatible with configured shape: {} !~= {}", shape_to_string(output_buffer.shape()), shape_to_string(_config.shape())));
                }

                // wait until next job is done
                if (_log) { _log->trace("waiting for block {}", id); }

                // normal copy
                auto idx = _buffer_index % _board->info().buffer_count;
                n = _board->wait_and_copy_buffer(output_buffer.data(), idx, _config.acquire_timeout);

                // advance buffers
                _buffer_index++;
            } catch (const teledyne::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error while waiting for block {}: {}", id, to_string(error)); }
            }
            if (_log) { _log->trace("acquired block {} with {} records", id, n); }

            // stop if necessary
            if (error && _config.stop_on_error) {
                // NOTE: call the internal _stop() because the caller may have already locked the mutex
                _stop(lock_is_held);
            };

            return std::make_tuple(n, error);
        }

        void _stop(bool lock_is_held) {
            std::unique_lock<std::mutex> lock(_mutex, std::defer_lock);
            if (!lock_is_held) {
                lock.lock();
            }

            if (running()) {
                if (_log) { _log->info("stopping acquisition"); }
                try {
                    _board->stop_capture();
                } catch (const teledyne::exception& e) {
                    if (_log) { _log->warn("exception while stopping acquisition: {}", to_string(e)); }
                }
            }

            if (_pool) {
                // abort all pending jobs
                _abort = true;
                // clear abort flag once job queue is flushed
                _pool->post([this]() { _abort = false; });
            }
        }

        std::shared_ptr<spdlog::logger> _log;

        size_t _buffer_index = 0;
        std::optional<teledyne::board_t> _board;

        std::atomic_bool _abort = false;
        std::optional<util::worker_pool_t> _pool;
        std::mutex _mutex;

        teledyne_config_t _config;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t>
    auto acquisition(std::shared_ptr<vortex::acquire::teledyne_acquisition_t> a) {
        using adapter = adapter<block_t>;
        auto w = acquisition<block_t>(a, base_t());

        w.stream_factory = [a]() {
            return [a]() -> typename adapter::spectra_stream_t {
                auto& cfg = a->config();

                // return a pre-sized but unbound buffer
                return sync::lockable<teledyne::teledyne_cpu_view_t<typename block_t::acquire_element_t>>(nullptr, { nullptr, nullptr }, cfg.shape(), cfg.stride(), -1);
            };
        };

        w.next_async = [a](block_t& block, typename adapter::spectra_stream_t& stream_, typename adapter::acquisition::callback_t&& callback) {
            std::visit([&](auto& stream) {
                try {
                    if constexpr (teledyne::is_teledyne_cpu_viewable<decltype(view(stream))>) {
                        a->next_async(block.id, stream, std::forward<typename adapter::acquisition::callback_t>(callback));
                    } else {
                        throw unsupported_view("only Teledyne CPU views are supported");
                    }
                } catch (const unsupported_view&) {
                    callback(0, std::current_exception());
                }
            }, stream_);
        };

        w.recycle = [a](block_t& block, typename adapter::spectra_stream_t& stream_) {
            std::visit([&](auto& stream) {
                if constexpr (teledyne::is_teledyne_cpu_viewable<decltype(view(stream))>) {
                    a->unlock(stream);
                } else {
                    throw unsupported_view("only Teledyne CPU views are supported");
                }
            }, stream_);
        };

        return w;
    }
}

#endif
