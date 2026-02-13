/** \rst

    Alazar card acquisition component

    This file provides the configuration and component for an
    Alazar-based acquisition.

    The configuration wraps the most commonly-used elements of the Alazar
    API in an object-oriented model.  The configuration struct contains all
    information needed to configure the card and acquisition.

    The component exposes a simple API for initialization and acquisition
    of blocks.  All work is handled in a background thread.  Both
    synchronous and asynchronous (via callbacks) options are available.

 \endrst */

#pragma once

#include <type_traits>
#include <optional>
#include <thread>

#include <spdlog/spdlog.h>

#include <vortex/acquire/dso.hpp>
#include <vortex/acquire/dso/alazar/config.hpp>

#include <vortex/driver/alazar/core.hpp>
#include <vortex/driver/alazar/board.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/variant.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/exception.hpp>

#include <vortex/core.hpp>

namespace vortex::acquire::detail {

    template<typename clock_t_, typename trigger_t_, typename option_t_>
    struct alazar_config_t : dso_config_t {
        using clock_t = clock_t_;
        using trigger_t = trigger_t_;
        using option_t = option_t_;

        struct device_t {
            U32 system_index = 1;
            U32 board_index = 1;
        };
        device_t device;

        clock_t clock = { alazar::clock::internal_t{} };
        trigger_t trigger = { alazar::trigger::single_external_t{} };

        std::vector<alazar::input_t> inputs;

        std::vector<option_t> options;

        std::vector<size_t> resampling;

        std::chrono::milliseconds acquire_timeout = std::chrono::seconds(1);

        bool stop_on_error = true;

        template<typename board_t>
        void validate(const board_t& board) {
            // require at least one channel
            if (inputs.empty()) {
                throw std::runtime_error("no inputs are configured");
            }

            // ensure channels are unique
            std::vector<alazar::channel_t> channels;
            for (auto& input : inputs) {
                channels.push_back(input.channel);
            }
            for (size_t i = 0; i < channels.size(); i++) {
                for (size_t j = 0; j < i; j++) {
                    if (channels[i] == channels[j]) {
                        throw std::runtime_error(fmt::format("cannot configure the same channel as two different inputs: input {} = {} and input {} = {}", i, alazar::to_string(channels[i]), j, alazar::to_string(channels[j])));
                    }
                }
            }

            // ensure samples per record meets alignment requirements
            if (samples_per_record() % board.info().sample_alignment_divisor != 0) {
                throw std::runtime_error(fmt::format("samples per record must be evenly divisible by {}: {}", board.info().sample_alignment_divisor, samples_per_record()));
            }

            // ensure samples per record meets minimum requirements
            if (samples_per_record() < board.info().min_samples_per_record) {
                throw std::runtime_error(fmt::format("samples per record must equal or exceed {}: {}", board.info().min_samples_per_record, samples_per_record()));
            }

            // ensure samples per record matches the number of kept samples during resampling
            if (resampling.size() > 0 && resampling.size() != samples_per_record()) {
                throw std::runtime_error(fmt::format("samples per record does not match number of resampled samples: {} vs {}", samples_per_record(), resampling.size()));
            }
        }

        template<typename board_t>
        void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) {
            CALL_CONST(clock, apply, board, channel_mask(), log);
            CALL_CONST(trigger, apply, board, log);

            for (auto& input : inputs) {
                input.apply(board, log);
            }

            for (auto& option : options) {
                CALL_CONST(option, apply, board, log);
            }

            // configure resampling
            if (resampling.size() > 0) {
                // copy and sort indicies in order for bitmasking
                auto idxs = resampling;
                std::sort(idxs.begin(), idxs.end());

                // samples are selected in a bitmask of 16-bit types
                using bitmask_t = U16;
                constexpr size_t bits = 8 * sizeof(bitmask_t);

                // need as many clocks as the last sample index rounded up
                auto clocks_per_record = bits * std::ceil(idxs.back() / bits);

                // build bitmask
                std::vector<bitmask_t> bitmask(clocks_per_record / bits);
                for (auto idx : idxs) {
                    // TODO: determine bit ordering (LSB is sample 0 or MSB is sample 0)
                    auto chunk = idx / bits;
                    auto bit = idx % bits;
                    bitmask[chunk] |= (1 << bit);
                }

                log->debug("configuring sample skipping {} clocks -> {} samples", clocks_per_record, idxs.back());
                board.configure_sample_skipping(clocks_per_record, bitmask.data());
            } else if (board.info().features.sample_skipping) {
                // disable
                board.configure_sample_skipping();
            }
        }

        size_t recommended_minimum_records_per_block() const {
            // per Alazar documentation, have at least 1 MB per acquired buffer to maximize DMA throughput
            auto constexpr minimum_bytes_per_block = 1uLL << 20;
            auto bytes_per_record = bytes_per_multisample() * samples_per_record();
            return std::max<size_t>(1, std::ceil(minimum_bytes_per_block / double(bytes_per_record)));
        }

        virtual size_t buffer_bytes_per_record() const {
            return bytes_per_multisample() * samples_per_record();
        }
        auto bytes_per_multisample() const {
            size_t n = 0;
            for (auto& input : inputs) {
                n += input.bytes_per_sample();
            }
            return n;
        }

        auto channel_mask() const {
            auto mask = alazar::channel_t(0);
            for (const auto& input : inputs) {
                mask |= input.channel;
            }
            return mask;
        }

        size_t channels_per_sample() const override {
            return inputs.size();
        }

        // TODO: use std::optional<size_t> instead of requiring a separate check
        bool samples_per_second_is_known() const { return std::holds_alternative<alazar::clock::internal_t>(clock); }
        auto& samples_per_second() { 
            if(samples_per_second_is_known()) {
                return std::get<alazar::clock::internal_t>(clock).samples_per_second;
            } else {
                throw std::runtime_error("samples per second is not defined with current clock setting");
            }
        }
        const auto& samples_per_second() const {
            if (samples_per_second_is_known()) {
                return std::get<alazar::clock::internal_t>(clock).samples_per_second;
            } else {
                throw std::runtime_error("samples per second is not defined with current clock setting");
            }
        }
    };

    template<typename config_t_, typename board_t_>
    class alazar_acquisition_t {
    public:

        using config_t = config_t_;
        using board_t = board_t_;
        using output_element_t = uint16_t;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    protected:

        using job_t = std::function<void()>;

    public:

        alazar_acquisition_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)), _worker(&alazar_acquisition_t::_worker_loop, this) { }

        virtual ~alazar_acquisition_t() {
            stop();

            _jobs.finish();
            _worker.join();
        }

        const config_t& config() const {
            return _config;
        }

        const board_t& board() const {
            return _board;
        }

        virtual void initialize(config_t config) {
            if (_log) { _log->debug("initializing Alazar board"); }

            // validate and accept the configuration
            config.validate();
            std::swap(_config, config);

            // load the board
            _board = _config.create_board();
            if (_log) { _log->info("using {} ({}) PCIe x{} @ {} Gbps", _board.info().type.model, _board.info().serial_number, _board.info().pcie.width, _board.info().pcie.speed_gbps); }

            // apply the configuration
            _config.apply(_board, _log);
        }

        virtual void prepare() {
            // prepare asynchronous acquisition
            if (_log) { _log->debug("configuring acquisition on channels {} with {} samples/record and {} records/block", alazar::to_string(_config.channel_mask()), _config.samples_per_record(), _config.records_per_block()); }
            _board.configure_capture(
                _config.channel_mask(),
                downcast<U32>(_config.samples_per_record()),
                downcast<U32>(_config.records_per_block()),
                alazar::infinite_acquisition
            );
        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!running()) {
                if (_log) { _log->info("starting acquisition"); }
                _board.start_capture();
            }
        }

        void stop() {
            _stop(false);
        }

        bool running() const {
            return _board.running();
        }

    protected:

        template<typename V>
        size_t _next(size_t id, const viewable<V>& buffer_) {
            const auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            {
                // post buffer
                auto error = _post_block(id, buffer, true);
                if (error) {
                    std::rethrow_exception(error);
                }
            }

            {
                // wait for buffer
                auto [n, error] = _wait_block(id, buffer, true);
                if (error) {
                    std::rethrow_exception(error);
                }

                return n;
            }
        }

        template<typename V>
        void _next_async(size_t id, const viewable<V>& buffer_, callback_t&& callback) {
            const auto& buffer = buffer_.derived_cast();
            std::unique_lock<std::mutex> lock(_mutex);

            // post buffer
            auto error = _post_block(id, buffer, true);
            if (error) {
                // report error via callback
                _jobs.push([this, error, callback = std::forward<callback_t>(callback)]() {

#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        (void)this; // to satisfy clang
                        // report no samples read and the error
                        std::invoke(callback, 0, error);
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (_log) { _log->critical("unhandled callback exception: {}\n{}", to_string(e), check_trace(e)); }
                    }
#endif

                });
            } else {
                // wait for result and report via callback
                _jobs.push([this, id, buffer, callback = std::forward<callback_t>(callback)]() {

                    // wait for buffer
                    auto [n, error] = _wait_block(id, buffer, false);

#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        std::invoke(callback, n, error);
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (_log) { _log->critical("unhandled callback exception: {}\n{}", to_string(e), check_trace(e)); }
                    }
#endif

                });
            }
        }

        template<typename V>
        auto _post_block(size_t id, const V& buffer, bool lock_is_held) {
            // NOTE: this check is required for type safety because the input argument to post_buffer(...) is void*
            static_assert(std::is_same_v<output_element_t, typename V::element_t>, "mismatch between output element and stream element");

            std::exception_ptr error;

            try {
                // determine if this block length is acceptable
                if (buffer.shape(0) != _config.records_per_block()) {
                    throw std::invalid_argument(fmt::format("block is not exactly equal to the configured size: {} != {}", buffer.shape(0), _config.records_per_block()));
                }

                // check that stream is appropriate shape
                if (!shape_is_compatible(buffer.shape(), _config.shape())) {
                    throw std::runtime_error(fmt::format("buffer shape is not compatible with configured shape: {} !~= {}", shape_to_string(buffer.shape()), shape_to_string(_config.shape())));
                }

                // ensure the buffer is appropriately sized
                auto required_size = size_t(_config.buffer_bytes_per_record() * _config.records_per_block());
                if (buffer.size_in_bytes() < required_size) {
                    throw std::runtime_error(fmt::format("output buffer is not appropriately sized: {} < {}", buffer.size_in_bytes(), required_size));
                }

                // post buffer
                if (_log) { _log->trace("posting block {}", id); }
                _board.post_buffer(buffer.data(), buffer.size_in_bytes());

            } catch (const alazar::exception& e) {
                error = std::current_exception();
                if (_log) { _log->error("error during posting for block {}: {}", id, to_string(e)); }
            }

            // stop if necessary
            if (error && _config.stop_on_error) {
                // NOTE: call the internal _stop() because the caller may have already locked the mutex
                _stop(lock_is_held);
            }

            return error;
        }

        template<typename V>
        auto _wait_block(size_t id, const V& buffer, bool lock_is_held) {
            // default to no records acquired
            size_t n = 0;
            std::exception_ptr error;

            // handle early abort
            if (_abort) {
                if (_log) { _log->trace("aborted block {}", id); }
                return std::make_tuple(n, error);
            }

            try {
                // wait until next job is done
                if (_log) { _log->trace("waiting for block {}", id); }
                _board.wait_buffer(buffer.data(), _config.acquire_timeout);

                // report full block acquisition
                n = _config.records_per_block();
            } catch (const alazar::exception& e) {
                error = std::current_exception();
                if (_log) { _log->error("error while waiting for block {}: {}", id, to_string(e)); }
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

            if (_board.running()) {
                if (_log) { _log->info("stopping acquisition"); }
                try {
                    _board.stop_capture();
                } catch (const alazar::exception& e) {
                    if (_log) { _log->warn("exception while stopping acquisition: {}", to_string(e)); }
                }
            }

            // abort all pending jobs
            _abort = true;
            // clear abort flag once job queue is flushed
            _jobs.push([this]() { _abort = false; });
        }

        void _worker_loop() {
            set_thread_name("Alazar Worker");
            setup_realtime();

            if (_log) { _log->debug("worker thread entered"); }

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif
                job_t job;
                while (_jobs.pop(job)) {
                    std::invoke(job);
                }
#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in Alazar acquisition worker thread: {}\n{}", to_string(e), check_trace(e)); }
            }
#endif

            if (_log) { _log->debug("worker thread exited"); }
        }

        std::shared_ptr<spdlog::logger> _log;

        std::atomic_bool _abort = false;
        sync::queue_t<job_t> _jobs;
        std::thread _worker;

        board_t _board;

        std::mutex _mutex;

        config_t _config;

    };

}
