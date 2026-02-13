/** \rst

    DAQmx-based analog and digital I/O

    The DAQmx component is built upon the DAQmx object-oriented
    driver.  Each instance of this component maps to a single
    DAQmx task, which means that multiple components will be
    required if both input and output are required or if multiple
    channel types are required.

    The configuration wraps the most commonly-used elements of the DAQx
    API in an object-oriented model.  The configuration struct contains all
    information needed to configure the task.

    The component exposes a simple API for initialization and handling
    of blocks.  All work is handled in a background thread.  Both
    synchronous and asynchronous (via callbacks) options are available.

 \endrst */

#pragma once

#include <optional>

#include <spdlog/spdlog.h>

#include <vortex/driver/daqmx.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/sync.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/variant.hpp>
#include <vortex/util/tuple.hpp>
#include <vortex/util/cast.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/stream.hpp>

namespace vortex::daqmx {

    namespace detail {

        struct base_t {
            size_t stream = 0;
            size_t channel = 0;

            bool operator==(const base_t& o) const {
                return stream == o.stream && channel == o.channel;
            }
        };

        struct digital_t : base_t {
            std::string line_name;
            static constexpr size_t max_bits = 32;

            bool operator==(const digital_t& o) const {
                return base_t::operator==(o) && line_name == o.line_name;
            }
        };

        struct analog_t : base_t {
            std::string port_name;

            double logical_units_per_physical_unit = 1;
            range_t<float64> limits = { {-10, 10} };

            bool operator==(const analog_t& o) const {
                return base_t::operator==(o) && port_name == o.port_name && logical_units_per_physical_unit == o.logical_units_per_physical_unit && limits == o.limits;
            }
        };

        struct analog_input_t : analog_t {};
        struct analog_output_t : analog_t {};

    }

    namespace channel {

        struct digital_output_t : detail::digital_t {
            void apply(daqmx::daqmx_t& daqmx, std::shared_ptr<spdlog::logger>& log) const {
                if (log) { log->debug("creating digital output {}", line_name); }
                daqmx.create_digital_output(line_name);
            }
        };
        struct digital_input_t : detail::digital_t {
            void apply(daqmx::daqmx_t& daqmx, std::shared_ptr<spdlog::logger>& log) const {
                if (log) { log->debug("creating digital input {}", line_name); }
                daqmx.create_digital_input(line_name);
            }
        };

        struct analog_voltage_output_t : detail::analog_output_t {
            void apply(daqmx::daqmx_t& daqmx, std::shared_ptr<spdlog::logger>& log) const {
                if (log) { log->debug("creating analog output {} with range [{} V, {} V] and scale {} lu/v", port_name, limits.min(), limits.max(), logical_units_per_physical_unit); }
                daqmx.create_analog_voltage_output(port_name, limits.min(), limits.max());
            }
        };
        struct analog_voltage_input_t : detail::analog_input_t {
            daqmx::terminal_t terminal = daqmx::terminal_t::referenced;

            void apply(daqmx::daqmx_t& daqmx, std::shared_ptr<spdlog::logger>& log) const {
                if (log) { log->debug("creating analog input {} with range [{} V, {} V] and scale {} lu/V", port_name, limits.min(), limits.max(), logical_units_per_physical_unit); }
                daqmx.create_analog_voltage_input(port_name, limits.min(), limits.max());
            }

            bool operator==(const analog_voltage_input_t& o) const {
                return detail::analog_input_t::operator==(o) && terminal == o.terminal;
            }
        };
    }
    // template<typename... Ts>
    // struct channel_t : std::variant<Ts...> {
    //     using base_t = std::variant<Ts...>;
    //     using base_t::base_t;

    //     auto apply(daqmx::daqmx_t& daqmx, std::shared_ptr<spdlog::logger>& log) const { DISPATCH_CONST(apply, daqmx, log); }

    //     bool operator==(const channel_t& o) const {
    //         return static_cast<const base_t&>(*this) == static_cast<const base_t&>(o);
    //     }
    // };
    using default_channel_t = std::variant<channel::analog_voltage_input_t, channel::analog_voltage_output_t, channel::digital_input_t, channel::digital_output_t>;

}

namespace vortex::io {

    template<typename channel_t_>
    struct daqmx_config_t {
        using channel_t = channel_t_;

        std::string name = "unknown";

        size_t& samples_per_second() { return _samples_per_second; }
        const size_t& samples_per_second() const { return _samples_per_second; }

        size_t& samples_per_block() { return _samples_per_block; }
        const size_t& samples_per_block() const { return _samples_per_block; }

        struct {
            std::string source = "pfi0";
            daqmx::edge_t edge = daqmx::edge_t::rising;
            size_t divisor = 1;
        } clock;

        size_t blocks_to_buffer = 1;

        seconds readwrite_timeout = seconds(1);

        std::vector<channel_t> channels;

        bool persistent_task = true;
        bool stop_on_error = true;

        void validate() {
            if (blocks_to_buffer < 1) {
                throw std::invalid_argument(fmt::format("must buffer at least 1 block: {}", blocks_to_buffer));
            }

            if (clock.divisor < 1) {
                throw std::invalid_argument(fmt::format("minimum clock divisor is 1: {}", clock.divisor));
            }
            if (samples_per_block() % clock.divisor != 0) {
                throw std::invalid_argument(fmt::format("samples per block ({}) must be evenly divisible by clock divisor ({}): {} != 0", samples_per_block(), clock.divisor, samples_per_block() % clock.divisor));
            }
            if (samples_per_second() % clock.divisor != 0) {
                throw std::invalid_argument(fmt::format("samples per second ({}) must be evenly divisible by clock divisor ({}): {} != 0", samples_per_second(), clock.divisor, samples_per_second() % clock.divisor));
            }

            // only support one NI-DAQmx channel per analog channel represented here
            for (auto& channel : channels) {
                std::visit(overloaded{
                    [&](const daqmx::detail::analog_t& o) {
                        if (_port_name_contains_multiple_channels(o.port_name)) {
                            throw std::invalid_argument(fmt::format("analog inputs/outputs must map to a single DAQmx channel: {}", o.port_name));
                        }
                    },
                    [](const daqmx::detail::digital_t&) {}
                }, channel);
            }
        }

    protected:

        bool _port_name_contains_multiple_channels(const std::string& s) {
            return s.find(':') != std::string::npos || s.find(',') != std::string::npos;
        }

        size_t _samples_per_block = 100;
        size_t _samples_per_second = 100'000;

    };

    template<typename config_t_>
    class daqmx_io_t {
    public:
        using config_t = config_t_;

        using analog_element_t = float64;
        using digital_element_t = uInt32;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    protected:

        using job_t = std::function<void()>;

    public:

        daqmx_io_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~daqmx_io_t() {
            stop();

            if (_pool) {
                _pool->wait_finish();
            }
        }

        const config_t& config() const {
            return _config;
        }

        void initialize(config_t config) {
            config.validate();
            std::swap(_config, config);

            if (_config.persistent_task) {
                _prepare_task();
            }

            // launch background worker
            _pool.emplace("NI DAQmx Worker", 1, [](size_t) { setup_realtime(); }, _log);
        }

    protected:

        void _prepare_task() {
            // create the DAQmx
            _daqmx = daqmx::daqmx_t(_config.name);

            // configure channels
            for (auto& channel : _config.channels) {
                CALL_CONST(channel, apply, _daqmx, _log);
            }

            // configure sample clock for continuous generation
            if (_log) { _log->debug("configuring sample clock from {} expecting {} samples/second with {}x upsampling and {} samples/block", _config.clock.source, _config.samples_per_second(), _config.clock.divisor, _config.samples_per_block()); }
            _daqmx.configure_sample_clock(_config.clock.source, daqmx::sample_mode_t::continuous, _config.samples_per_second() / _config.clock.divisor, _config.samples_per_block() / _config.clock.divisor, _config.clock.divisor, _config.clock.edge);

            // allocate channel staging buffers
            // TODO: there has got to be a more elegant way to do this by iterating over types
            size_t analog_output_width = 0, analog_input_width = 0, digital_output_width = 0, digital_input_width = 0;

            for (auto& channel : _config.channels) {
                std::visit(overloaded{
                    [&](const daqmx::detail::analog_output_t& o) { analog_output_width++; },
                    [&](const daqmx::detail::analog_input_t& o) { analog_input_width++; },
                    [&](const daqmx::channel::digital_output_t&) { digital_output_width++; },
                    [&](const daqmx::channel::digital_input_t&) { digital_input_width++; },
                }, channel);
            }

            auto daq_samples = _config.samples_per_block() / _config.clock.divisor;
            _analog_input_buffer.resize(std::vector<size_t>{ daq_samples, analog_input_width });
            _analog_output_buffer.resize(std::vector<size_t>{ daq_samples, analog_output_width });
            _digital_input_buffer.resize(std::vector<size_t>{ daq_samples, digital_input_width });
            _digital_output_buffer.resize(std::vector<size_t>{ daq_samples, digital_output_width });

            auto buffer_size = _config.blocks_to_buffer * _config.samples_per_block() / _config.clock.divisor;
            if (analog_output_width + digital_output_width > 0) {
                // disable signal regeneration for live streaming
                if (_log) { _log->debug("disabling regeneration"); };
                _daqmx.set_regeneration(false);

                // inform NI-DAQmx of output buffering requirements
                if (_log) { _log->debug("setting output buffer of {} samples/channel", buffer_size); };
                _daqmx.set_output_buffer_size(buffer_size);
            }
            if (analog_input_width + digital_input_width > 0) {
                // inform NI-DAQmx of input buffering requirements
                if (_log) { _log->debug("setting input buffer of {} samples/channel", buffer_size); };
                _daqmx.set_input_buffer_size(buffer_size);
            }
        }

    public:

        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        size_t next(size_t count, const std::tuple<Vs...>& streams) {
            return next(0, count, streams);
        }
        template<typename... Vs, typename = typename std::enable_if_t<(is_cpu_viewable<Vs> && ...)>>
        size_t next(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            std::unique_lock<std::mutex> lock(_mutex);

            // dispatch now
            auto [n, error] = _dispatch_block(id, count, streams);
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

            // dispatch in background
            _pool->post([this, id, count, streams, callback = std::forward<callback_t>(callback_)]() {
                auto result = _dispatch_block(id, count, streams);
                std::apply(callback, result);
            });
        }

        void prepare() {
            if (!_config.persistent_task) {
                _prepare_task();
            }
        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!running()) {
                if (_log) { _log->info("starting I/O"); }
                _daqmx.start_task();
            }
        }

        void stop() {
            std::unique_lock<std::mutex> lock(_mutex);

            _stop();
        }

    protected:

        void _stop() {
            if (running()) {
                if (_log) { _log->info("stopping I/O"); }
                try {
                    if (_config.persistent_task) {
                        _daqmx.stop_task();
                    }
                    else {
                        if (_log) { _log->debug("releasing task resources"); }
                        _daqmx.clear_task();
                    }
                }
                catch (const daqmx::exception& e) {
                    if (_log) { _log->warn("exception while stopping: {}", to_string(e)); }
                }
            }
        }

    public:

        bool running() const {
            return _daqmx.running();
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
                if (count % _config.clock.divisor != 0) {
                    raise(_log, "block size ({}) is not evenly divisible by clock divisor ({}): {} != 0", count, _config.clock.divisor, count % _config.clock.divisor);
                }

                // perform the operation
                n = _process_block(id, count, streams);
            } catch (const daqmx::exception&) {
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
        size_t _process_block(size_t id, size_t count, const std::tuple<Vs...>& streams) {
            if (_log) { _log->trace("processing block {}", id); }

            auto block_samples = std::min(count, _config.samples_per_block());
            auto daq_samples = block_samples / _config.clock.divisor;
            auto block_step = _config.clock.divisor;

            // populate the output buffers from the block
            size_t analog_output_index = 0, digital_output_index = 0;
            for (auto& channel : _config.channels) {
                std::visit(overloaded{
                    [&](const daqmx::detail::analog_output_t& o) {
                        // access the stream
                        vortex::template select<analog_element_t>(streams, o.stream, [&](const auto& stream_) {
                            auto& stream = stream_.derived_cast();
                            _check_buffer(stream, block_samples, o.channel);

                            // prepare the copy by downsampling as needed
                            auto src_lu = xt::view(stream.to_xt(), xt::range(0, block_samples, block_step), o.channel);
                            auto dst_pu = xt::view(_analog_output_buffer, xt::range(0, daq_samples), analog_output_index++);

                            // load the data with scaling
                            dst_pu = xt::clip(src_lu / o.logical_units_per_physical_unit, o.limits.min(), o.limits.max());
                        });
                    },
                    [&](const daqmx::channel::digital_output_t& o) {
                        // access the stream
                        vortex::template select<digital_element_t>(streams, o.stream, [&](const auto& stream_) {
                            auto& stream = stream_.derived_cast();
                            _check_buffer(stream, block_samples, o.channel);

                            // prepare the copy by downsampling as needed
                            auto src = xt::view(stream.derived_cast().to_xt(), xt::range(0, block_samples, block_step), o.channel);
                            auto dst = xt::view(_digital_output_buffer, xt::range(0, daq_samples), digital_output_index++);

                            // load the data directly
                            dst = src;
                        });
                    },
                    // NOTE: cannot use const auto& here because then the analog functions will not get called
                    [&](const daqmx::detail::analog_input_t&) {},   // ignore
                    [&](const daqmx::channel::digital_input_t&) {}  // ignore
                }, channel);
            }

            // send the outputs
            if (_analog_output_buffer.size() > 0) {
                if (_log) { _log->trace("writing {} samples of analog I/O", daq_samples); }
                _daqmx.write_analog(daq_samples, _analog_output_buffer, _config.readwrite_timeout);
            }
            if (_digital_output_buffer.size() > 0) {
                if (_log) { _log->trace("writing {} samples of digital I/O", daq_samples); }
                _daqmx.write_digital(daq_samples, _digital_output_buffer, _config.readwrite_timeout);
            }

            // read the inputs
            if (_analog_input_buffer.size() > 0) {
                if (_log) { _log->trace("reading {} samples of analog I/O", daq_samples); }
                _daqmx.read_analog(daq_samples, _analog_input_buffer, _config.readwrite_timeout);
            }
            if (_digital_input_buffer.size() > 0) {
                if (_log) { _log->trace("reading {} samples of digital I/O", daq_samples); }
                _daqmx.read_digital(daq_samples, _digital_input_buffer, _config.readwrite_timeout);
            }

            // populate the block from the input buffers
            size_t analog_input_index = 0, digital_input_index = 0;
            for (auto& channel : _config.channels) {
                std::visit(overloaded{
                    [&](const daqmx::detail::analog_input_t& o) {
                        // access the stream
                        vortex::template select<analog_element_t>(streams, o.stream, [&](auto& stream_) {
                            auto& stream = stream_.derived_cast();
                            _check_buffer(stream, block_samples, o.channel);

                            // prepare the copy by upsampling as needed
                            auto src_pu = xt::view(_analog_input_buffer, xt::range(0, daq_samples), analog_input_index++, xt::newaxis());
                            auto dst_lu = xt::reshape_view(xt::view(stream.derived_cast().to_xt(), xt::range(0, block_samples), o.channel), { block_samples / block_step, block_step });

                            // load the data with scaling
                            dst_lu = src_pu * o.logical_units_per_physical_unit;
                        });
                    },
                    [&](const daqmx::channel::digital_input_t& o) {
                        // access the stream
                        vortex::template select<digital_element_t>(streams, o.stream, [&](auto& stream_) {
                            auto& stream = stream_.derived_cast();
                            _check_buffer(stream, block_samples, o.channel);

                            // prepare the copy by upsampling as needed
                            auto src = xt::view(_digital_input_buffer, xt::range(0, daq_samples), digital_input_index++, xt::newaxis());
                            auto dst = xt::reshape_view(xt::view(stream.derived_cast().to_xt(), xt::range(0, block_samples), o.channel), { block_samples / block_step, block_step });

                            // load the data directly
                            dst = src;
                        });
                    },
                    // NOTE: cannot use const auto& here because then the analog functions will not get called
                    [&](const daqmx::detail::analog_output_t&) {},   // ignore
                    [&](const daqmx::channel::digital_output_t&) {}  // ignore
                }, channel);
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

        xt::xtensor<analog_element_t, 2> _analog_output_buffer, _analog_input_buffer;
        xt::xtensor<digital_element_t, 2> _digital_output_buffer, _digital_input_buffer;

        daqmx::daqmx_t _daqmx;

        std::optional<util::worker_pool_t> _pool;
        std::mutex _mutex;

        config_t _config;

    };

}
