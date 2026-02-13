/** \rst

    utility classes for acquiring from files

    These classes replicate the interface of an acquisition component
    (e.g., Alazar card) but read data from disk.  They are primarily intended
    for use in testing, although the can be used for offline processing
    applications too.

 \endrst */

#pragma once

#include <fstream>
#include <tuple>

#include <spdlog/logger.h>

#include <vortex/memory/cpu.hpp>

#include <vortex/storage/detail/raw.hpp>

namespace vortex::acquire {

    struct file_config_t : null_config_t {

        std::string path;

        bool loop = true;

        virtual void validate() { }

    };

    template<typename output_element_t_, typename config_t_>
    class file_acquisition_t {
    public:

        using config_t = config_t_;
        using output_element_t = output_element_t_;
        using callback_t = std::function<void(size_t, std::exception_ptr)>;

        file_acquisition_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~file_acquisition_t() {
            stop();
        }

        void initialize(config_t config) {
            // accept configuration
            std::swap(_config, config);
        }

        const config_t& config() const {
            return _config;
        }

        void prepare() {
            // close any previously open file so acquisition restarts
            _close();

            // open file here to permit preloading by default
            _open();
        }
        void start() {
            // open file here for ease of use
            _open();
        }
        void stop() {
            // close file
            _close();
        }

        template<typename V>
        size_t next(const cpu_viewable<V>& buffer) {
            return next(0, buffer);
        }
        template<typename V>
        size_t next(size_t id, const cpu_viewable<V>& buffer) {
            auto [n, error] = _load(id, buffer.derived_cast().morph_right(3));
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
        void next_async(size_t id, const cpu_viewable<V>& buffer, callback_t&& callback) {
            std::apply(callback, _load(id, buffer.derived_cast().morph_right(3)));
        }

    protected:

        auto _open() {
            if (!_in.is_open()) {
                if (_log) { _log->debug("opening file \"{}\"", _config.path); }
                _in.exceptions(~std::ios::goodbit);
                _in.open(_config.path, std::ios::binary);
            }
        }
        auto _close() {
            if (_in.is_open()) {
                if (_log) { _log->debug("closing file \"{}\"", _config.path); }
                _in.close();
            }
        }

        auto _load(size_t id, const fixed_cpu_view_t<output_element_t, 3>& buffer) {
            if (!_in.is_open()) {
                throw std::runtime_error("file is not open");
            }

            size_t loaded = 0;
            std::exception_ptr error;

            auto record_size = _config.samples_per_record() * _config.channels_per_sample();

            try {
                // require appropriate record and sample sizes
                if (buffer.shape(1) != _config.samples_per_record()) {
                    throw std::invalid_argument(fmt::format("block samples is not exactly equal to the configured size: {} != {}", buffer.shape(1), _config.samples_per_record()));
                }
                if (buffer.shape(2) != _config.channels_per_sample()) {
                    throw std::invalid_argument(fmt::format("block channels is not exactly equal to the configured size: {} != {}", buffer.shape(2), _config.channels_per_sample()));
                }

                // populate buffer
                if (_log) { _log->trace("acquiring block {}", id); }
                while (loaded < buffer.count()) {

                    // calculate available elements
                    auto current = _in.tellg();
                    _in.seekg(0, std::ios_base::end);
                    auto available = (_in.tellg() - current) / sizeof(output_element_t);

                    // calculate read size to avoid reading past end of file
                    size_t needed = buffer.count() - loaded;
                    size_t n = std::min<size_t>(needed, available);

                    // load values
                    _in.seekg(current);
                    storage::detail::read_raw(_in, buffer.data() + loaded, n);
                    loaded += n;

                    // check for incomplete read
                    if (n < needed) {
                        // check if loopable
                        if (_config.loop) {
                            // restart
                            _in.seekg(0, std::ios_base::beg);
                            continue;
                        }
                    }

                    break;
                }

                // check for partial record
                if (loaded % record_size != 0) {
                    throw std::runtime_error(fmt::format("partial record detected: {} % {} != 0", loaded, record_size));
                }

            } catch (const std::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error during acquiring of block {}: {}", id, to_string(error)); }
            }

            // report result in number of records
            return std::make_tuple(loaded / record_size, error);
        }

        config_t _config;

        std::ifstream _in;

        std::shared_ptr<spdlog::logger> _log;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto acquisition(std::shared_ptr<vortex::acquire::file_acquisition_t<Args...>> a) {
        using adapter = adapter<block_t>;
        auto w = acquisition<block_t>(a, base_t());

        w.next_async = [a](block_t& block, typename adapter::spectra_stream_t& stream_, typename adapter::acquisition::callback_t&& callback) {
            std::visit([&](auto& stream) {
                try {
                    view_as_cpu([&](auto buffer) {
                        a->next_async(block.id, buffer.range(block.length), std::forward<typename adapter::acquisition::callback_t>(callback));
                    }, stream);
                } catch (const unsupported_view&) {
                    callback(0, std::current_exception());
                }
            }, stream_);
        };

        return w;
    }
}

#endif
