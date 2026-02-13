#pragma once

#include <array>
#include <variant>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <xtensor/core/xnoalias.hpp>

#include <vortex/acquire/dso.hpp>

#include <vortex/driver/imaq.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/tuple.hpp>
#include <vortex/util/variant.hpp>

#include <vortex/core.hpp>

namespace vortex::imaq {

    struct line_trigger_t {
        uInt32 line = 0;
        uInt32 skip = 0;
        imaq::polarity_t polarity = imaq::polarity_t::high;
        imaq::signal_t signal = imaq::signal_t::external;

        template<typename board_t>
        void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
            if (log) { log->debug("configuring line trigger on line {} with polarity {}, signal {}, and skip {}", line, imaq::to_string(polarity), imaq::to_string(signal), skip); }
            board.configure_line_trigger(line, skip, polarity, signal);
        }
    };

    struct frame_trigger_t {
        uInt32 line = 0;
        imaq::polarity_t polarity = imaq::polarity_t::high;
        imaq::signal_t signal = imaq::signal_t::external;

        template<typename board_t>
        void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
            if (log) { log->debug("configuring frame trigger on line {} with polarity {} and signal {}", line, imaq::to_string(polarity), imaq::to_string(signal)); }
            board.configure_frame_trigger(line, polarity, signal);
        }
    };

    struct trigger_output_t {
        uInt32 line = 0;
        imaq::source_t source = imaq::source_t::hsync;
        imaq::polarity_t polarity = imaq::polarity_t::high;
        imaq::signal_t signal = imaq::signal_t::external;

        template<typename board_t>
        void apply(board_t& board, std::shared_ptr<spdlog::logger>& log) const {
            if (log) { log->debug("configuring trigger output {} on line {} with polarity {} and signal {}", imaq::to_string(source), line,  imaq::to_string(polarity), imaq::to_string(signal)); }
            board.configure_trigger_output(line, source, polarity, signal);
        }
    };

    template<typename... Ts>
    struct imaq_input_ts { };

    namespace detail {

        template<typename... Ts>
        struct to_ring_buffer {};
        template<typename... Ts>
        struct to_ring_buffer<imaq_input_ts<Ts...>> {
            using type = std::variant<cpu_tensor_t<Ts>...>;
        };

        template<typename... Ts>
        struct to_tuple {};
        template<typename... Ts>
        struct to_tuple<imaq_input_ts<Ts...>> {
            using type = std::tuple<Ts...>;
        };
    }

}

namespace vortex::acquire {

    struct imaq_config_t : dso_config_t {

        std::string device_name = "img0";

        std::array<size_t, 2> offset = { {0, 0} };

        auto& sample_offset() { return offset[0]; }
        const auto& sample_offset() const { return offset[0]; }
        auto& record_offset() { return offset[1]; }
        const auto& record_offset() const { return offset[1]; }

        std::optional<imaq::line_trigger_t> line_trigger;
        std::optional<imaq::frame_trigger_t> frame_trigger;
        std::vector<imaq::trigger_output_t> trigger_output = { imaq::trigger_output_t{} };

        size_t ring_size = 10;

        std::chrono::milliseconds acquire_timeout = std::chrono::seconds(1);

        bool stop_on_error = true;
        bool bypass_region_check = false;

        void validate() {
            if (ring_size == 0) {
                throw std::invalid_argument(fmt::format("ring size ({}) must be non-zero", ring_size));
            }
        }

    };

    template<typename config_t_, typename input_ts>
    class imaq_acquisition_t {
    public:

        using config_t = config_t_;
        using output_element_t = uint16_t;

        using callback_t = std::function<void(size_t, std::exception_ptr)>;

    protected:

        using ring_buffer_t = typename imaq::detail::to_ring_buffer<input_ts>::type;
        struct ring_dtype_t {
            bool compatible;
            size_t bits;
            std::string name;

            std::function<void*(const imaq::imaq_t&, ring_buffer_t&, std::array<size_t, 3>)> allocator;
        };
        using job_t = std::function<void()>;

    public:

        imaq_acquisition_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) {}

        virtual ~imaq_acquisition_t() {
            stop();

            if (_pool) {
                _pool->wait_finish();
            }
        }

        const config_t& config() const {
            return _config;
        }

        const imaq::imaq_t& imaq() const {
            return _imaq;
        }

        virtual void initialize(config_t config) {
            if (_log) { _log->debug("initializing IMAQ acquisition"); }

            // validate and accept the configuration
            config.validate();
            std::swap(_config, config);

            // create the IMAQ
            _imaq = imaq::imaq_t(_config.device_name);
            if (_log) { _log->info("using {}:{} ({}) with window ({}, {})-({} x {}) @ {} bits per pixel", _config.device_name, _imaq.info().device, _imaq.info().serial, _imaq.info().acquisition_window.top, _imaq.info().acquisition_window.left, _imaq.info().acquisition_window.height, _imaq.info().acquisition_window.width, _imaq.info().bits_per_pixel); }

            // validate datatype
            ring_dtype_t dtype = _validate_datatype();

            // configure acquisition region
            imaq::imaq_t::roi_t target_roi;
            target_roi.left = _config.sample_offset();
            target_roi.width = _config.samples_per_record();
            target_roi.top = _config.record_offset();
            target_roi.height = _config.records_per_block();

            auto acquire_roi = _imaq.fit_region(target_roi);
            if (acquire_roi.width < target_roi.width || acquire_roi.height < target_roi.height) {
                auto msg = fmt::format("requested region with offset ({}, {}) is larger than maximum: ({} x {}) > ({} x {}) (records x samples)", _config.record_offset(), _config.sample_offset(), _config.records_per_block(), _config.samples_per_record(), acquire_roi.height, acquire_roi.width);
                if (_config.bypass_region_check) {
                    if (_log) { _log->warn(msg); }
                } else {
                    throw std::runtime_error(msg);
                }
            }

            if (_log) { _log->debug("configuring acquire region with offset ({}, {}) and shape ({} x {})", acquire_roi.top, acquire_roi.left, acquire_roi.height, acquire_roi.width); }
            _imaq.configure_region(acquire_roi);

            _copy_roi.top = target_roi.top - acquire_roi.top;
            _copy_roi.left = target_roi.left - acquire_roi.left;
            _copy_roi.width = std::min<intmax_t>(target_roi.width, acquire_roi.width - _copy_roi.left);
            _copy_roi.height = std::min<intmax_t>(target_roi.height, acquire_roi.height - _copy_roi.top);
            if (_log) { _log->debug("configuring copy region with offset ({}, {}) and shape ({} x {})", _copy_roi.top, _copy_roi.left, _copy_roi.height, _copy_roi.width); }

            // configure signal options
            for (const auto& o : _config.trigger_output) {
                o.apply(_imaq, _log);
            }
            if (_config.line_trigger) {
                _config.line_trigger->apply(_imaq, _log);
            }
            if (_config.frame_trigger) {
                _config.frame_trigger->apply(_imaq, _log);
            }

            // configure frame timeout
            _imaq.configure_frame_timeout(_config.acquire_timeout);

            // allocate and configure ring buffer
            std::array<size_t, 3> shape = { { acquire_roi.height, acquire_roi.width, _config.channels_per_sample() } };
            if (_log) { _log->debug("allocating {} intermediate buffers of type {} and shape [{}] for ring buffer", _config.ring_size, _to_string(dtype), shape_to_string(shape)); }

            _ring.resize(_config.ring_size);
            std::vector<void*> ptrs;
            for (auto& buffer : _ring) {
                ptrs.push_back(dtype.allocator(_imaq, buffer, shape));
            }

            if (_log) { _log->debug("configuring continuous acqusition with ring buffer of size {}", ptrs.size()); }
            _imaq.configure_ring(ptrs);

            // launch worker pool
            _pool.emplace("IMAQ Acquisition", 1, [](size_t) { setup_realtime(); }, _log);
        }

    protected:

        ring_dtype_t _validate_datatype() {
            // consider all possible types
            std::vector<ring_dtype_t> types;
            for_each_tuple(typename imaq::detail::to_tuple<input_ts>::type{}, [&](const auto& o) {
                // extract type information
                using T = std::decay_t<decltype(o)>;

                // collect details
                auto bits = 8 * sizeof(T);
                // NOTE: IMAQ expands 12 bit acquisitions into 16 bits internally
                // ref: https://forums.ni.com/t5/Machine-Vision/What-happens-to-a-12-bit-image-in-LabVIEW-IMAQ-Vision/td-p/52335
                auto compatible = bits >= _imaq.info().bits_per_pixel;
                auto allocator = [](const imaq::imaq_t& imaq, ring_buffer_t& rb, std::array<size_t, 3> shape) -> void* {
                    // allocate
                    cpu_tensor_t<T> buf;
                    buf.resize(shape);

                    // check that buffer sized appropriately
                    if (buf.size_in_bytes() != imaq.required_buffer_size()) {
                        throw std::runtime_error(fmt::format("mismatch between allocated and required buffer size: {} != {} (suggests internal error)", buf.size_in_bytes(), imaq.required_buffer_size()));
                    }

                    // save pointer for later
                    auto ptr = buf.data();

                    // move to ring buffer
                    rb = std::move(buf);
                    return ptr;
                };

                // store details
                types.push_back({ compatible, bits, typeid(T).name(), allocator });
            });

            // partition by match or not
            auto it = std::partition(types.begin(), types.end(), [](const auto& o) { return o.compatible; });

            if (it == types.begin()) {
                // unsupported bits per pixel
                std::vector<std::string> unsupported;
                std::transform(it, types.end(), std::back_inserter(unsupported), [](const auto& o) { return _to_string(o); });

                throw std::invalid_argument(fmt::format("IMAQ device {} bytes per pixel of {} does not match any acquisition datatype: ", _config.device_name, _imaq.info().bytes_per_pixel, join(unsupported, ", ")));
            }

            // choose the smallest possible datatype
            return *std::min_element(types.begin(), it, [](const auto& a, const auto& b) { return a.bits < b.bits; });
        }

        static auto _to_string(const ring_dtype_t& dtype) {
            return fmt::format("{} ({} bits)", dtype.name, dtype.bits);
        }

    public:

        virtual void prepare() {

        }

        void start() {
            std::unique_lock<std::mutex> lock(_mutex);

            if (!running()) {
                if (_log) { _log->info("starting acquisition"); }
                _frame_count = 0;
                _imaq.start_capture();
            }
        }

        void stop() {
            _stop(false);
        }

        bool running() const {
            return _imaq.running();
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

    protected:

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

                // lock the next frame
                // NOTE: throws when requested index is not available
                auto frame = _imaq.lock_frame(downcast<uInt32>(_frame_count++));

                // copy to block buffer
                std::visit([&](auto& input_buffer) {
                    auto src = xt::view(view(input_buffer).to_xt(), xt::range(_copy_roi.top, _copy_roi.top + _copy_roi.height), xt::range(_copy_roi.left, _copy_roi.left + _copy_roi.width), xt::all());
                    auto dst = xt::view(output_buffer.to_xt(), xt::range(_copy_roi.top, _copy_roi.top + _copy_roi.height), xt::range(_copy_roi.left, _copy_roi.left + _copy_roi.width), xt::all());
                    xt::noalias(dst) = src;
                }, _ring[frame.actual_index % _ring.size()]);

                // report full block acquisition
                n = _config.records_per_block();
            } catch (const imaq::exception&) {
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

            if (_imaq.running()) {
                if (_log) { _log->info("stopping acquisition"); }
                try {
                    _imaq.stop_capture();
                } catch (const imaq::exception& e) {
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

        std::vector<ring_buffer_t> _ring;
        imaq::imaq_t::roi_t _copy_roi;
        imaq::imaq_t _imaq;

        std::atomic_bool _abort = false;
        std::atomic<size_t> _frame_count;
        std::optional<util::worker_pool_t> _pool;
        std::mutex _mutex;

        config_t _config;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto acquisition(std::shared_ptr<vortex::acquire::imaq_acquisition_t<Args...>> a) {
        using adapter = adapter<block_t>;
        auto w = acquisition<block_t>(a, base_t());

        w.stream_factory = []() {
            return []() -> typename adapter::spectra_stream_t {
                return sync::lockable<cuda::cuda_host_tensor_t<typename block_t::acquire_element_t>>();
            };
        };

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
