/** \rst

    CPU-based OCT processor

    This OCT processor uses xtensor and FFTW to perform OCT processing without
    CUDA support. It is intended for debugging purposes only or for use if
    non-CUDA OCT processing must be done (e.g., on a laptop without CUDA
    capability).

 \endrst */

#pragma once

#include <thread>
#include <functional>

#include <tbb/parallel_for.h>

#include <xtensor/views/xindex_view.hpp>
#include <xtensor/misc/xcomplex.hpp>

#include <spdlog/spdlog.h>

#include <vortex/process/base.hpp>

#include <vortex/driver/fftw.hpp>

#include <vortex/memory/cpu.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/thread.hpp>
#include <vortex/util/timing.hpp>

// #define VORTEX_TIMING_CPU_PROCESSOR

namespace vortex::process {

    template<typename T>
    struct cpu_processor_config_t : processor_config_t<T> {
        using element_t = T;

        size_t slots = 2;

        void validate() override {
            processor_config_t<T>::validate();

            if (slots < 1) {
                throw std::invalid_argument(fmt::format("minimum number of slots is 1: {}", slots));
            }
        }

    };

    template<typename input_element_t_, typename output_element_t_, typename float_element_t_, typename index_element_t_, typename config_t_>
    class cpu_processor_t : public processor_t<config_t_> {
    public:

        using base_t = processor_t<config_t_>;
        using config_t = config_t_;

        using input_element_t = input_element_t_;
        using output_element_t = output_element_t_;
        using float_element_t = float_element_t_;
        using index_element_t = index_element_t_;

        static_assert(std::is_same_v<typename config_t::element_t, float_element_t>, "configuration type for resampling and filtering must match the float element type");

        using callback_t = std::function<void(std::exception_ptr)>;

    protected:

        struct slot_t {
            size_t id;

            cpu_tensor_t<float_element_t> float_records, float_ascans;
            cpu_tensor_t<std::complex<float_element_t>> complex_ascans;

            // use pointer because event is not copyable or movable (uses a condition variable)
            std::shared_ptr<sync::event_t> average_ready, done;

            fft::fftw_plan_t<float_element_t> ifft;

            slot_t* next = nullptr;

            slot_t() {
                average_ready = std::make_shared<sync::event_t>();
                done = std::make_shared<sync::event_t>();
            }
        };

        using job_t = std::function<void()>;

    public:

        cpu_processor_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~cpu_processor_t() {
            if (_pool) {
                _pool->wait_finish();
            }

#if defined(VORTEX_TIMING_CPU_PROCESSOR)
            timing.write(std::cout);
#endif
        }

        virtual void initialize(config_t config) {
            if (_log) { _log->debug("initializing CPU OCT processor"); }

            // check and accept configuration
            config.validate();
            std::swap(_config, config);

            // allocate internal resources
            _recalculate();

            // launch worker pool
            _pool.emplace("CPU OCT", _config.slots, [](size_t) { setup_realtime(); }, _log);
        }

        virtual void change(config_t new_config) {
            // check and accept configuration
            new_config.validate();
            std::swap(_config, new_config);

            // update interal resources
            _recalculate();
        }

        template<typename V1, typename V2>
        void next(const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, bool append_history = true) {
            next(0, input_buffer, output_buffer, append_history);
        }
        template<typename V1, typename V2>
        void next(size_t id, const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, bool append_history = true) {
            std::unique_lock<std::mutex> lock(_mutex);

            std::exception_ptr error;
            sync::event_t done;

            // schedule processing
            _next_async(id, input_buffer, output_buffer, append_history, [&](std::exception_ptr error_) {
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
            next_async(0, input_buffer, output_buffer, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, callback_t&& callback) {
            next_async(id, input_buffer, output_buffer, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, bool append_history, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, append_history, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cpu_viewable<V1>& input_buffer, const cpu_viewable<V2>& output_buffer, bool append_history, callback_t&& callback) {
            std::unique_lock<std::mutex> lock(_mutex);
            _next_async(id, input_buffer, output_buffer, append_history, std::forward<callback_t>(callback));
        }

    protected:

        template<typename V1, typename V2>
        void _next_async(size_t id, const cpu_viewable<V1>& input_buffer_, const cpu_viewable<V2>& output_buffer_, bool append_history, callback_t&& callback) {

            const auto& input_buffer = input_buffer_.derived_cast();
            const auto& output_buffer = output_buffer_.derived_cast();

            // validate processing
            // NOTE: obtain slot now synchronously because thread pool tasks may run ahead of each other, yielding slot wait deadlock
            auto slot = &_next_slot();

            _pool->post([this, id, slot, input_buffer, output_buffer, append_history]() {

                std::exception_ptr error;
                try {
                    // perform the processing
                    _process(*slot, id, input_buffer, output_buffer, append_history);
                } catch (const std::exception&) {
                    error = std::current_exception();
                    if (_log) { _log->error("error during processing for block {}: {}", id, to_string(error)); }
                }
                return std::make_tuple(error);

            }, std::forward<callback_t>(callback));

        }

        void _recalculate() {
            // indicate that update is in progress
            _update_complete.unset();

            // set up A-scan averaging for constant frequency artifact removal
            if (_config.average_window > 0) {
                // TODO: adjust buffer while retaining history
                {
                    auto shape = std::array<size_t, 2>{ _config.average_window, _config.samples_per_record() };
                    if (_log) { _log->debug("allocating average record history buffer with [{}] elements", shape_to_string(shape)); }
                    _average_record_buffer.resize(shape);
                }
                {
                    auto shape = std::array<size_t, 2>{ 1, _config.samples_per_record() };
                    if (_log) { _log->debug("allocating average record with [{}] elements", shape_to_string(shape)); }
                    _average_record.resize(shape);
                }

                _average_record_index = _average_record_count = 0;

            } else {
                // release buffer
                _average_record_buffer.clear();
                _average_record.clear();
                _average_record_index = _average_record_count = 0;
            }

            // set up slots
            {
                _slots.resize(_config.slots);
                auto input_shape = head<2>(_config.input_shape());
                auto output_shape = head<2>(_config.output_shape());

                size_t id = 0;
                for (auto& slot : _slots) {
                    slot.id = id++;

                    // allocate intermediate buffers
                    if (_config.average_window > 0) {
                        if (_log) { _log->debug("slot {} allocating average buffer with [{}] elements", slot.id, shape_to_string(input_shape)); }
                        slot.float_records.resize(input_shape);
                    } else {
                        slot.float_records.clear();
                    }
                    if (_config.resampling_samples.size() > 0) {
                        if (_log) { _log->debug("slot {} allocating resampling buffer with [{}] elements", slot.id, shape_to_string(output_shape)); }
                        slot.float_ascans.resize(output_shape);
                    } else {
                        slot.float_ascans.clear();
                    }
                    if (_config.spectral_filter.size() > 0 || _config.enable_ifft) {
                        if (_log) { _log->debug("slot {} allocating complex buffer with [{}] elements", slot.id, shape_to_string(output_shape)); }
                        slot.complex_ascans.resize(output_shape);
                    } else {
                        slot.complex_ascans.clear();
                    }

                    // FFT planning
                    if (_config.enable_ifft) {
                        if (_log) { _log->debug("slot {} planning FFT", slot.id); }
                        std::vector<int> fft_shape = { downcast<int>(_config.samples_per_ascan()) };

                        // NOTE: FFTW requires fully allocated buffers for plan generation
                        auto n = downcast<int>(_config.samples_per_ascan());
                        slot.ifft.inverse(
                            downcast<int>(_config.ascans_per_block()),               // number of FFTs
                            { 1, 1 },                                                // stride between successive samples
                            { n, n },                                                // stride between successive A-scans
                            { n },                                                   // length of FFT
                            slot.complex_ascans.data(), slot.complex_ascans.data()   // data pointers
                        );
                    } else {
                        slot.ifft.destroy();
                    }

                    // slots start in done state
                    slot.done->set();
                }

                // only first slot is ready for average
                _slots[0].average_ready->set();

                // setup next links
                if (_slots.size() > 1) {
                    for (size_t i = 0; i < _slots.size(); i++) {
                        _slots[i].next = &_slots[(i + 1) % _slots.size()];
                    }
                }
            }

            // resampling
            if (_config.resampling_samples.size() > 0) {
                if (_config.samples_per_ascan() < 2) {
                    throw std::runtime_error(fmt::format("at least two samples per record required for resampling: {}", _config.samples_per_ascan()));
                }

                // the sample index before the requested index
                _resample_before_index = xt::cast<index_element_t>(xt::floor(_config.resampling_samples));
                // the sample index after the requested index
                _resample_after_index = _resample_before_index + 1;

                // compute the sampling ratio
                _resample_after_weight = _config.resampling_samples - _resample_before_index;
                _resample_before_weight = (float_element_t(1.0) - _resample_after_weight);

                // fix the edge cases
                for (size_t i = 0; i < _config.samples_per_ascan(); i++) {
                    if (_resample_before_index(i) < 0 || _resample_after_index(i) < 0) {
                        // clamp to first sample
                        _resample_before_index(i) = _resample_after_index(i) = 0;
                    } else if (_resample_before_index(i) >= downcast<ptrdiff_t>(_config.samples_per_record()) || _resample_after_index(i) >= downcast<ptrdiff_t>(_config.samples_per_record())) {
                        // clamp to last sample
                        _resample_before_index(i) = _resample_after_index(i) = downcast<index_element_t>(std::max(ptrdiff_t(0), downcast<ptrdiff_t>(_config.samples_per_record()) - 1));
                    }
                }
            } else {
                // release
                _resample_before_index = {};
                _resample_after_index = {};
                _resample_before_weight = {};
                _resample_after_weight = {};
            }

            // indicate that the update is complete
            _update_complete.set();

            if (_log) { _log->debug("recalculation complete for block size {} x {} (A-scans x samples)", _config.ascans_per_block(), _config.samples_per_ascan()); }
        }

        auto& _next_slot() {
            // wait for new slot to finish processing
            auto& slot = _slots[_next_slot_index];
            if (_log) { _log->trace("waiting for slot {}", slot.id); }

            slot.done->wait();

            // reset slot waits
            slot.done->unset();

            // advance slots
            _next_slot_index = (_next_slot_index + 1) % _slots.size();

            return slot;
        }

        template<typename V1, typename V2>
        void _process(slot_t& slot, size_t id, const cpu_viewable<V1>& input_buffer_, const cpu_viewable<V2>& output_buffer_, bool append_history) {
            auto& input_buffer = input_buffer_.derived_cast();
            auto& output_buffer = output_buffer_.derived_cast();

            if (_log) { _log->trace("processing block {} (slot {})", id, slot.id); }

            auto count = std::min(input_buffer.shape(0), output_buffer.shape(0));
            auto records = std::min(count, _config.records_per_block());
            // for convenience and consistency
            auto& ascans = records;

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

            // wait for any updates to finish
            _update_complete.wait();

            // view the input buffer
            cpu_view_t<const input_element_t> input = input_buffer.morph_right(3).index_right({ _config.channel });

            // tracking of active source to simplify partial processing logic
            using real_memory_t = unique_variant<cpu_view_t<const input_element_t>, cpu_view_t<input_element_t>, cpu_view_t<float_element_t>>;
            using complex_memory_t = unique_variant<cpu_view_t<const input_element_t>, cpu_view_t<input_element_t>, cpu_view_t<float_element_t>, cpu_view_t<std::complex<float_element_t>>>;
            real_memory_t real_source = input;
            complex_memory_t complex_source = input;

            // wait for the prior job to finish averaging before modifying the history buffer
            slot.average_ready->wait();
            if (slot.next) {
                // clear wait for next slot usage
                slot.average_ready->unset();
            }

            // average A-scan
            if (_average_record.count() > 0) {
                if (append_history) {
                    {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                        util::stopwatch_t sw(timing.book["history"]);
#endif
                        // update the history window
                        size_t count = std::min(records, _average_record_buffer.shape(0));
                        size_t released = 0;

                        while (released < count) {
                            // never copy off the end of the window buffer
                            size_t available = std::min(_average_record_buffer.shape(0) - _average_record_index, count - released);
                            // start from the end of the mag for recency
                            size_t index = records - count + released;

                            // perform the copy
                            auto dst = xt::view(view(_average_record_buffer).to_xt(), xt::range(_average_record_index, _average_record_index + available), xt::all());
                            auto src = xt::view(input.to_xt(), xt::range(index, index + available), xt::all());
                            dst = src;

                            _average_record_index = (_average_record_index + available) % _average_record_buffer.shape(0);
                            released += available;
                        }
                        _average_record_count = std::min(_average_record_count + count, _average_record_buffer.shape(0));
                    }

                    // now update the average record
                    {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                        util::stopwatch_t sw(timing.book["mean"]);
#endif
                        auto src = xt::view(view(_average_record_buffer).to_xt(), xt::range(0, _average_record_count), xt::all());
                        auto dst = view(_average_record).to_xt();
                        dst = xt::mean(src, 0, xt::evaluation_strategy::immediate);
                    }
                }

                // remove the average from this block
                if (_average_record_count > 0) {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                    util::stopwatch_t sw(timing.book["demean"]);
#endif
                    view(slot.float_records).to_xt() = input.to_xt() - xt::view(view(_average_record).to_xt(), xt::newaxis(), xt::all(), xt::newaxis());
                    real_source = view(slot.float_records);
                }
            }
            // indicate that next slot may proceed with averaging
            if (slot.next) {
                slot.next->average_ready->set();
            }

            // NOTE: transition from working in samples_per_record above to samples_per_ascan below

            // resample
            if (_config.resampling_samples.size() > 0) {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                util::stopwatch_t sw(timing.book["resample"]);
#endif
                std::visit([&](auto& buffer) {
                    // avoid temporary to avoid constness that will prevent assignment below
                    auto dst = view(slot.float_ascans).to_xt();

                    // resample each record
                    tbb::parallel_for(ptrdiff_t(0), downcast<ptrdiff_t>(records), [&](ptrdiff_t r) {
                        auto before = xt::index_view(xt::view(buffer.to_xt(), r, xt::all()), _resample_before_index);
                        auto after = xt::index_view(xt::view(buffer.to_xt(), r, xt::all()), _resample_after_index);
                        xt::view(dst, r, xt::all()) = _resample_before_weight * before + _resample_after_weight * after;
                    });
                }, real_source);
                real_source = view(slot.float_ascans);
            }

            // spectral filtering + dispersion compensation
            if (_config.spectral_filter.size() > 0) {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                util::stopwatch_t sw(timing.book["filter"]);
#endif
                std::visit([&](auto& buffer) {
                    // NOTE: length of input and FFT guaranteed to match if operating directly on input
                    view(slot.complex_ascans).to_xt() = xt::cast<std::complex<float_element_t>>(buffer.to_xt()) * xt::view(_config.spectral_filter, xt::newaxis(), xt::all());
                 }, real_source);
                complex_source = view(slot.complex_ascans);
            } else {
                std::visit([&](auto& buffer) { complex_source = buffer; }, real_source);
            }

            // inverse FFT without normalization
            float_element_t normalization;
            if (_config.enable_ifft) {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                util::stopwatch_t sw(timing.book["fft"]);
#endif
                std::visit(overloaded{
                    [&](cpu_view_t<std::complex<float_element_t>>& buffer) {
                        if (buffer.is_contiguous()) {
                            // ready for FFT
                            // NOTE: FFT operates on the slot buffers because the number of planned FFTs is fixed
                            slot.ifft.execute(buffer.data(), slot.complex_ascans.data());
                        } else {
                            // NOTE: length of input and FFT guaranteed to match if operating directly on input
                            view(slot.complex_ascans).to_xt() = xt::cast<std::complex<float_element_t>>(buffer.to_xt());

                            // NOTE: FFT operates on the slot buffers because the number of planned FFTs is fixed
                            slot.ifft.execute(slot.complex_ascans.data(), slot.complex_ascans.data());
                        }
                    },
                    [&](auto& buffer) {
                        // NOTE: length of input and FFT guaranteed to match if operating directly on input
                        view(slot.complex_ascans).to_xt() = xt::cast<std::complex<float_element_t>>(buffer.to_xt());

                        // NOTE: FFT operates on the slot buffers because the number of planned FFTs is fixed
                        slot.ifft.execute(slot.complex_ascans.data(), slot.complex_ascans.data());
                    }
                }, complex_source);
                complex_source = view(slot.complex_ascans);

                normalization = 1 / float_element_t(_config.samples_per_ascan());
            } else {
                normalization = 1;
            }

            // view the output buffer
            cpu_view_t<output_element_t> output = output_buffer.morph_right(2);

            // log10 + abs + normalize
            {
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
                util::stopwatch_t sw(timing.book["log10_abs"]);
#endif

                auto _round_clip_cast_assign = [&](auto&& in, auto&& out) {
                    if constexpr (std::is_floating_point_v<output_element_t>) {
                        out = in;
                    } else {
                        constexpr auto min = std::numeric_limits<output_element_t>::lowest();
                        constexpr auto max = std::numeric_limits<output_element_t>::max();

                        out = xt::cast<output_element_t>(xt::clip(xt::round(in), float_element_t(min), float_element_t(max)));
                    }
                };

                std::visit([&](auto& buffer) {
                    if (_config.levels) {
                        float_element_t scale, offset;

                        if constexpr (std::is_integral_v<output_element_t>) {
                            // map to [levels_min, levels_max]
                            constexpr auto min = std::numeric_limits<output_element_t>::lowest();
                            constexpr auto max = std::numeric_limits<output_element_t>::max();
                            scale = float_element_t(max - min) / _config.levels->length();
                            offset = -_config.levels->min() * scale + min;
                        } else {
                            // scale to [0, 1]
                            scale = float_element_t(1) / _config.levels->length();
                            offset = -_config.levels->min() * scale;
                        }

                        if (_config.enable_log10) {
                            if (_config.enable_magnitude) {
                                _round_clip_cast_assign(((_config.enable_square ? 20 : 10) * scale) * xt::log10(xt::abs(buffer.to_xt())) + (offset + std::log10(normalization)), output.to_xt());
                            } else {
                                _round_clip_cast_assign(((_config.enable_square ? 20 : 10) * scale) * xt::log10(xt::real(buffer.to_xt())) + (offset + std::log10(normalization)), output.to_xt());
                            }
                        } else if (_config.enable_square) {
                            _round_clip_cast_assign(xt::square((normalization * std::sqrt(scale)) * xt::abs(buffer.to_xt())) + offset, output.to_xt());
                        } else {
                            if (_config.enable_magnitude) {
                                _round_clip_cast_assign((scale * normalization) * xt::abs(buffer.to_xt()) + offset, output.to_xt());
                            } else {
                                _round_clip_cast_assign((scale * normalization) * xt::real(buffer.to_xt()) + offset, output.to_xt());
                            }
                        }
                    } else {
                        if (_config.enable_log10) {
                            if (_config.enable_magnitude) {
                                _round_clip_cast_assign((_config.enable_square ? 20 : 10) * xt::log10(normalization * xt::abs(buffer.to_xt())), output.to_xt());
                            } else {
                                _round_clip_cast_assign((_config.enable_square ? 20 : 10) * xt::log10(normalization * xt::real(buffer.to_xt())), output.to_xt());
                            }
                        } else if (_config.enable_square) {
                            _round_clip_cast_assign(xt::square(normalization * xt::abs(buffer.to_xt())), output.to_xt());
                        } else {
                            if (_config.enable_magnitude) {
                                _round_clip_cast_assign(normalization * xt::abs(buffer.to_xt()), output.to_xt());
                            } else {
                                _round_clip_cast_assign(normalization * xt::real(buffer.to_xt()), output.to_xt());
                            }
                        }
                    }
                }, complex_source);
            }

            // set completion event
            slot.done->set();

            if (_log) { _log->trace("processed block {} (slot {})", id, slot.id); }
#if defined(VORTEX_TIMING_CPU_PROCESSOR)
            timing.count += 1;
#endif
        }

        std::shared_ptr<spdlog::logger> _log;

        std::optional<util::completion_worker_pool_t<std::exception_ptr>> _pool;

        std::mutex _mutex;

        sync::event_t _update_complete;
        size_t _next_slot_index = 0;
        std::vector<slot_t> _slots;

        // resampling
        xt::xtensor<float_element_t, 1> _resample_before_weight, _resample_after_weight;
        xt::xtensor<index_element_t, 1> _resample_before_index, _resample_after_index;

        // averaging
        cpu_tensor_t<input_element_t> _average_record_buffer;
        cpu_tensor_t<float_element_t> _average_record;
        size_t _average_record_index, _average_record_count;

        using base_t::_config;

    public:

#if defined(VORTEX_TIMING_CPU_PROCESSOR)
        util::timing_book_t timing;
#endif

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto processor(std::shared_ptr<vortex::process::cpu_processor_t<Args...>> a) {
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
