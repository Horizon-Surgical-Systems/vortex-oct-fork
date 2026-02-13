/** \rst

    CUDA-based OCT processor

    This is a highly-optimized CUDA-based OCT processor.  It is capable of concurrent
    processing in multiple streams and supports low-overhead dynamic updates for
    certain configuration items (e.g., complex filter).

    The component exposes a simple API for initialization and acquisition
    of blocks.  All work is handled in a background thread.  Both
    synchronous and asynchronous (via callbacks) options are available.

 \endrst */

#pragma once

#include <thread>
#include <functional>

#include <spdlog/spdlog.h>

#include <vortex/process/base.hpp>

#include <vortex/driver/cuda/fft.hpp>
#include <vortex/driver/cuda/copy.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/variant.hpp>

namespace vortex::process {

    namespace detail {

        void signed_cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 2>& out
        );

        void resample(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint32_t, 1>& before_index, const cuda::strided_t<const uint32_t, 1>& after_index,
            const cuda::strided_t<const float, 1>& before_weight, const cuda::strided_t<const float, 1>& after_weight,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 2>& out
        );
        void resample(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint32_t, 1>& before_index, const cuda::strided_t<const uint32_t, 1>& after_index,
            const cuda::strided_t<const float, 1>& before_weight, const cuda::strided_t<const float, 1>& after_weight,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<float, 2>& out
        );

        void resample_phase(
            const cuda::stream_t& stream,
            cuda::strided_t<const float, 2> phase, cuda::strided_t<const float, 1> phase_max,
            cuda::strided_t<const uint16_t, 2> in,
            cuda::strided_t<float, 2> out
        );
        void resample_phase(
            const cuda::stream_t& stream,
            cuda::strided_t<const float, 2> phase, cuda::strided_t<const float, 1> phase_max,
            cuda::strided_t<const float, 2> in,
            cuda::strided_t<float, 2> out
        );

        size_t prepare_sum(
            const cuda::strided_t<uint32_t, 2>& keys
        );
        void sum(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint32_t, 2>& keys,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<float, 1>& out_sum,
            const cuda::strided_t<uint32_t, 1>& out_count,
            void* scratch_ptr, size_t scratch_size
        );
        void sum(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint32_t, 2>& keys,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 1>& out_sum,
            const cuda::strided_t<uint32_t, 1>& out_count,
            void* scratch_ptr, size_t scratch_size
        );

        void compute_average_record(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 2>& average_record_buffer,
            const cuda::strided_t<float, 1>& average_record
        );

        void subtract_average_record(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& average,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 2>& out
        );
        void subtract_average_record(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& average,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<float, 2>& out
        );

        void complex_filter(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<const cuFloatComplex, 1>& filter,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );
        void complex_filter(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<const cuFloatComplex, 1>& filter,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );

        void cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );
        void cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );
        // needed for completeness but is just a copy
        void cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const cuFloatComplex, 2>& in,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );

        void copy(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<uint16_t, 2>& out
        );
        void copy(
            const cuda::stream_t& stream,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 2>& out
        );
        void copy(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<float, 2>& out
        );
        void copy(
            const cuda::stream_t& stream,
            const cuda::strided_t<const int8_t, 2>& in,
            const cuda::strided_t<int8_t, 2>& out
        );

#define _DECLARE(factor_t, in_t, out_t) \
            void abs_normalize( \
                const cuda::stream_t& stream, \
                factor_t factor, \
                bool enable_log10, bool enable_square, bool enable_magnitude, bool enable_levels, \
                factor_t level_min, factor_t level_max, \
                const cuda::strided_t<const in_t, 2>& in,\
                const cuda::strided_t<out_t, 2>& out \
            );

        _DECLARE(float, uint16_t, float);
        _DECLARE(float, float, float);
        _DECLARE(float, cuFloatComplex, float);
        _DECLARE(float, uint16_t, int8_t);
        _DECLARE(float, float, int8_t);
        _DECLARE(float, cuFloatComplex, int8_t);

        // NOTE: not yet ready to support double for the other functions but there is no reason it cannot be done
        //_DECLARE(double, uint16_t, double);
        //_DECLARE(double, double, double);
        //_DECLARE(double, cuDoubleComplex, double);
        //_DECLARE(double, uint16_t, int8_t);
        //_DECLARE(double, double, int8_t);
        //_DECLARE(double, cuDoubleComplex, int8_t);

#undef _DECLARE

        void demean_and_cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& sum, float divisor,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );
        void demean_and_cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& sum, float divisor,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<cuFloatComplex, 2>& out
        );
        void demean_and_cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& sum, float divisor,
            const cuda::strided_t<const uint16_t, 2>& in,
            const cuda::strided_t<float, 2>& out
        );
        void demean_and_cast(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 1>& sum, float divisor,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<float, 2>& out
        );

        void hilbert_window(
            const cuda::stream_t& stream,
            const cuda::strided_t<cuFloatComplex, 2>& inout
        );

        void phase_differences(const cuda::stream_t& stream,
            const cuda::strided_t<const cuFloatComplex, 2>& in,
            const cuda::strided_t<float, 2> out
        );

        size_t prepare_accumulate_max_phase(
            const cuda::strided_t<uint32_t, 2>& keys
        );
        void accumulate_max_phase(
            const cuda::stream_t& stream,
            const cuda::strided_t<const float, 2>& in,
            const cuda::strided_t<const uint32_t, 2>& keys,
            const cuda::strided_t<float, 2>& out_accum,
            const cuda::strided_t<float, 1>& out_max,
            const cuda::strided_t<uint32_t, 1>& out_count,
            void* scratch_ptr, size_t scratch_size
        );

    }

    template<typename T>
    struct cuda_processor_config_t : processor_config_t<T> {
        using element_t = T;

        int device = 0;
        size_t slots = 2;

        std::optional<size_t> clock_channel;
        bool interpret_as_signed = false;

        void validate() override {
            processor_config_t<T>::validate();

            if (slots < 1) {
                throw std::invalid_argument(fmt::format("minimum number of slots is 1: {}", slots));
            }

            // validate CUDA device index by attempting to set it active
            {
                std::exception_ptr error;
                auto prior = cuda::device();
                try {
                    cuda::device(device);
                } catch (const cuda::exception&) {
                    error = std::current_exception();
                }

                cuda::device(prior);
                if (error) {
                    std::rethrow_exception(error);
                }
            }

            if (clock_channel && resampling_samples.size() > 0) {
                throw std::runtime_error("static (resampling_samples) and dynamic (clock_channel) resampling are mutually exclusive");
            }
        }

        using processor_config_t<T>::resampling_samples;
    };

    template<typename input_element_t_, typename output_element_t_, typename float_element_t_, typename index_element_t_, typename config_t_>
    class cuda_processor_t : public processor_t<config_t_> {
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

            cuda::cuda_device_tensor_t<float_element_t> float_records, float_records_average, float_ascans, float_clock_sum, float_phase, float_phase_max;
            cuda::cuda_device_tensor_t<cuda::complex<float_element_t>> complex_ascans, complex_clock;

            cuda::cuda_device_tensor_t<uint8_t> scratch;

            cuda::event_t post_average;
            cuda::event_t done;

            cuda::stream_t stream;
            cuda::fft_plan_t<float_element_t> fft_plan;

            slot_t* prior = nullptr;
        };

        using job_t = std::function<void()>;

    public:

        cuda_processor_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)), _worker(&cuda_processor_t::_worker_loop, this) { }

        virtual ~cuda_processor_t() {
            _jobs.finish();
            _worker.join();
        }

        virtual void initialize(config_t config) {
            if (_log) { _log->debug("initializing CUDA OCT processor"); }

            config.validate();

            cudaDeviceProp prop;
            auto error = cudaGetDeviceProperties(&prop, config.device);
            if (error) {
                raise(_log, "unable to access device {}: {}", config.device, cudaGetErrorName(error));
            }
            if (_log) { _log->info("using {} (device {}) with compute capability {}.{} and {} bytes of memory", prop.name, config.device, prop.major, prop.minor, prop.totalGlobalMem); }

            std::swap(_config, config);
            _recalculate(config, false);
        }

        virtual void change(config_t new_config) {
            new_config.validate();

            std::swap(_config, new_config);
            _recalculate(new_config);
        }

        template<typename V1, typename V2>
        void next(const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, bool append_history = true) {
            next(0, input_buffer, output_buffer, append_history);
        }
        template<typename V1, typename V2>
        void next(size_t id, const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, bool append_history = true) {
            std::unique_lock<std::mutex> lock(_mutex);

            auto& slot = _next_slot();
            {
                // schedule processing
                auto error = _next_block(slot, id, input_buffer, output_buffer, nullptr, nullptr, append_history);
                if (error) {
                    std::rethrow_exception(error);
                }
            }

            {
                // wait for completion
                auto error = _wait_block(slot, id);
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }

        template<typename V1, typename V2>
        void next_async(const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, nullptr, nullptr, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, callback_t&& callback) {
            next_async(id, input_buffer, output_buffer, nullptr, nullptr, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, bool append_history, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, nullptr, nullptr, append_history, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, bool append_history, callback_t&& callback) {
            next_async(id, input_buffer, output_buffer, nullptr, nullptr, append_history, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, const cuda::event_t* start_event, const cuda::event_t* done_event, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, start_event, done_event, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, const cuda::event_t* start_event, const cuda::event_t* done_event, callback_t&& callback) {
            next_async(id, input_buffer, output_buffer, start_event, done_event, true, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, const cuda::event_t* start_event, const cuda::event_t* done_event, bool append_history, callback_t&& callback) {
            next_async(0, input_buffer, output_buffer, start_event, done_event, append_history, std::forward<callback_t>(callback));
        }
        template<typename V1, typename V2>
        void next_async(size_t id, const cuda::cuda_viewable<V1>& input_buffer, const cuda::cuda_viewable<V2>& output_buffer, const cuda::event_t* start_event, const cuda::event_t* done_event, bool append_history, callback_t&& callback) {
            std::unique_lock<std::mutex> lock(_mutex);

            // schedule processing
            auto& slot = _next_slot();
            // NOTE: perform error checking before dispatch because incomplete dispatch will lead to incorrect event signaling
            auto error = _next_block(slot, id, input_buffer, output_buffer, start_event, done_event, append_history);
            if (error) {
                // report error via callback
                _jobs.push([this, error, callback = std::forward<callback_t>(callback)]() {

#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        (void)this; // to satisfy clang
                        std::invoke(callback, error);
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (_log) { _log->critical("unhandled callback exception: {}\n{}", to_string(e), check_trace(e)); }
                    }
#endif

                });
            } else {
                _jobs.push([this, id, &slot, callback = std::forward<callback_t>(callback)]() {

                    // wait for completion
                    auto error = _wait_block(slot, id);

#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        std::invoke(callback, error);
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (_log) { _log->critical("unhandled callback exception: {}\n{}", to_string(e), check_trace(e)); }
                    }
#endif

                });
            }
        }

    protected:

        void _recalculate(const config_t& prior, bool lightweight = true) {
            cuda::device(_config.device);

            auto input_shape = head<2>(_config.input_shape());
            auto output_shape = head<2>(_config.output_shape());

            if (lightweight) {
                // lightweight updates do not...

                // ...require large buffer memory allocations/releases
                lightweight &= (_config.samples_per_ascan() <= prior.samples_per_ascan());
                lightweight &= (_config.ascans_per_block() <= prior.ascans_per_block());
                lightweight &= (_config.average_window <= prior.average_window);
                lightweight &= (_config.slots == prior.slots);

                // ...require planning an FFT
                lightweight &= !_config.enable_ifft || (_fft_plan_length == _config.samples_per_ascan());
            }

            if (lightweight) {
                if (_log) { _log->debug("beginning lightweight recalculation"); }

                // insert the update between ongoing work on the GPU
                for (auto& slot : _slots) {
                    _update_stream.wait(slot.done);
                }
            } else {
                if (_log) { _log->debug("beginning heavyweight recalculation"); }

                // wait for all jobs to finish here on the CPU before continuing
                for (auto& slot : _slots) {
                    slot.done.sync();
                }
            }

            // set up record averaging for constant frequency artifact removal
            if (_config.average_window > 0) {
                // TODO: adjust buffer while retaining history
                {
                    auto shape = std::array<size_t, 2>{ _config.average_window, _config.samples_per_record() };
                    if (_log) { _log->debug("allocating average record history buffer with [{}] elements", shape_to_string(shape)); }
                    _average_record_buffer.resize(shape);
                }
                if (!lightweight) {
                    // cannot guarantee that currently buffered records are valid
                    _average_record_index = _average_record_count = 0;
                }
            } else {
                // release buffer
                _average_record_buffer.clear();
                _average_record_index = _average_record_count = 0;
            }

            // setup keys for sums
            // NOTE: required prior to slot preparation
            //if (_config.average_window > 0) {
            //    // generate keys mapping
            //    xt::xtensor<index_element_t, 2> keys(input_shape);
            //    xt::view(keys, xt::all(), xt::all()) = xt::view(xt::arange(input_shape[1]), xt::newaxis(), xt::all());
            //    _sync_with_device(keys, _sample_keys);
            //} else {
            //    _sample_keys.clear();
            //}
            if (_config.clock_channel) {
                // generate keys mapping
                xt::xtensor<index_element_t, 2> keys(input_shape);
                xt::view(keys, xt::all(), xt::all()) = xt::view(xt::arange(input_shape[0]), xt::all(), xt::newaxis());
                _sync_with_device(keys, _record_keys);
            } else {
                _record_keys.clear();
            }
            if (_record_keys.valid() || _sample_keys.valid()) {
                // unused but required output buffer shared by all slots
                _unused_counts.resize({ std::max(input_shape[0], input_shape[1]) });
            } else {
                _unused_counts.clear();
            }

            // resampling
            if (_config.resampling_samples.size() > 0) {
                if (_config.samples_per_ascan() < 2) {
                    throw std::runtime_error(fmt::format("at least two samples per record required for resampling: {}", _config.samples_per_ascan()));
                }

                // the sample index before the requested index
                xt::xtensor<index_element_t, 1> before_index = xt::cast<index_element_t>(xt::floor(_config.resampling_samples));
                // the sample index after the requested index
                xt::xtensor<index_element_t, 1> after_index = before_index + 1;

                // compute the sampling ratio
                xt::xtensor<float_element_t, 1> after_weight = _config.resampling_samples - before_index;
                xt::xtensor<float_element_t, 1> before_weight = (float_element_t(1.0) - after_weight);

                // fix the edge cases
                for (size_t i = 0; i < _config.samples_per_ascan(); i++) {
                    if (before_index(i) < 0 || after_index(i) < 0) {
                        // clamp to first sample
                        before_index(i) = after_index(i) = 0;
                    } else if (before_index(i) >= downcast<ptrdiff_t>(_config.samples_per_record()) || after_index(i) >= downcast<ptrdiff_t>(_config.samples_per_record())) {
                        // clamp to last sample
                        before_index(i) = after_index(i) = downcast<index_element_t>(std::max<ptrdiff_t>(0, downcast<ptrdiff_t>(_config.samples_per_record()) - 1));
                    }
                }

                // transfer to GPU
                if (_log) { _log->debug("allocating resampling indices and weights"); }
                _sync_with_device(before_index, _resample_before_index);
                _sync_with_device(after_index, _resample_after_index);
                _sync_with_device(before_weight, _resample_before_weight);
                _sync_with_device(after_weight, _resample_after_weight);
                if (_log) { _log->debug("resampling buffers required {} bytes", _resample_before_index.size_in_bytes() + _resample_after_index.size_in_bytes() + _resample_before_weight.size_in_bytes() + _resample_after_weight.size_in_bytes()); }
            } else {
                // release
                _resample_before_index.clear();
                _resample_after_index.clear();
                _resample_before_weight.clear();
                _resample_after_weight.clear();
            }

            // spectral filtering
            if (_config.spectral_filter.size() > 0) {
                // transfer to GPU
                if (_log) { _log->debug("allocating complex filter with [{}] elements", shape_to_string(_config.spectral_filter.shape())); }
                _sync_with_device(_config.spectral_filter, _complex_filter);
            } else {
                // release
                _complex_filter.clear();
            }

            size_t new_fft_plan_length = _fft_plan_length;

            // set up slots
            if (!lightweight) {
                _slots.resize(_config.slots);

                size_t id = 0;
                for (auto& slot : _slots) {
                    slot.id = id++;
                    std::vector<size_t> per_record_shape = { input_shape[0] };
                    std::vector<size_t> per_sample_shape = { input_shape[1] };

                    size_t scratch_size = 0;

                    // allocate intermediate buffers
                    if (_config.average_window > 0 || _config.interpret_as_signed || _config.clock_channel) {
                        if (_log) { _log->debug("slot {} allocating spectral buffer with [{}] elements", slot.id, shape_to_string(input_shape)); }
                        slot.float_records.resize(input_shape);
                    } else {
                        slot.float_records.clear();
                    }
                    if (_config.average_window > 0) {
                        if (_log) { _log->debug("slot {} allocating spectral average buffer with [{}] elements", slot.id, shape_to_string(per_sample_shape)); }
                        slot.float_records_average.resize(per_sample_shape);

                        //// reserve scratch space
                        //scratch_size = std::max(scratch_size, detail::prepare_sum(view(_sample_keys)));
                    } else {
                        slot.float_records_average.clear();
                    }
                    if (_config.clock_channel) {
                        if (_log) { _log->debug("slot {} allocating clock buffer with [{}] elements", slot.id, shape_to_string(input_shape)); }
                        slot.complex_clock.resize(input_shape);

                        if (_log) { _log->debug("slot {} allocating clock average buffer with [{}] elements", slot.id, shape_to_string(per_record_shape)); }
                        slot.float_clock_sum.resize(per_record_shape);

                        if (_log) { _log->debug("slot {} allocating phase buffer with [{}] elements", slot.id, shape_to_string(input_shape)); }
                        slot.float_phase.resize(input_shape);

                        if (_log) { _log->debug("slot {} allocating phase max buffer with [{}] elements", slot.id, shape_to_string(per_record_shape)); }
                        slot.float_phase_max.resize(per_record_shape);

                        // reserve scratch space
                        scratch_size = std::max(scratch_size, detail::prepare_accumulate_max_phase(view(_record_keys)));
                        scratch_size = std::max(scratch_size, detail::prepare_sum(view(_record_keys)));
                    } else {
                        slot.complex_clock.clear();
                        slot.float_clock_sum.clear();
                        slot.float_phase.clear();
                        slot.float_phase_max.clear();
                    }
                    if (_config.resampling_samples.size() > 0 || _config.clock_channel) {
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

                    // allocate scratch space
                    if (scratch_size > 0) {
                        if (_log) { _log->debug("slot {} allocating scratch space with {} bytes", slot.id, scratch_size); }
                        slot.scratch.resize({ scratch_size });
                    } else {
                        slot.scratch.clear();
                    }

                    // FFT planning
                    if (_config.enable_ifft || _config.clock_channel) {
                        if (_fft_plan_length != _config.samples_per_ascan()) {
                            if (_log) { _log->debug("slot {} planning FFT of length {}", slot.id, _config.samples_per_ascan()); }
                            std::vector<int> fft_shape = { downcast<int>(_config.samples_per_ascan()) };

                            slot.fft_plan.plan_many(downcast<int>(_config.ascans_per_block()), fft_shape, &slot.stream);
                            new_fft_plan_length = _config.samples_per_ascan();
                        }
                    } else {
                        slot.fft_plan.destroy();
                        new_fft_plan_length = 0;
                    }
                }

                // update FFT length
                _fft_plan_length = new_fft_plan_length;

                // setup prior links
                if (_slots.size() > 1) {
                    for (size_t i = 0; i < _slots.size(); i++) {
                        _slots[(i + 1) % _slots.size()].prior = &_slots[i];
                    }
                }
            }

            // indicate that the update is complete
            _update_complete.record(_update_stream);

            if (_log) { _log->debug("recalculation complete for block size {} x {} (A-scans x samples)", _config.ascans_per_block(), _config.samples_per_ascan()); }
        }

        auto& _next_slot() {
            // wait for new slot to finish processing
            auto& slot = _slots[_next_slot_index];
            if (_log) { _log->trace("waiting for slot {}", slot.id); }

            // NOTE: this wait is needed to avoid recording the same CUDA event twice which might cause problems with stream synchronization
            slot.done.sync();

            // advance slots
            _next_slot_index = (_next_slot_index + 1) % _slots.size();

            return slot;
        }

        template<typename V1, typename V2>
        auto _next_block(slot_t& slot, size_t id, const cuda::cuda_viewable<V1>& input_buffer_, const cuda::cuda_viewable<V2>& output_buffer_, const cuda::event_t* start_event, const cuda::event_t* done_event, bool append_history) {
            const auto& input_buffer = input_buffer_.derived_cast();
            const auto& output_buffer = output_buffer_.derived_cast();
            std::exception_ptr error;

            try {
                // check that buffers are appropriate shape
                if (!shape_is_compatible(input_buffer.shape(), _config.input_shape())) {
                    throw std::runtime_error(fmt::format("input stream shape is not compatible with configured input shape: {} !~= {}", shape_to_string(input_buffer.shape()), shape_to_string(_config.input_shape())));
                }
                if (!shape_is_compatible(output_buffer.shape(), _config.output_shape())) {
                    throw std::runtime_error(fmt::format("output stream shape is not compatible with configured output shape: {} !~= {}", shape_to_string(output_buffer.shape()), shape_to_string(_config.output_shape())));
                }

                auto shaped_input_buffer = input_buffer.morph_right(3);
                auto shaped_output_buffer = output_buffer.morph_right(2);

                // perform the operation
                if (_config.clock_channel) {
                    _dispatch_block(slot, id, shaped_input_buffer.index_right({ _config.channel }), shaped_input_buffer.index_right({ *_config.clock_channel }), shaped_output_buffer, start_event, done_event, append_history);
                } else {
                    _dispatch_block(slot, id, shaped_input_buffer.index_right({ _config.channel }), {}, shaped_output_buffer, start_event, done_event, append_history);
                }
            } catch (const std::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error during dispatching for block {} (slot {}): {}", id, slot.id, to_string(error)); }
            }

            return error;
        }

        auto _dispatch_block(slot_t& slot, size_t id, const cuda::fixed_cuda_view_t<const input_element_t, 2>& input_buffer_, const cuda::fixed_cuda_view_t<const input_element_t, 2>& clock_buffer_, const cuda::fixed_cuda_view_t<output_element_t, 2>& output_buffer_, const cuda::event_t* start_event, const cuda::event_t* done_event, bool append_history) {
            const auto& input_buffer = input_buffer_.derived_cast();
            const auto& clock_buffer = clock_buffer_.derived_cast();
            const auto& output_buffer = output_buffer_.derived_cast();

            if (_log) { _log->trace("dispatching block {} (slot {})", id, slot.id); }
            cuda::device(_config.device);

            // wait for any updates to finish
            if (start_event) {
                slot.stream.wait(*start_event);
            }
            slot.stream.wait(_update_complete);

            auto records = std::min(input_buffer.shape(0), _config.records_per_block());

            // tracking of active source to simplify partial processing logic
            using real_memory_t = unique_variant<cuda::fixed_cuda_view_t<const input_element_t, 2>, cuda::cuda_view_t<input_element_t>, cuda::cuda_view_t<float_element_t>>;
            using complex_memory_t = unique_variant<cuda::fixed_cuda_view_t<const input_element_t, 2>, cuda::cuda_view_t<input_element_t>, cuda::cuda_view_t<float_element_t>, cuda::cuda_view_t<cuda::complex<float_element_t>>>;
            real_memory_t real_source = input_buffer;
            complex_memory_t complex_source = input_buffer;

            if (_config.interpret_as_signed) {
                auto input = input_buffer;
                auto output = view(slot.float_records);
                detail::signed_cast(slot.stream, input, output);
                real_source = output;
            }

            // wait for the prior job to finish averaging before modifying the history buffer
            if (slot.prior) {
                slot.stream.wait(slot.prior->post_average);
            }

            // average record
            if (_average_record_buffer.valid() && append_history) {
                auto history = view(_average_record_buffer);

                std::visit([&](auto& input) {
                    // update the history window
                    size_t count = std::min(records, _config.average_window);
                    size_t released = 0;

                    while (released < count) {
                        // never copy off the end of the window buffer
                        size_t available = std::min(_config.average_window - _average_record_index, count - released);
                        // start from the end of the input for recency
                        size_t index = records - count + released;

                        auto src = input.range(index, index + available);
                        auto dst = history.range(_average_record_index, _average_record_index + available);

                        // perform the copy
                        detail::copy(slot.stream, src, dst);

                        _average_record_index = (_average_record_index + available) % _config.average_window;
                        released += available;
                    }
                    _average_record_count = std::min(_average_record_count + count, _config.average_window);

                    // now update the average record
                    detail::compute_average_record(slot.stream, history.range(_average_record_count), slot.float_records_average);
                }, real_source);
            }
            slot.post_average.record(slot.stream);
            if (_average_record_buffer.valid() && _average_record_count > 0) {
                // remove the average from this block
                std::visit([&](auto& input) {
                    auto output = view(slot.float_records);
                    detail::subtract_average_record(slot.stream, slot.float_records_average, input, output);
                    real_source = output;
                }, real_source);
            }

            // NOTE: transition from working in samples_per_record above to samples_per_ascan below

            // resampling
            if (_config.resampling_samples.size() > 0) {
                auto output = view(slot.float_ascans);
                std::visit([&](auto& buffer) {
                    detail::resample(slot.stream, _resample_before_index, _resample_after_index, _resample_before_weight, _resample_after_weight, buffer, output);
                }, real_source);
                real_source = output;
            } else if (_config.clock_channel) {
                // convert to floating point
                if (_config.interpret_as_signed) {
                    detail::signed_cast(slot.stream, clock_buffer, slot.float_phase);
                } else {
                    detail::copy(slot.stream, clock_buffer, slot.float_phase);
                }

                // remove the clock mean and convert from real to complex
                detail::sum(slot.stream, _record_keys, slot.float_phase, slot.float_clock_sum, _unused_counts, slot.scratch.data(), slot.scratch.count());
                detail::demean_and_cast(slot.stream, slot.float_clock_sum, clock_buffer.shape(1), slot.float_phase, slot.complex_clock);

                // apply Hilbert transform
                // NOTE: cuFFT plans are set to a specific stream on creation
                // NOTE: length of clock and FFT guaranteed to match
                slot.fft_plan.forward(view(slot.complex_clock), view(slot.complex_clock));
                detail::hilbert_window(slot.stream, slot.complex_clock);
                slot.fft_plan.inverse(view(slot.complex_clock), view(slot.complex_clock));

                // compute phase differences and integrate into phase signal
                auto& tmp_phase = slot.float_records;
                detail::phase_differences(slot.stream, slot.complex_clock, tmp_phase);
                detail::accumulate_max_phase(slot.stream, tmp_phase, view(_record_keys), slot.float_phase, slot.float_phase_max, _unused_counts, slot.scratch.data(), slot.scratch.count());

                // perform resampling
                auto output = view(slot.float_ascans);
                std::visit([&](auto& buffer) {
                    detail::resample_phase(slot.stream, slot.float_phase, slot.float_phase_max, buffer, output);
                }, real_source);
                real_source = output;
            }

            // spectral filtering + dispersion compensation
            if (_complex_filter.count() > 0) {
                auto output = view(slot.complex_ascans);
                std::visit([&](auto& buffer) {
                    // NOTE: length of input and FFT guaranteed to match if operating directly on input
                    detail::complex_filter(slot.stream, buffer, _complex_filter, output);
                }, real_source);
                complex_source = output;
            } else {
                std::visit([&](auto& buffer) { complex_source = buffer; }, real_source);
            }

            // inverse FFT without normalization
            float_element_t normalization;
            if (_config.enable_ifft) {
                auto output = view(slot.complex_ascans);

                std::visit(overloaded{
                    [&](cuda::cuda_view_t<cuda::complex<float_element_t>>& buffer) {
                        if (buffer.is_contiguous()) {
                            // ready for FFT
                            // NOTE: cuFFT plans are set to a specific stream on creation
                            slot.fft_plan.inverse(buffer, output);
                        } else {
                            // deinterleave for FFT
                            // NOTE: length of input and FFT guaranteed to match if operating directly on input
                            detail::cast(slot.stream, buffer, output);

                            // NOTE: cuFFT plans are set to a specific stream on creation
                            slot.fft_plan.inverse(output, output);
                        }
                    },
                    [&](auto& buffer) {
                        // NOTE: length of input and FFT guaranteed to match if operating directly on input
                        detail::cast(slot.stream, buffer, output);

                        // NOTE: cuFFT plans are set to a specific stream on creation
                        slot.fft_plan.inverse(output, output);
                    }
                }, complex_source);
                complex_source = output;

                normalization = 1 / float_element_t(_config.samples_per_ascan());
            } else {
                normalization = 1;
            }

            float_element_t levels_min, levels_max;
            if (_config.levels) {
                levels_min = _config.levels->min();
                levels_max = _config.levels->max();
            }

            // log10 + abs + normalize + levels
            std::visit([&](auto& buffer) {
                detail::abs_normalize(slot.stream, normalization, _config.enable_log10, _config.enable_square, _config.enable_magnitude, !!_config.levels, levels_min, levels_max, buffer, output_buffer);
            }, complex_source);

            // set completion event
            slot.done.record(slot.stream);
            if (done_event) {
                done_event = &slot.done;
            }

            if (_log) { _log->trace("dispatched block {} (slot {})", id, slot.id); }
        }

        auto _wait_block(slot_t& slot, size_t id) {
            std::exception_ptr error;

            try {
                // wait until next job is done
                if (_log) { _log->trace("waiting for block {} (slot {})", id, slot.id); }
                slot.done.sync();
            } catch (const cuda::exception&) {
                error = std::current_exception();
                if (_log) { _log->error("error while (approximately) processing block {}: {}", id, to_string(error)); }
            }
            if (_log) { _log->trace("processed block {} (slot {})", id, slot.id); }

            return error;
        }

        void _worker_loop() {
            set_thread_name("CUDA Worker");
            setup_realtime();

            if (_log) { _log->debug("worker thread entered"); }
            cuda::device(_config.device);

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif
                job_t job;
                while (_jobs.pop(job)) {
                    std::invoke(job);
                }
#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in CUDA processor worker thread: {}\n{}", to_string(e), check_trace(e)); }
            }
#endif

            if (_log) { _log->debug("worker thread exited"); }
        }

        template<typename T, size_t N>
        void _sync_with_device(const xt::xtensor<T, N>& host, cuda::cuda_device_tensor_t<cuda::device_type<T>>& device) {
            if (!host.is_contiguous()) {
                throw std::runtime_error("non-contiguous arrays not yet supported");
            }

            if (host.size() > 0) {
                device.resize(host.shape());
                cuda::copy(view(host), view(device), &_update_stream);
            } else {
                device.clear();
            }
        }

        std::shared_ptr<spdlog::logger> _log;

        sync::queue_t<job_t> _jobs;
        std::thread _worker;

        std::mutex _mutex;

        cuda::event_t _update_complete;
        cuda::stream_t _update_stream;
        size_t _next_slot_index = 0;
        std::vector<slot_t> _slots;

        // FFT
        size_t _fft_plan_length = 0;

        // static resampling
        cuda::cuda_device_tensor_t<index_element_t> _resample_before_index, _resample_after_index;
        cuda::cuda_device_tensor_t<float_element_t> _resample_before_weight, _resample_after_weight;

        // dynamic resampling
        cuda::cuda_device_tensor_t<index_element_t> _record_keys, _unused_counts, _sample_keys;

        // averaging
        cuda::cuda_device_tensor_t<float_element_t> _average_record_buffer;
        size_t _average_record_index, _average_record_count;

        // spectral filtering and dispersion compensation
        cuda::cuda_device_tensor_t<cuda::complex<float_element_t>> _complex_filter;

        using base_t::_config;

    };

}
