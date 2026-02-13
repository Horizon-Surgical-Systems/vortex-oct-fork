#pragma once

#include <vortex/acquire/dso/alazar/host.hpp>

namespace vortex::acquire {

    template<typename clock_t, typename trigger_t, typename option_t>
    struct alazar_fft_config_t : alazar_config_t<clock_t, trigger_t, option_t> {
        using base_t = alazar_config_t<clock_t, trigger_t, option_t>;
        using base_t::device, base_t::records_per_block, base_t::samples_per_record, base_t::channels_per_sample, base_t::bytes_per_multisample;

        size_t fft_length = 1024;
        xt::xtensor<std::complex<float>, 1> spectral_filter;
        bool include_time_domain = false;

        xt::xtensor<int16_t, 1> background;

        std::array<size_t, 3> shape() const override { return { ascans_per_block(), samples_per_ascan(), channels_per_sample() }; }

        size_t buffer_bytes_per_record() const override { return samples_per_ascan() * bytes_per_multisample(); }

        size_t samples_per_ascan() const { return _fft_record_length(); }

        size_t& ascans_per_block() { return records_per_block(); }
        const size_t& ascans_per_block() const { return records_per_block(); }

        void validate() override {
            base_t::validate();

            auto board = base_t::create_board();

            // ensure spectral filter matches the FFT length
            if (spectral_filter.size() > 0 && spectral_filter.size() != fft_length) {
                throw std::invalid_argument(fmt::format("spectral filter and FFT length mismatch: {} vs {}", spectral_filter.size(), fft_length));
            }

            // ensure background record matches FFT length
            if (background.size() > 0 && background.size() != fft_length) {
                throw std::invalid_argument(fmt::format("background and FFT length mismatch: {} vs {}", background.size(), fft_length));
            }

            // enforce maximum FFT length
            auto max_fft_length = board.info().find_fft_info().max_record_length;
            if (fft_length > max_fft_length) {
                throw std::invalid_argument(fmt::format("FFT length exceeds maximum size: {} > {}", fft_length, max_fft_length));
            }

            // check if FFT length power of two
            // ref: https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
            bool po2 = fft_length && !(fft_length & (fft_length - 1));
            if (!po2) {
                auto next_po2 = 1uLL << size_t(std::ceil(std::log2(fft_length)));
                throw std::invalid_argument(fmt::format("samples/A-scan (FFT length) is not a power of 2: {} (consider {})", fft_length, next_po2));
            }

            // check if time domain is supported
            if (include_time_domain && !board.info().find_fft_info().fft_time_domain_supported) {
                throw std::invalid_argument("FFT with time-domain records is not supported by the hardware or the compiled " ALAZAR_LIBRARY_VERSION_STRING);
            }
        }

        void apply(alazar::board_t& board, std::shared_ptr<spdlog::logger>& log) override {
            base_t::apply(board, log);

            // configure window
            if (spectral_filter.size() > 0) {
                log->debug("configuring spectral filter of length {} samples", spectral_filter.size());
                board.configure_fft_window(spectral_filter.size(), spectral_filter.data());
            } else {
                // default is flattop
                log->debug("configuring flat-top spectral filter");
                board.configure_fft_window(fft_length);
            }

            // configure background
            if (background.size() > 0) {
                log->debug("configuring background subtraction of length {} samples", background.size());
                board.configure_fft_background_subtraction(background.size(), background.data());
            } else {
                // disable
                board.configure_fft_background_subtraction();
            }
        }

    protected:

        auto _fft_record_length() const {
            if (include_time_domain) {
                // NOTE: samples_per_record is padded to FFT length for time-domain output
                return fft_length + fft_length / 2;
            } else {
                // NOTE: DSP module only outputs the first half of FFT
                return fft_length / 2;
            }
        }

    };

    template<typename config_t>
    class alazar_fft_acquisition_t : public alazar_acquisition_t<config_t> {
    public:

        using base_t = alazar_acquisition_t<config_t>;
        using base_t::base_t;
        using callback_t = typename base_t::callback_t;

        void prepare() override {
            // prepare asynchronous acquisition
            if (_log) { _log->debug("configuring FFT acquisition on channels {} with {} samples/record, {} samples/A-scan, and {} records/block", alazar::to_string(_config.channel_mask()), _config.samples_per_record(), _config.samples_per_ascan(), _config.records_per_block()); }
            _board.configure_capture_fft(
                _config.channel_mask(),
                downcast<U32>(_config.samples_per_record()),
                downcast<U32>(_config.records_per_block()),
                downcast<U32>(_config.fft_length),
                FFT_OUTPUT_FORMAT_U16_LOG | (_config.include_time_domain ? FFT_OUTPUT_FORMAT_RAW_PLUS_FFT : 0)
            );
        }

    protected:

        using base_t::_config, base_t::_log, base_t::_board;

    };

}

#if defined(VORTEX_ENABLE_ENGINE)

#include <vortex/engine/adapter.hpp>

namespace vortex::engine::bind {
    template<typename block_t, typename... Args>
    auto acquisition(std::shared_ptr<vortex::acquire::alazar_fft_acquisition_t<vortex::acquire::alazar_fft_config_t<Args...>>> a) {
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
