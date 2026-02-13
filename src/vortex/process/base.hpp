/** \rst

    base class and configuration for OCT processors

    OCT processors support average subtraction, IFFT, complex filtering,
    and resampling.  The processor has input shape of #records/block x
    #samples/record x #channels/sample and output shape of #A-scans/block x
    #samples/A-scan.  Note that the processor only outputs single-channel
    output currently.  The resampling vector, if specified, determines
    the number of samples/A-scan.

 \endrst */

#pragma once

#include <complex>

#include <fmt/format.h>

#include <vortex/core.hpp>

namespace vortex::process {

    template<typename T>
    struct processor_config_t {
        using element_t = T;

        virtual ~processor_config_t() {}

        size_t channel = 0;

        std::array<size_t, 3> input_shape() const { return { ascans_per_block(), samples_per_record(), channels_per_sample() }; }

        auto& records_per_block() { return _ascans_per_block; }
        const auto& records_per_block() const { return _ascans_per_block; }

        auto& samples_per_record() { return _samples_per_record; }
        const auto& samples_per_record() const { return _samples_per_record; }

        auto& channels_per_sample() { return _channels_per_sample; }
        const auto& channels_per_sample() const { return _channels_per_sample; }

        std::array<size_t, 3> output_shape() const { return { ascans_per_block(), samples_per_ascan(), 1 }; }

        auto& ascans_per_block() { return _ascans_per_block; }
        const auto& ascans_per_block() const { return _ascans_per_block; }

        size_t samples_per_ascan() const {
            if (resampling_samples.size() > 0) {
                return resampling_samples.size();
            } else {
                return samples_per_record();
            }
        }

        size_t average_window = 0;

        xt::xtensor<std::complex<T>, 1> spectral_filter;

        xt::xtensor<T, 1> resampling_samples;

        std::optional<range_t<double>> levels;

        bool enable_ifft = true;
        bool enable_log10 = true;
        bool enable_square = true;
        bool enable_magnitude = true;

        virtual void validate() {
            if (channel >= channels_per_sample()) {
                throw std::invalid_argument(fmt::format("channel exceeds channels/sample: {} vs {}", channel, channels_per_sample()));
            }

            if (spectral_filter.size() > 0 && spectral_filter.size() != samples_per_ascan()) {
                throw std::invalid_argument(fmt::format("spectral filter and samples/A-scan mismatch: {} vs {}", spectral_filter.size(), samples_per_ascan()));
            }

            if (resampling_samples.size() > 0) {
                if (resampling_samples.size() != samples_per_ascan()) {
                    throw std::invalid_argument(fmt::format("resampling samples and samples/A-scan mismatch: {} vs {}", resampling_samples.size(), samples_per_ascan()));
                }

                auto range = xt::minmax(resampling_samples, xt::evaluation_strategy::immediate)[0];
                if (range[0] < 0 || range[1] > samples_per_record() - 1) {
                    throw std::invalid_argument(fmt::format("resampling indices out of range [0, {}]: [{}, {}]", samples_per_record() - 1, range[0], range[1]));
                }
            } else {
                if (samples_per_ascan() != samples_per_record()) {
                    throw std::invalid_argument(fmt::format("when not resampling, samples/A-scan and samples/record must match: {} vs {}", samples_per_ascan(), samples_per_record()));
                }
            }
        }

    protected:

        size_t _ascans_per_block = 1000;
        size_t _samples_per_record = 1000;
        size_t _channels_per_sample = 1;

    };

    template<typename config_t>
    class processor_t {
    public:

        const auto& config() const {
            return _config;
        }

    protected:

        config_t _config;

    };

}
