#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <complex>
#include <stdexcept>

#include <AlazarApi.h>
#include <AlazarDSP.h>
#include <AlazarCmd.h>

namespace vortex::alazar {

    enum class channel_t : U32 {
        A = CHANNEL_A,
        B = CHANNEL_B,
        C = CHANNEL_C,
        D = CHANNEL_D,
        E = CHANNEL_E,
        F = CHANNEL_F,
        G = CHANNEL_G,
        H = CHANNEL_H,
        I = CHANNEL_I,
        J = CHANNEL_J,
        K = CHANNEL_K,
        L = CHANNEL_L,
        M = CHANNEL_M,
        N = CHANNEL_N,
        O = CHANNEL_O,
        P = CHANNEL_P,
    };
    constexpr std::array<channel_t, 16> all_channels = {{ channel_t::A, channel_t::B, channel_t::C, channel_t::D, channel_t::E, channel_t::F, channel_t::G, channel_t::H, channel_t::I, channel_t::J, channel_t::K, channel_t::L, channel_t::M, channel_t::N, channel_t::P }};
    channel_t operator~(const channel_t& a);
    channel_t operator&(const channel_t& a, const channel_t& b);
    channel_t operator|(const channel_t& a, const channel_t& b);
    channel_t operator^(const channel_t& a, const channel_t& b);
    channel_t& operator&=(channel_t& a, const channel_t& b);
    channel_t& operator|=(channel_t& a, const channel_t& b);
    channel_t& operator^=(channel_t& a, const channel_t& b);
    std::string to_string(const channel_t& v);

    enum class coupling_t : U32 {
        DC = DC_COUPLING,
        AC = AC_COUPLING,
    };
    std::string to_string(const coupling_t& v);

    enum class clock_edge_t : U32 {
        rising = CLOCK_EDGE_RISING,
        falling = CLOCK_EDGE_FALLING,
    };
    std::string to_string(const clock_edge_t& v);

    enum class trigger_slope_t : U32 {
        positive = TRIGGER_SLOPE_POSITIVE,
        negative = TRIGGER_SLOPE_NEGATIVE,
    };
    std::string to_string(const trigger_slope_t& v);

    constexpr size_t trigger_range_TTL = 0;
    constexpr U32 infinite_acquisition = 0x7fffffff;

    class board_t {
    public:

        //
        // basic functions
        //

        board_t();
        board_t(U32 system_index, U32 board_index);
        ~board_t();

        struct info_t {
            U32 system_index, board_index;

            U32 serial_number, calibration_date;

            U32 onboard_memory_size;
            U8 bits_per_sample;

            struct type_t {
                U32 id;
                std::string model;
            } type;

            struct pcie_t {
                U32 speed, width;
                double speed_gbps;
            } pcie;

            std::vector<channel_t> supported_channels;
            std::vector<size_t> supported_sampling_rates;
            size_t max_sampling_rate() const;

            struct input_combination_t {
                size_t impedance_ohms, input_range_millivolts;
            };
            std::vector<input_combination_t> supported_input_combinations;

            U32 max_pretrigger_samples;
            U32 min_samples_per_record;

            U32 sample_alignment_divisor;
            U32 nearest_aligned_samples_per_record(double target_samples_per_record) const;
            U32 smallest_aligned_samples_per_record(double target_samples_per_record) const;

            struct features_t {
                // database features
                bool set_external_clock_level = false;

                std::optional<size_t> adc_calibration_sampling_rate;

                bool dual_edge_sampling = false;
                bool sample_skipping = false;

                // detected features
                bool dual_port_memory = false;
            };
            features_t features;

            struct dsp_t {
                dsp_module_handle handle;

                U32 type;
                struct version_t { U16 major, minor; };
                version_t version;
                U32 max_record_length;
                channel_t supported_channels;

                bool fft_time_domain_supported = false;
                bool fft_subtractor_supported = false;

                double max_trigger_rate_per_fft_length(U32 fft_length) const;
            };
            std::vector<dsp_t> dsp;

            const dsp_t& find_fft_info() const;

            struct dac_t {
                size_t sequence_count = 0;
                std::unordered_map<size_t, size_t> slot_sizes;

                double volts_per_lsb = 5.0 / 32767;
                uint16_t zero_point = 32768;
            };
            dac_t dac;
        };

        const info_t& info() const;

        const HANDLE& handle() const;

        //
        // clock setup
        //

        // level has range [0, 1]
        void configure_clock_external(
            float level_ratio = 0.5f,
            coupling_t coupling = coupling_t::AC,
            clock_edge_t clock_edge = clock_edge_t::rising,
            U32 decimation = 0
        );
        void configure_clock_internal(size_t samples_per_second, U32 decimation = 0);

        //
        // trigger setup
        //

        // level has range [-1, 1]
        void configure_single_trigger_external(
            size_t range_millivolts = trigger_range_TTL,
            float level_ratio = 0.0f,
            size_t delay_samples = 0,
            trigger_slope_t slope = trigger_slope_t::positive,
            coupling_t coupling = coupling_t::DC
        );
        void configure_dual_trigger_external(
            size_t range_millivolts = trigger_range_TTL,
            float level_ratio_first = 0.0f, float level_ratio_second = 0.0f,
            size_t delay_samples = 0,
            trigger_slope_t slope_first = trigger_slope_t::positive,
            coupling_t coupling = coupling_t::DC
        );

        //
        // input setup
        //

        void configure_input(
            channel_t channel,
            coupling_t coupling = coupling_t::DC,
            size_t range_millivolts = 400,
            size_t impedance_ohms = 50
        );

        //
        // auxilliary I/O
        //

        void configure_auxio_trigger_out();
        void configure_auxio_clock_out();
        void configure_auxio_pacer_out(U32 divider = 2);

        //
        // FFT options
        //

        void configure_fft_window(U32 samples_per_record, const std::complex<float>* window);
        void configure_fft_window(U32 samples_per_record, float* real = nullptr, float* imaginary = nullptr);

        void configure_fft_background_subtraction(U32 samples_per_record = 0, const S16* background_record = nullptr);

        //
        // sample skipping
        //

        void configure_sample_skipping(U32 clocks_per_record = 0, U16* sample_bitmap = nullptr);

        //
        // acquisition control
        //

        void configure_capture(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 records_per_acquisition = infinite_acquisition, long transfer_offset = 0);
        void configure_capture_fft(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 fft_length, U32 output_format, long transfer_offset = 0);
        void start_capture();
        void stop_capture();

        //
        // buffer management
        //

        void post_buffer(void* ptr, size_t size_in_bytes);
        void wait_buffer(void* ptr);
        void wait_buffer(void* ptr, const std::chrono::milliseconds& timeout);

        //
        // DAC control
        //

        void start_dac();
        void stop_dac();
        void configure_dac_mode(int32_t ascans_per_bscan, bool enable_bscan_mode);
        void configure_dac_sequence(size_t sequence_idx, size_t slot_idx, int32_t repetitions, int32_t start_idx, int32_t end_idx);
        void write_dac_slot(size_t slot_idx, const uint32_t* data, int32_t count, int32_t offset);
        void set_dac_park_position(uint16_t x, uint16_t y);

        //
        // additional options
        //

        //void set_bits_per_sample(U8 channel, size_t bits);
        //void set_packing_mode(size_t bits, U8 channel = CHANNEL_ALL);
        void set_dual_edge_sampling(U8 channel_mask, bool enable);
        void set_ignore_bad_clock(bool enable, double good_seconds = 0.0, double bad_seconds = 0.0);

        //size_t bits_per_sample(U8 channel) const;
        double buffer_bytes_per_sample() const;
        size_t samples_per_record() const;
        size_t record_capture_count() const;
        size_t pending_buffer_count() const;

        bool valid() const;
        bool running() const;

    protected:

        U32 _capture_flags() const;

        void _configure_external_trigger(size_t range_millivolts, size_t delay_samples, coupling_t coupling);

        RETURN_CODE _dsp_aware_abort_capture();
        RETURN_CODE _dsp_aware_wait_buffer(void* buffer, U32 timeout_ms);

        bool _started = false;
        bool _dsp_active = false;

        HANDLE _handle = NULL;

        info_t _info = { 0 };

    };

    std::vector<board_t> enumerate();

}

#if defined(VORTEX_ENABLE_ALAZAR_GPU)

#include <ATS_GPU.h>

namespace vortex::alazar {

    class gpu_board_t : public board_t {
    public:

        gpu_board_t();
        gpu_board_t(U32 system_index, U32 board_index, U32 gpu_device_index);
        ~gpu_board_t();

        struct info_t : board_t::info_t {
            U32 gpu_device_index;
        };

        const info_t& info() const;

        //
        // acquisition control
        //

        void configure_capture(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 records_per_acquisition = infinite_acquisition, long transfer_offset = 0, U32 gpu_flags = 0);
        void start_capture();
        void stop_capture();

        //
        // buffer management
        //

        void post_buffer(void* ptr, size_t size_in_bytes);
        void wait_buffer(void* ptr);
        void wait_buffer(void* ptr, const std::chrono::milliseconds& timeout);

    protected:

        info_t _info;

    };

}

#endif
