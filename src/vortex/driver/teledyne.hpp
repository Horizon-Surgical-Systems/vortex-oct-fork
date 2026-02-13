#pragma once

#include <string>
#include <vector>
#include <array>
#include <optional>
#include <chrono>
#include <complex>
#include <stdexcept>
#include <memory>
#include <mutex>

#include <ADQAPI.h>

#if ADQAPI_VERSION_MAJOR <= 5
#  define ADQ_REFERENCE_CLOCK_SOURCE_INTERNAL ADQ_CLOCK_REFERENCE_SOURCE_INTERNAL
#  define ADQ_REFERENCE_CLOCK_SOURCE_PORT_CLK ADQ_CLOCK_REFERENCE_SOURCE_PORT_CLK
#  define ADQ_REFERENCE_CLOCK_SOURCE_PXIE_10M ADQ_CLOCK_REFERENCE_SOURCE_PXIE_10M
#  define ADQReferenceClockSource ADQClockReferenceSource
#endif

namespace vortex::teledyne {

    enum class trigger_source_t {
        port_trig = ADQ_EVENT_SOURCE_TRIG,
        port_sync = ADQ_EVENT_SOURCE_SYNC,
        port_gpio = ADQ_EVENT_SOURCE_GPIOA0,
        periodic = ADQ_EVENT_SOURCE_PERIODIC
    };
    std::string to_string(const trigger_source_t& v);

    enum class clock_generator_t {
        internal_pll = ADQ_CLOCK_GENERATOR_INTERNAL_PLL,
        external_clock = ADQ_CLOCK_GENERATOR_EXTERNAL_CLOCK,
    };
    std::string to_string(const clock_generator_t& v);

    enum class clock_reference_source_t {
        internal = ADQ_REFERENCE_CLOCK_SOURCE_INTERNAL,
        port_clk = ADQ_REFERENCE_CLOCK_SOURCE_PORT_CLK,
        PXIE_10M = ADQ_REFERENCE_CLOCK_SOURCE_PXIE_10M
    };
    std::string to_string(const clock_reference_source_t& v);

    enum class clock_edges_t {
        rising = 0,
        falling = 1,
        both = 2,
    };
    std::string to_string(const clock_edges_t& v);

    enum class fft_mode_t {
        disabled = -1,
        complex = 0,
        magnitude = 1,
        log_magnitude = 2,
    };
    std::string to_string(const fft_mode_t& v);

    constexpr int64_t infinite_acquisition = ADQ_INFINITE_NOF_RECORDS;
    constexpr size_t bytes_per_sample = 2;

    std::string channel_mask_to_string(uint32_t channel_mask);

    class board_t {
    public:

        //
        // basic functions
        //

        board_t(unsigned int board_index = 0);
        ~board_t();

        struct info_t {
            unsigned int board_index = 0;
            int device_number = 1; // =board_index+1
            size_t buffer_count = ADQ_MAX_NOF_BUFFERS;

            // XXX: obtain this from the system
            size_t hugepage_bytes = 1073741824;
            
            // This struct defines the constant parameters of the digitizer, i.e. parameters that cannot be modified by the user.
            ADQConstantParameters parameters;

            struct fwoct_t {
                bool detected;
                size_t alignment_divisor = 8;

                size_t version;
                size_t id;
                size_t oct_channel_count;
                size_t clock_channel_count;
                size_t fft_capacity;
                size_t sample_time_capacity;
            } fwoct;
        };

        const info_t& info() const;

        void* handle() const;

        //
        // clock setup
        //

        void configure_sampling_clock(
            double sampling_frequency,
            double reference_frequency,
            ADQClockGenerator clock_generator,
            ADQReferenceClockSource reference_source,
            double delay_adjustment,
            int32_t low_jitter_mode_enabled
        );

        //
        // input setup
        //

        void configure_trigger_source(
            ADQEventSource trigger_source,
            double periodic_trigger_frequency,
            int64_t horizontal_offset_samples,
            int32_t trigger_skip_factor
        );

        //
        // auxilliary I/O
        //

        void configure_trigger_sync(bool enable);

        //
        // FWOCT options
        //

        void configure_fwoct(
            uint32_t record_length,
            clock_edges_t kclock_edges,
            double clock_delay_samples,
            double relative_phase_increment,
            const int16_t* background,
            const std::complex<float>* window,
            fft_mode_t fft_mode,
            uint32_t channel_mapping
        );

        //
        // acquisition control
        //

        void configure_capture(
            uint32_t channel_mask,
            int64_t samples_per_record,
            int32_t records_per_buffer,
            int64_t records_per_acquisition,
            bool test_pattern_signal,
            int64_t sample_skip_factor
        );
        void commit_configuration();
        void start_capture();
        void stop_capture();

        //
        // buffer management
        //
        
        void configure_hugepages(bool enable);

        size_t wait_and_copy_buffer(void* ptr, size_t index, const std::chrono::milliseconds& timeout);
        size_t wait_and_lock_buffer(void** ptr, size_t index, const std::chrono::milliseconds& timeout);
        void unlock_buffer(size_t buffer_index) const;

        bool valid() const;
        bool running() const;

    protected:

        size_t _wait_and_lock_buffer(void** ptr, size_t index, const std::chrono::milliseconds& timeout);

        void _wait_for_buffer(size_t buffer_index, const std::chrono::milliseconds& timeout);
        uint32_t _first_active_channel();

        uint32_t _read_register(uint32_t regnum);
        void _write_register(uint32_t regnum, uint32_t data);
        void _write_register(uint32_t regnum, uint32_t mask, uint32_t data);

        bool _started = false;

        /// ADQ control unit handle.
        void* _handle = nullptr;

        /// The ADQ parameter set.
        ADQParameters _adq;

        std::vector<size_t> _channels;

        /// Synchronizes concurrent buffer and stop requests.
        std::mutex _mutex;

        info_t _info;
    };

    using device_list_entry_t = ADQInfoListEntry; // describes a Teledyne device present in the system

    std::vector<device_list_entry_t> enumerate(); // list of all Teledyne devices present in the system

    void validate_ADQAPI_version();

    class exception : public std::runtime_error {
    public:
        using runtime_error::runtime_error;
    };
    class buffer_overflow : public exception {
    public:
        using exception::exception;
    };
    class wait_timeout : public exception {
    public:
        using exception::exception;
    };
}
