#pragma once

#include <string>
#include <chrono>
#include <optional>
#include <type_traits>

#include <niimaq.h>

#include <vortex/core.hpp>

auto inline format_as(IMG_SIGNAL_TYPE v)  { return static_cast<std::underlying_type_t<decltype(v)>>(v); }

namespace vortex::imaq {

    enum class signal_t {
        none = IMG_SIGNAL_NONE,
        external = IMG_SIGNAL_EXTERNAL,
        rtsi = IMG_SIGNAL_RTSI,
        iso_in = IMG_SIGNAL_ISO_IN,
        iso_out = IMG_SIGNAL_ISO_OUT,
        status = IMG_SIGNAL_STATUS,
        scaled_encoder = IMG_SIGNAL_SCALED_ENCODER,
        software_trigger = IMG_SIGNAL_SOFTWARE_TRIGGER,
    };
    std::string to_string(const signal_t& v);

    enum class polarity_t : uInt32 {
        low = IMG_TRIG_POLAR_ACTIVEL,
        high = IMG_TRIG_POLAR_ACTIVEH,
    };
    std::string to_string(const polarity_t& v);

    enum class source_t : uInt32 {
        disabled = IMG_TRIG_DRIVE_DISABLED,
        acquisition_in_progress = IMG_TRIG_DRIVE_AQ_IN_PROGRESS,
        acquisition_done = IMG_TRIG_DRIVE_AQ_DONE,
        pixel_clock = IMG_TRIG_DRIVE_PIXEL_CLK,
        unasserted = IMG_TRIG_DRIVE_UNASSERTED,
        asserted = IMG_TRIG_DRIVE_ASSERTED,
        hsync = IMG_TRIG_DRIVE_HSYNC,
        vsync = IMG_TRIG_DRIVE_VSYNC,
        frame_start = IMG_TRIG_DRIVE_FRAME_START,
        frame_done = IMG_TRIG_DRIVE_FRAME_DONE,
        scaled_encoder = IMG_TRIG_DRIVE_SCALED_ENCODER,
    };
    std::string to_string(const source_t& v);

    constexpr uInt32 current_frame = IMG_CURRENT_BUFFER;
    constexpr uInt32 infinite_timeout = IMG_TIMEOUT_INFINITE;

    class imaq_t {
    public:

        //
        // interface creation and destruction
        //

        imaq_t();
        imaq_t(std::string interface_name);

        // no copying
        imaq_t(const imaq_t&) = delete;
        imaq_t& operator=(const imaq_t&) = delete;

        // moving allowed
        imaq_t(imaq_t&& other);
        imaq_t& operator=(imaq_t && other);

        virtual ~imaq_t();

        //
        // buffer management
        //

        template<typename T>
        void configure_ring(const std::vector<T*>& buffers) {
            std::vector<void*> buffers_void;
            std::copy(buffers.begin(), buffers.end(), std::back_inserter(buffers_void));
            configure_ring(buffers_void);
        }
        void configure_ring(std::vector<void*>& buffers, uInt32 skip = 0);

        struct locked_frame_t {
            uInt32 target_index, actual_index;
            void* ptr;

            // no copying
            locked_frame_t(const locked_frame_t&) = delete;
            locked_frame_t& operator=(const locked_frame_t&) = delete;

            // moving allowed
            locked_frame_t(locked_frame_t&& other);
            locked_frame_t& operator=(locked_frame_t&& other);

            locked_frame_t(const imaq_t& imaq_);
            ~locked_frame_t();

        protected:

            const imaq_t* imaq = nullptr;

        };

        locked_frame_t lock_frame(uInt32 index = current_frame, bool requested_only = true) const;

        uInt32 required_buffer_size() const;

        //
        // region of interest
        //

        struct roi_t {
            uInt32 top, left, height, width;

            // NOTE: zero is interpreted as same as width
            uInt32 pixels_per_row = 0;
        };

        roi_t fit_region(const roi_t& roi);
        void configure_region(const roi_t& roi);
        roi_t query_region() const;

        //
        // trigger management
        //

        void configure_line_trigger(uInt32 line, uInt32 skip = 0, polarity_t polarity = polarity_t::high, signal_t signal = signal_t::external);
        void configure_frame_trigger(uInt32 line, polarity_t polarity = polarity_t::high, signal_t signal = signal_t::external);
        void configure_frame_trigger(uInt32 line, const std::chrono::milliseconds& timeout, polarity_t polarity = polarity_t::high, signal_t signal = signal_t::external);
        void configure_trigger_output(uInt32 line, source_t source, polarity_t polarity = polarity_t::high, signal_t signal = signal_t::external);

        //
        // timeouts
        //

        void configure_frame_timeout(const std::chrono::milliseconds& timeout);

        //
        // acquisition control
        //

        using callback_t = std::function<void(std::exception_ptr, void*)>;

        void start_capture();
        void start_capture(callback_t&& callback);
        void stop_capture();

        //
        // accessors
        //

        struct info_t {
            uInt32 device, serial;

            std::optional<std::chrono::utc_clock::time_point> calibration;

            struct resolution_t {
                uInt32 horizontal, vertical;
            };
            resolution_t resolution;

            roi_t acquisition_window;

            bool line_scan;

            uInt32 bits_per_pixel, bytes_per_pixel;
        };
        const info_t& info() const { return _info; }

        bool valid() const;
        const std::string& name() const;
        INTERFACE_ID handle() const;

        bool running() const;

    protected:

        INTERFACE_ID _interface = 0;
        SESSION_ID _session = 0;
        std::string _name;

        info_t _info;

        bool _started = false;

        void _destroy();

        static uInt32 _callback_wrapper(SESSION_ID session, IMG_ERR error, uInt32 signal, void* data);

    };

    std::vector<std::string> enumerate();

    class exception : public std::runtime_error {
    public:
        using runtime_error::runtime_error;
    };
    class buffer_overflow : public exception {
    public:
        using exception::exception;
    };
    class device_in_use : public exception {
    public:
        using exception::exception;
    };
    class incompatible_region : public exception {
    public:
        using exception::exception;
    };
    class timeout : public exception {
    public:
        using exception::exception;
    };

    std::string to_string(IMG_ERR error);

}

namespace vortex {

    inline IMG_SIGNAL_TYPE cast(const imaq::signal_t& o) { return static_cast<IMG_SIGNAL_TYPE>(o); };

}
