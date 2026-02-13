#pragma once

#include <string>
#include <array>
#include <vector>
#include <memory>
#include <chrono>
#include <optional>

#include <asio/read.hpp>
#include <asio/write.hpp>
#include <asio/io_service.hpp>
#include <asio/serial_port.hpp>

#include <vortex/core.hpp>
#include <vortex/util/sync.hpp>

namespace vortex::machdsp {

    class exception : public std::runtime_error {
    public:
        using runtime_error::runtime_error;
    };
    class timeout : public exception {
    public:
        using exception::exception;
    };

    static constexpr uint16_t PROTOCOL_MAGIC = 0x1234;

    enum class msg_type_t : uint16_t {
        MSG_HELLO                   = 0,

        MSG_PARAMS_READ             = 1,   // params unused
        MSG_PARAMS_WRITE            = 2,   // params unused
        MSG_PARAMS_RESET            = 3,   // params unused

        MSG_STREAM_START            = 10,  // params unused
        MSG_STREAM_STOP             = 12,  // params unused
        MSG_STREAM_RESET            = 13,  // params unused
        MSG_STREAM_STATUS           = 14,  // params unused

        MSG_BLOCK_WRITE             = 20,  // params unused
        MSG_BLOCK_READ              = 21,  // params unused
        MSG_BLOCK_READWRITE         = 22,  // params unused

        MSG_PING                    = 30,  // params unused

        MSG_ERROR                   = 40,  // params unused
        MSG_HARD_RESET              = 41,  // params unused
    };

    enum class settings_entry_t : uint16_t {
        SETTINGS_UNDERFLOW          = 1,
        SETTINGS_OVERFLOW           = 2,
        SETTINGS_BLOCKS             = 3,
        SETTINGS_ENABLE_RECEIVE     = 4,
        SETTINGS_TRIGGER_EDGE       = 5,
        SETTINGS_TRIGGER_INTERNAL   = 6,
        SETTINGS_SAMPLE_DIVISOR     = 7,
        SETTINGS_AUX_DIVISOR        = 8,
    };

    enum class stream_state_t : uint8_t {
        STREAM_CONTINUE             = 0,
        STREAM_EXIT                 = 1,
        STREAM_ABORT                = 2,
    };

    enum class underflow_t : uint8_t {
        UNDERFLOW_ERROR             = 1,  // report an error and stop streaming
        UNDERFLOW_LOOP              = 2,  // repeat the buffer
        UNDERFLOW_HOLD              = 3,  // repeat the last block
        UNDERFLOW_IGNORE            = 4,  // keep writing
    };

    enum class overflow_t : uint8_t {
        OVERFLOW_ERROR              = 1,  // report an error and stop streaming
        OVERFLOW_IGNORE             = 2,  // keep reading
    };

#define MACHDSP_ERROR_OK                    0
#define MACHDSP_ERROR_UART_TRANSMIT         1
#define MACHDSP_ERROR_UART_RECEIVE          2
#define MACHDSP_ERROR_SAI_TRANSMIT          3
#define MACHDSP_ERROR_SAI_RECEIVE           4
#define MACHDSP_ERROR_SAI_GENERAL           5
#define MACHDSP_ERROR_TRIGGER               6
#define MACHDSP_ERROR_AUX                   7
#define MACHDSP_ERROR_SETTINGS              8

#define MACHDSP_ERROR_UNDERFLOW             1
#define MACHDSP_ERROR_OVERFLOW              2
#define MACHDSP_ERROR_BUFFER_TOO_LARGE      3
#define MACHDSP_ERROR_INVALID_DIVISOR       4
#define MACHDSP_ERROR_PAYLOAD_TOO_LARGE     5
#define MACHDSP_ERROR_INVALID_SETTING       6
#define MACHDSP_ERROR_INVALID_MESSAGE       7

#define MACHDSP_EDGE_RISING                	0
#define MACHDSP_EDGE_FALLING                1

    struct scoped_error_t {
        uint16_t subsystem = 0;
        uint16_t underlying = 0;

        operator bool() const;
    };
    std::string to_string(const scoped_error_t& error);

    struct settings_t {
        underflow_t underflow_behavior;
        overflow_t overflow_behavior;
        uint16_t block_count;
        uint16_t block_samples;
        uint16_t max_buffer_samples;
        uint8_t enable_receive;
        uint8_t trigger_edge;
        uint16_t internal_trigger;
        uint16_t sample_divisor;
        uint16_t max_sample_divisor;
        uint16_t aux_divisor;
        uint16_t max_aux_divisor;

        static constexpr size_t BUFFER_BYTES = 8 * sizeof(uint16_t) + 4 * sizeof(uint8_t);
        static settings_t decode(uint8_t* buffer, size_t buffer_size);
    };

    struct stream_status_t {
        uint16_t tx_block_idx;
        uint16_t rx_block_idx;
        uint16_t tx_block_valid;
        uint16_t rx_block_valid;
        stream_state_t state;

        static constexpr size_t BUFFER_BYTES = 4 * sizeof(uint16_t) + 1 * sizeof(uint8_t) + 1; // extra byte for padding
        static stream_status_t decode(uint8_t* buffer, size_t buffer_size);
    };

    struct info_t {
        struct version_t {
            uint16_t major, minor, features;
        };
        version_t version;

        uint16_t zero_point = 32768;
        double per_lsb = 1.0 / 32767;

        size_t max_buffer_samples, max_sample_divisor, max_aux_divisor;
    };

    static constexpr auto MAX_PARAMS = 4;
    using params_t = std::array<uint16_t, MAX_PARAMS>;

    struct msg_header_t {
        msg_type_t cmd;
        uint16_t payload_size;
        params_t params;

        static constexpr auto BUFFER_BYTES = 6 * sizeof(uint16_t);
        static msg_header_t decode(uint8_t* buffer, size_t buffer_size);
    };

    class machdsp_t {
    public:

        //
        // creation and destruction
        //

        machdsp_t();
        machdsp_t(const std::string& port_name, size_t baud_rate);

        // no copying
        machdsp_t(const machdsp_t& other) = delete;
        machdsp_t& operator=(const machdsp_t& other) = delete;

        // moving allowed
        machdsp_t(machdsp_t&& other);
        machdsp_t& operator=(machdsp_t&& other);

        virtual ~machdsp_t();

        //
        // low-level functions
        //

    protected:

        msg_header_t _send_and_recv(msg_type_t cmd, const params_t& params);
        msg_header_t _send_and_recv(msg_type_t cmd, const void* payload, size_t payload_bytes, const params_t& params);
        void _send(msg_type_t cmd, const params_t& params);
        void _send(msg_type_t cmd, const void* payload, size_t payload_bytes, const params_t& params);
        msg_header_t _recv();
        msg_header_t _recv(void* payload, size_t max_payload_bytes);

    public:

        //
        // high-level functions
        //

        uint16_t ping(uint16_t index = 0);
        void stream_start();
        void stream_stop();
        void stream_reset();
        stream_status_t stream_status();

        void block_write(const void* data_out, size_t data_out_bytes);
        size_t block_read(void* data_in, size_t max_data_in_bytes);
        size_t block_read_write(const void* data_out, size_t data_out_bytes, void* data_in, size_t max_data_in_bytes);

        void write_setting(settings_entry_t setting, uint16_t value0, uint16_t value1 = 0, uint16_t value2 = 0);
        settings_t read_settings();
        void reset_settings();

        void set_trigger_edge_rising(bool rising);
        void set_internal_trigger(size_t divisor);
        void set_sample_divisor(size_t divisor);
        void set_aux_divisor(size_t divisor);
        void set_blocks(size_t block_count, size_t block_samples);
        void set_enable_receive(bool enable);
        void set_overflow_behavior(overflow_t behavior);
        void set_underflow_behavior(underflow_t behavior);

        void check_error();
        void clear_error();

        void hard_reset();

        //
        // accessors
        //

        void set_timeout(const std::chrono::seconds& timeout);
        const std::chrono::seconds& timeout() const;

        bool valid() const;
        const info_t& info() const;

    protected:

        //
        // internals
        //

        template<typename buffers_t>
        void _timed_read(buffers_t&& buffers) {
            _check_port();

            // perform the read
            std::optional<asio::error_code> error;
            asio::async_read(*_port, std::forward<buffers_t>(buffers), [&](const asio::error_code& error_, size_t n) {
                error = error_;
            });

            // wait up to timeout for completion
            _ios->restart();
            _ios->run_for(_timeout);
            _check_error("serial read", error);
        }
        template<typename buffers_t>
        void _timed_write(buffers_t&& buffers) {
            _check_port();

            // perform the write
            std::optional<asio::error_code> error;
            asio::async_write(*_port, std::forward<buffers_t>(buffers), [&](const asio::error_code& error_, size_t n) {
                error = error_;
            });

            // wait up to timeout for completion
            _ios->restart();
            _ios->run_for(_timeout);
            _check_error("serial write", error);
        }

        void _check_port() const;
        void _check_error(const char* prefix, const std::optional<asio::error_code>& error);

        info_t _info;

        std::chrono::seconds _timeout = std::chrono::seconds(1);

        std::unique_ptr<asio::io_service> _ios;
        std::unique_ptr<asio::serial_port> _port;

    };


}
