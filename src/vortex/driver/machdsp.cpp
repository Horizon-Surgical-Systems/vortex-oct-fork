#include <vortex/driver/machdsp.hpp>

#include <unordered_map>

#include <asio/write.hpp>
#include <asio/read.hpp>

#include <fmt/format.h>

#include <vortex/util/cast.hpp>

using namespace vortex::machdsp;

static std::unordered_map<size_t, const char*> _error_subsystems = {
    {MACHDSP_ERROR_OK, "OK"},
    {MACHDSP_ERROR_UART_TRANSMIT, "UART Tx"},
    {MACHDSP_ERROR_UART_RECEIVE, "UART Rx"},
    {MACHDSP_ERROR_SAI_TRANSMIT, "SAI Tx"},
    {MACHDSP_ERROR_SAI_RECEIVE, "SAI Rx"},
    {MACHDSP_ERROR_SAI_GENERAL, "SAI General"},
    {MACHDSP_ERROR_TRIGGER, "trigger"},
    {MACHDSP_ERROR_AUX, "aux"},
    {MACHDSP_ERROR_SETTINGS, "settings"},
};

static std::unordered_map<size_t, const char*> _error_underlyings = {
    {MACHDSP_ERROR_OK, "OK"},
    {MACHDSP_ERROR_UNDERFLOW, "underflow"},
    {MACHDSP_ERROR_OVERFLOW, "overflow"},
    {MACHDSP_ERROR_BUFFER_TOO_LARGE, "buffer too large"},
    {MACHDSP_ERROR_INVALID_DIVISOR, "invalid divisor"},
    {MACHDSP_ERROR_PAYLOAD_TOO_LARGE, "payload to large"},
    {MACHDSP_ERROR_INVALID_SETTING, "invalid setting"},
    {MACHDSP_ERROR_INVALID_MESSAGE, "invalid message"},
};

scoped_error_t::operator bool() const {
    return underlying != MACHDSP_ERROR_OK;
}

std::string vortex::machdsp::to_string(const scoped_error_t& error) {
    auto its = _error_subsystems.find(error.subsystem);
    auto itu = _error_underlyings.find(error.underlying);

    if (its == _error_subsystems.end() || (error && itu == _error_underlyings.end())) {
        return fmt::format("unknown ({}, {})", static_cast<size_t>(error.subsystem), static_cast<size_t>(error.underlying));
    }

    std::string msg = fmt::format("{} ({})", its->second, static_cast<size_t>(error.subsystem));
    if (error) {
        msg += fmt::format(": {} ({})", itu == _error_underlyings.end() ? "unknown" : itu->second, static_cast<size_t>(error.underlying));
    }

    return msg;
}

template<typename T>
static T _consume(uint8_t*& ptr, uint8_t* end) {
    if (ptr >= end) {
        throw std::runtime_error("read past end of buffer");
    }

    T val = *reinterpret_cast<T*>(ptr);
    ptr += sizeof(T);
    return val;
}

settings_t settings_t::decode(uint8_t* buffer, size_t buffer_size) {
    if (buffer_size < settings_t::BUFFER_BYTES) {
        throw std::runtime_error(fmt::format("settings buffer is undersized: {} < {}", buffer_size, settings_t::BUFFER_BYTES));
    }
    auto end = buffer + buffer_size;

    settings_t s;
    s.underflow_behavior = _consume<underflow_t>(buffer, end);
    s.overflow_behavior = _consume<overflow_t>(buffer, end);
    s.block_count = _consume<uint16_t>(buffer, end);
    s.block_samples = _consume<uint16_t>(buffer, end);
    s.max_buffer_samples = _consume<uint16_t>(buffer, end);
    s.enable_receive = _consume<uint8_t>(buffer, end);
    s.trigger_edge = _consume<uint8_t>(buffer, end);
    s.internal_trigger = _consume<uint16_t>(buffer, end);
    s.sample_divisor = _consume<uint16_t>(buffer, end);
    s.max_sample_divisor = _consume<uint16_t>(buffer, end);
    s.aux_divisor = _consume<uint16_t>(buffer, end);
    s.max_aux_divisor = _consume<uint16_t>(buffer, end);

    return s;
}

stream_status_t stream_status_t::decode(uint8_t* buffer, size_t buffer_size) {
    if (buffer_size < stream_status_t::BUFFER_BYTES) {
        throw std::runtime_error(fmt::format("stream status buffer is undersized: {} < {}", buffer_size, stream_status_t::BUFFER_BYTES));
    }
    auto end = buffer + buffer_size;

    stream_status_t ss;
    ss.tx_block_idx = _consume<uint16_t>(buffer, end);
    ss.rx_block_idx = _consume<uint16_t>(buffer, end);
    ss.tx_block_valid = _consume<uint16_t>(buffer, end);
    ss.rx_block_valid = _consume<uint16_t>(buffer, end);
    ss.state = _consume<stream_state_t>(buffer, end);

    return ss;
}

msg_header_t msg_header_t::decode(uint8_t* buffer, size_t buffer_size) {
    if (buffer_size < msg_header_t::BUFFER_BYTES) {
        throw std::runtime_error(fmt::format("message header buffer is undersized: {} < {}", buffer_size, msg_header_t::BUFFER_BYTES));
    }
    auto end = buffer + buffer_size;

    msg_header_t msg;
    msg.cmd = _consume<msg_type_t>(buffer, end);
    msg.payload_size = _consume<uint16_t>(buffer, end);
    for (auto& v : msg.params) {
        v = _consume<uint16_t>(buffer, end);
    }

    return msg;
}

machdsp_t::machdsp_t() {

}

machdsp_t::machdsp_t(const std::string& port_name, size_t baud_rate) {
    _ios = std::make_unique<asio::io_service>(1);

    _port = std::make_unique<asio::serial_port>(*_ios, port_name);
    _port->set_option(asio::serial_port::baud_rate(baud_rate));

    auto msg = _send_and_recv(msg_type_t::MSG_HELLO, { PROTOCOL_MAGIC });

    if (msg.params[0] != PROTOCOL_MAGIC) {
        throw exception(fmt::format("invalid magic: 0x{:04x} != 0x{:04x}", msg.params[0], PROTOCOL_MAGIC));
    }

    _info.version.major = msg.params[1];
    _info.version.minor = msg.params[2];
    _info.version.features = msg.params[3];

    auto settings = read_settings();
    _info.max_buffer_samples = settings.max_buffer_samples;
    _info.max_sample_divisor = settings.max_sample_divisor;
    _info.max_aux_divisor = settings.max_aux_divisor;
}

machdsp_t::machdsp_t(machdsp_t&& other) {
    *this = std::move(other);
}
machdsp_t& machdsp_t::operator=(machdsp_t&& other) {
    std::swap(_info, other._info);
    std::swap(_timeout, other._timeout);
    std::swap(_port, other._port);
    std::swap(_ios, other._ios);
    return *this;
}

machdsp_t::~machdsp_t() {
    if (_port) {
        if (_port->is_open()) {
            _port->cancel();
            _port->close();
        }
        _port.reset();
    }
    if (_ios) {
        _ios.reset();
    }
}

msg_header_t machdsp_t::_send_and_recv(msg_type_t cmd, const params_t& params) {
    return _send_and_recv(cmd, nullptr, 0, params);
}
msg_header_t machdsp_t::_send_and_recv(msg_type_t cmd, const void* payload, size_t payload_bytes, const params_t& params) {
    _send(cmd, payload, payload_bytes, params);
    return _recv();
}
void machdsp_t::_send(msg_type_t cmd, const params_t& params) {
    return _send(cmd, nullptr, 0, params);
}
void machdsp_t::_send(msg_type_t cmd, const void* payload, size_t payload_bytes, const params_t& params) {
    // error checking
    if ((payload == nullptr) ^ (payload_bytes == 0)) {
        throw std::runtime_error("null pointer with non-zero payload or vice-versa");
    }

    std::vector<asio::const_buffer> buffers;

    // build the header
    std::vector<uint16_t> header;
    header.push_back(cast(cmd));
    header.push_back(downcast<uint16_t>(payload_bytes));
    std::copy(params.cbegin(), params.cend(), std::back_inserter(header));
    buffers.emplace_back(header.data(), header.size() * sizeof(decltype(header)::value_type));

    // add the payload
    if (payload) {
        buffers.emplace_back(payload, payload_bytes);
    }

    // send the data
    _timed_write(buffers);
}
msg_header_t machdsp_t::_recv() {
    return _recv(nullptr, 0);
}
msg_header_t machdsp_t::_recv(void* payload, size_t max_payload_bytes) {
    // error checking
    if ((payload == nullptr) ^ (max_payload_bytes == 0)) {
        throw std::runtime_error("null pointer with non-zero payload or vice-versa");
    }

    // read the header
    std::vector<uint8_t> buffer;
    buffer.resize(msg_header_t::BUFFER_BYTES);
    _timed_read(asio::buffer(buffer.data(), buffer.size()));
    auto header = msg_header_t::decode(buffer.data(), buffer.size());

    // error checking
    if (header.payload_size > 0 && !payload) {
        throw std::runtime_error("received header with payload but destination buffer is null");
    }
    if (header.payload_size > max_payload_bytes) {
        throw std::runtime_error(fmt::format("header payload size exceeds buffer size: {} > {}", header.payload_size, max_payload_bytes));
    }

    // read the payload
    if (header.payload_size > 0) {
        _timed_read(asio::buffer(payload, header.payload_size));
    }

    // check for errors
    if (header.cmd == msg_type_t::MSG_ERROR) {
        auto error = scoped_error_t{ header.params[0], header.params[1] };
        if (error) {
            throw exception(fmt::format("protocol error: {}", to_string(error)));
        }
    }

    return header;
}

uint16_t machdsp_t::ping(uint16_t index) {
    return _send_and_recv(msg_type_t::MSG_PING, { index, 0, 0, 0 }).params[0];
}

void machdsp_t::stream_start() {
    _send_and_recv(msg_type_t::MSG_STREAM_START, {});
}
void machdsp_t::stream_stop() {
    _send_and_recv(msg_type_t::MSG_STREAM_STOP, {});
}
void machdsp_t::stream_reset() {
    _send_and_recv(msg_type_t::MSG_STREAM_RESET, {});
}
stream_status_t machdsp_t::stream_status() {
    _send(msg_type_t::MSG_STREAM_STATUS, {});

    std::vector<uint8_t> buffer;
    buffer.resize(stream_status_t::BUFFER_BYTES);
    auto header = _recv(buffer.data(), buffer.size());

    buffer.resize(header.payload_size);
    return stream_status_t::decode(buffer.data(), buffer.size());
}

void machdsp_t::block_write(const void* data_out, size_t data_out_bytes) {
    _send_and_recv(msg_type_t::MSG_BLOCK_WRITE, data_out, data_out_bytes, {});
}
size_t machdsp_t::block_read(void* data_in, size_t max_data_in_bytes) {
    auto header = _send_and_recv(msg_type_t::MSG_BLOCK_READ, data_in, max_data_in_bytes, {});
    return header.payload_size;
}
size_t machdsp_t::block_read_write(const void* data_out, size_t data_out_bytes, void* data_in, size_t max_data_in_bytes) {
    _send(msg_type_t::MSG_BLOCK_READWRITE, data_out, data_out_bytes, {});
    // first response is the read
    auto header = _recv(data_in, max_data_in_bytes);
    // second response is the write
    _recv();
    return header.payload_size;
}

void machdsp_t::write_setting(settings_entry_t setting, uint16_t value0, uint16_t value1, uint16_t value2) {
    _send_and_recv(msg_type_t::MSG_PARAMS_WRITE, { cast(setting), value0, value1, value2 });
}
settings_t machdsp_t::read_settings() {
    _send(msg_type_t::MSG_PARAMS_READ, {});

    std::vector<uint8_t> buffer;
    buffer.resize(settings_t::BUFFER_BYTES);
    auto header = _recv(buffer.data(), buffer.size());

    buffer.resize(header.payload_size);
    return settings_t::decode(buffer.data(), buffer.size());
}
void machdsp_t::reset_settings() {
    _send_and_recv(msg_type_t::MSG_PARAMS_RESET, {});
}

void machdsp_t::set_trigger_edge_rising(bool rising) {
    write_setting(settings_entry_t::SETTINGS_TRIGGER_EDGE, rising ? MACHDSP_EDGE_RISING : MACHDSP_EDGE_FALLING);
}
void machdsp_t::set_internal_trigger(size_t divisor) {
    write_setting(settings_entry_t::SETTINGS_TRIGGER_INTERNAL, divisor);
}
void machdsp_t::set_sample_divisor(size_t divisor) {
    write_setting(settings_entry_t::SETTINGS_SAMPLE_DIVISOR, divisor);
}
void machdsp_t::set_aux_divisor(size_t divisor) {
    write_setting(settings_entry_t::SETTINGS_AUX_DIVISOR, divisor);
}
void machdsp_t::set_blocks(size_t block_count, size_t block_samples) {
    write_setting(settings_entry_t::SETTINGS_BLOCKS, block_count, block_samples);
}
void machdsp_t::set_enable_receive(bool enable) {
    write_setting(settings_entry_t::SETTINGS_ENABLE_RECEIVE, enable);
}
void machdsp_t::set_overflow_behavior(overflow_t behavior) {
    write_setting(settings_entry_t::SETTINGS_OVERFLOW, cast(behavior));
}
void machdsp_t::set_underflow_behavior(underflow_t behavior) {
    write_setting(settings_entry_t::SETTINGS_UNDERFLOW, cast(behavior));
}

void machdsp_t::check_error() {
    _send_and_recv(msg_type_t::MSG_ERROR, {});
}

void machdsp_t::clear_error() {
    try {
        check_error();
    } catch (const exception&) {
        // ignore
    }
}

void machdsp_t::hard_reset() {
    _send(msg_type_t::MSG_HARD_RESET, {});
    // NOTE: no recv since the STM32 is rebooting
}

void  machdsp_t::set_timeout(const std::chrono::seconds& timeout) {
    _timeout = timeout;
}
const std::chrono::seconds& machdsp_t::timeout() const {
    return _timeout;
}

bool machdsp_t::valid() const {
    return !!_port;
}

const info_t& machdsp_t::info() const {
    return _info;
}

void machdsp_t::_check_port() const {
    if (!valid()) {
        throw exception("port is not open");
    }
}

void machdsp_t::_check_error(const char* prefix, const std::optional<asio::error_code>& error) {
    if (!error.has_value()) {
        // no callback so cancel and report timeout
        _port->cancel();
        throw machdsp::timeout(fmt::format("{}: timeout", prefix));
    } else if (*error) {
        // received an error
        throw exception(fmt::format("{}: {}", prefix, error->message()));
    } else {
        // no error
    }
}
