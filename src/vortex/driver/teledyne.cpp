#include <vortex/driver/teledyne.hpp>
#include <vortex/util/cast.hpp>
#include <vortex/util/platform.hpp>

#include <algorithm>
#include <iostream>

#include <fmt/format.h>

#define FWOCT_REG_MODULEID         0
#define FWOCT_REG_RECORDLENGTH     1
#define FWOCT_REG_DEBUG            2
#define FWOCT_REG_ERRORS           3
#define FWOCT_REG_ADDRRESET        4
#define FWOCT_REG_TABLEDATA        5
#define FWOCT_REG_TABLECONFIG      6
#define FWOCT_REG_KCLKDELAY        7
#define FWOCT_REG_CONFIG           8
#define FWOCT_REG_PHASEINC         9
#define FWOCT_REG_CLEARERRORS      10
#define FWOCT_REG_TRIGREP          11
#define FWOCT_REG_AVOIDZEROPADDING 12

#define FWOCT_ID_MAGIC          0x4b
#define FWOCT_PHASEINC_UNITY    0x10000

#define FWOCT_BACKGROUND_TABLE_BASE 2
#define FWOCT_WINDOW_TABLE_BASE     4

#define USER_LOGIC_2 2

using namespace vortex::teledyne;

std::string vortex::teledyne::to_string(const trigger_source_t& v) {
    switch (v) {
    case trigger_source_t::port_trig: return "port TRIG";
    case trigger_source_t::port_sync: return "port SYNC";
    case trigger_source_t::port_gpio: return "port GPIOA0";
    case trigger_source_t::periodic: return "periodic";
    default:
        throw std::invalid_argument(fmt::format("invalid trigger source value: {}", cast(v)));
    }
}

std::string vortex::teledyne::to_string(const clock_generator_t& v) {
    switch (v) {
    case clock_generator_t::internal_pll: return "internal";
    case clock_generator_t::external_clock: return "external";
    default:
        throw std::invalid_argument(fmt::format("invalid clock generator value: {}", cast(v)));
    }
}

std::string vortex::teledyne::to_string(const clock_reference_source_t& v) {
    switch (v) {
    case clock_reference_source_t::internal: return "internal PLL";
    case clock_reference_source_t::port_clk: return "port CLK";
    case clock_reference_source_t::PXIE_10M: return "PXIE 10M";
    default:
        throw std::invalid_argument(fmt::format("invalid clock reference value: {}", cast(v)));
    }
}

std::string vortex::teledyne::to_string(const clock_edges_t& v) {
    switch (v) {
    case clock_edges_t::rising: return "rising";
    case clock_edges_t::falling: return "falling";
    case clock_edges_t::both: return "both";
    default:
        throw std::invalid_argument(fmt::format("invalid clock edge value: {}", cast(v)));
    }
}

std::string vortex::teledyne::to_string(const fft_mode_t& v) {
    switch (v) {
    case fft_mode_t::disabled: return "disabled";
    case fft_mode_t::complex: return "complex";
    case fft_mode_t::magnitude: return "magnitude";
    case fft_mode_t::log_magnitude: return "log magnitude";
    default:
        throw std::invalid_argument(fmt::format("invalid FFT mode value: {}", cast(v)));
    }
}

std::string vortex::teledyne::channel_mask_to_string(uint32_t channel_mask)
{
    std::string s;
    for (size_t i = 0; i < std::numeric_limits<decltype(channel_mask)>::digits; i++) {
        auto x = channel_mask & (1 << i);
        if (x == 0) {
            continue;
        }
        if (!s.empty()) {
            s += "+";
        }
        s += std::to_string(i);
    }
    if (s.empty())
        s = "None";
    return s;
}

static auto _mask(size_t n) {
    return (1uLL << n) - 1uLL;
}

static auto _pop_bits(uint32_t& data, size_t n) {
    uint32_t result = data & _mask(n);
    data >>= n;
    return result;
}

static void _push_bits(uint32_t& data, size_t n, uint32_t value) {
    auto masked_value = _mask(n) & value;
    if (masked_value != value) {
        throw std::invalid_argument(fmt::format("value {} exceeds {} bits", value, n));
    }
    data = (data << n) | masked_value;
}

template<typename T, size_t IntBits, size_t FracBits, typename U>
static auto _fixed_point(U value) {
    return T(value * (1uLL << FracBits)) & _mask(IntBits + FracBits);
}

board_t::board_t(unsigned int board_index) {

    validate_ADQAPI_version();

    // Initialize the ADQ control unit object.
    _handle = CreateADQControlUnit();
    if (!valid()) {
        throw exception("failed to create ADQ API handle");
    }

    // Enable the ADQ error trace log.
    ADQControlUnit_EnableErrorTrace(handle(), LOG_LEVEL_INFO, ".");

    // List the available devices connected to the host computer.
    // Note: ADQ requires this to be called before ADQControlUnit_SetupDevice().
    ADQInfoListEntry* adq_list = nullptr;
    unsigned int n_devices = 0;
    if (!ADQControlUnit_ListDevices(handle(), &adq_list, &n_devices)) {
        throw exception("unable to list devices");
    }

    if (board_index >= n_devices) {
        throw exception(fmt::format("device index is out of range: {} >= {}", board_index, n_devices));
    }

    // Initial configuration of ADQ device;
    if (!ADQControlUnit_SetupDevice(handle(), board_index)) {
        throw exception(fmt::format("unable to set up device {}", board_index));
    }

    // Initialize our copy of the ADQ param structure.
    std::memset(&_adq, 0, sizeof(_adq));
    if (ADQ_InitializeParameters(handle(), board_index + 1, ADQ_PARAMETER_ID_TOP, &_adq) != sizeof(_adq))
        throw exception("unable to allocate device parameters");

    _info.parameters = _adq.constant;
    _info.board_index = board_index;
    // NOTE: device IDs in ADQ calls start at 1
    _info.device_number = board_index + 1;

#if ADQAPI_VERSION_MAJOR < 10
    // Tell the API whether to allocate transfer buffers in the host RAM.
    _adq.transfer.common.transfer_records_to_host_enabled = 1;
#else
    _adq.transfer.common.record_buffer_memory_owner = ADQ_MEMORY_OWNER_API;
#endif

    // If this parameter is set to 0 (default), transfer buffers will be allocated as independent memory
    // regions for all active channels. If the parameter is set to 1, the API will allocate nof_buffers
    // contiguous memory ranges, each corresponding to a transfer buffer index. Each contiguous memory range
    // will contain one transfer buffer for each active channel, placed back-to-back.
    // NOTE: this is required for the vortex memory layout
    _adq.transfer.common.packed_buffers_enabled = 1;

    // This flag controls whether ADQ may overwrite the content of buffers that
    // are currently in use by the application. Normally, this should be set to 1 to
    // prevent buffer data from being overwritten. However, it may be set to 0 to
    // completely avoid the possibility of an overflow.
    _adq.transfer.common.write_lock_enabled = 1;

    // Enable "data transfer" interface by setting marker_mode to ADQ_MARKER_MODE_HOST_MANUAL.
    _adq.transfer.common.marker_mode = ADQ_MARKER_MODE_HOST_MANUAL;

    // check for FWOCT
    auto data = _read_register(FWOCT_REG_MODULEID);
    _info.fwoct.version = _pop_bits(data, 8);
    _info.fwoct.id = _pop_bits(data, 8);
    _info.fwoct.clock_channel_count = _pop_bits(data, 1);
    _info.fwoct.oct_channel_count = _pop_bits(data, 2);
    _info.fwoct.fft_capacity = 1uLL << _pop_bits(data, 5);
    _info.fwoct.sample_time_capacity = 1uLL << _pop_bits(data, 5);
    _info.fwoct.detected = _info.fwoct.id == FWOCT_ID_MAGIC;
}

board_t::~board_t() {
    if(_started) {
        // ignore return value in destructor
        ADQ_StopDataAcquisition(handle(), info().device_number);
    }

    if(valid()) {
        DeleteADQControlUnit(_handle);
    }
}

const board_t::info_t& board_t::info() const {
    return _info;
}

void* board_t::handle() const {
    return _handle;
}

bool board_t::valid() const {
    return _handle != nullptr;
}

bool board_t::running() const {
    return _started;
}

void board_t::configure_sampling_clock(
    double sampling_frequency,
    double reference_frequency,
    ADQClockGenerator clock_generator,
    ADQReferenceClockSource reference_source,
    double delay_adjustment,
    int32_t low_jitter_mode_enabled)
{
    ADQClockSystemParameters& clock_system = _adq.constant.clock_system;
    clock_system.sampling_frequency = sampling_frequency;
    clock_system.reference_frequency = reference_frequency;
    clock_system.clock_generator = clock_generator;
    clock_system.reference_source = reference_source;
    clock_system.delay_adjustment = delay_adjustment;
    clock_system.delay_adjustment_enabled = (delay_adjustment != 0);
    clock_system.low_jitter_mode_enabled = low_jitter_mode_enabled;
}

void board_t::configure_trigger_source(
    ADQEventSource trigger_source,
    double periodic_trigger_frequency,
    int64_t horizontal_offset_samples,
    int32_t trigger_skip_factor
)
{
    _adq.event_source.periodic.frequency = periodic_trigger_frequency;
    for (int ch = 0; ch < _info.parameters.nof_channels; ++ch) {
        _adq.acquisition.channel[ch].horizontal_offset = horizontal_offset_samples;
        _adq.acquisition.channel[ch].trigger_source = trigger_source;
        _adq.acquisition.channel[ch].trigger_blocking_source = (trigger_skip_factor > 1) ? ADQ_FUNCTION_PATTERN_GENERATOR0 : ADQ_FUNCTION_INVALID;
    }

    ADQPatternGeneratorParameters& patternGenerator = _adq.function.pattern_generator[0];
    if (trigger_skip_factor > 1) {
        if (_adq.constant.nof_pattern_generators < 1) {
            throw exception(fmt::format("triggering skipping requires a pattern generator but there are none: {}", _adq.constant.nof_pattern_generators));
        }
        patternGenerator.nof_instructions = 2;
        // Instruction 0: Generator output 0 while waiting for the first trigger event.
        patternGenerator.instruction[0].op = ADQ_PATTERN_GENERATOR_OPERATION_EVENT;
        patternGenerator.instruction[0].source = trigger_source;
        patternGenerator.instruction[0].count = 1;
        patternGenerator.instruction[0].output_value = 0;
        patternGenerator.instruction[0].output_value_transition = 0;
        // Instruction 1: Generator 1 while waiting for N-1 trigger events.
        patternGenerator.instruction[1].op = ADQ_PATTERN_GENERATOR_OPERATION_EVENT;
        patternGenerator.instruction[1].source = trigger_source;
        patternGenerator.instruction[1].count = trigger_skip_factor - 1;
        patternGenerator.instruction[1].output_value = 1;
        patternGenerator.instruction[1].output_value_transition = 1;
    } else {
        // Disable pattern generator.
        patternGenerator.nof_instructions = 0;
    }
}

void board_t::configure_trigger_sync(bool enable) {
    if(enable) {

        // enable daisy chaining from the trigger input
        {
            auto& info = _adq.function.daisy_chain;
            // copy the trigger from the first channel
            info.source = _adq.acquisition.channel[_first_active_channel()].trigger_source;
            // info.edge = ADQ_EDGE_RISING;
            info.arm = ADQ_ARM_AT_ACQUISITION_START;
            info.resynchronization_enabled = true;
            info.position = 0;
        }

        // pass the daisy chain signal as a sync output
        {
            auto& info = _adq.port[ADQ_PORT_SYNC].pin[0];
            info.direction = ADQ_DIRECTION_OUT;
            info.function = ADQ_FUNCTION_DAISY_CHAIN;
            info.invert_output = 0;
            info.input_impedance = ADQ_IMPEDANCE_50_OHM;
        }

    } else {

        // disable daisy chaining
         _adq.function.daisy_chain.source = ADQ_EVENT_SOURCE_INVALID;
        // disable sync output
        _adq.port[ADQ_PORT_SYNC].pin[0].function = ADQ_FUNCTION_INVALID;

    }
}

uint32_t board_t::_read_register(uint32_t regnum) {
    uint32_t result;
    auto error = ADQ_ReadUserRegister(handle(), info().device_number, USER_LOGIC_2, regnum, &result);
    if (error != ADQ_EOK) {
        throw exception(fmt::format("failed to read register {}: {}", regnum, error));
    }
    return result;
}
void board_t::_write_register(uint32_t regnum, uint32_t data) {
    _write_register(regnum, 0, data);
}
void board_t::_write_register(uint32_t regnum, uint32_t mask, uint32_t data) {
    auto error = ADQ_WriteUserRegister(handle(), info().device_number, USER_LOGIC_2, regnum, mask, data, nullptr);
    if (error != ADQ_EOK) {
        throw exception(fmt::format("failed to write to value {} to register {} with mask {}: {}", data, regnum, mask, error));
    }
}

void board_t::configure_fwoct(uint32_t record_length, clock_edges_t clock_edges, double clock_delay_samples, double relative_phase_increment, const int16_t* background, const std::complex<float>* window, fft_mode_t fft_mode, uint32_t channel_mapping) {
    // basic options
    _write_register(FWOCT_REG_RECORDLENGTH, record_length);
    _write_register(FWOCT_REG_DEBUG, channel_mapping);

    // delay in samples as signed fixed-point integer (15.12)
    _write_register(FWOCT_REG_KCLKDELAY, _fixed_point<int32_t, 15, 12>(clock_delay_samples));

    // resampling
    auto phaseinc = uint32_t(FWOCT_PHASEINC_UNITY * relative_phase_increment);
    _write_register(FWOCT_REG_PHASEINC, phaseinc);
    
    // prepare FFT
    size_t fft_po2 = 0;
    size_t fft_magnitude = 0;
    if (fft_mode != fft_mode_t::disabled) {
        // check version
        if (info().fwoct.version < 8) {
            throw std::runtime_error(fmt::format("FFT requires FWOCT version >= 8 (current version == {})", info().fwoct.version));
        }

        // compute and validate FFT size
        fft_po2 = std::ceil(std::log2(record_length));
        fft_magnitude = cast(fft_mode);
    }

    // general configuration
    // NOTE: this disables table reading as needed to populate background and window
    uint32_t config = 0;
    _push_bits(config, 1, 0);                                          // DENDCBLOCKER: enable DC blocker
    _push_bits(config, 1, 0);                                          // ENTABLEREAD: disable table reading
    _push_bits(config, 1, 0);                                          // ENFFTPREPING: disable ping-pong
    _push_bits(config, 4, fft_po2);                                    // FFTLEN
    _push_bits(config, 1, 0);                                          // RESERVED
    _push_bits(config, 2, fft_magnitude);                              // MAGSELECT
    _push_bits(config, 1, fft_mode != fft_mode_t::disabled);           // ENFFT
    _push_bits(config, 1, background != nullptr || window != nullptr); // ENFFTPRE
    _push_bits(config, 1, 0);                                          // RESERVED
    _push_bits(config, 2, cast(clock_edges));                          // EDGETYPES
    _write_register(FWOCT_REG_CONFIG, config);

    // prepare FFT preprocessor
    if (background != nullptr || window != nullptr) {
        // NOTE: both background and window must be configured if either is enabled

        // check version
        if (info().fwoct.version < 8) {
            throw std::runtime_error(fmt::format("FFT preprocessor requires FWOCT version >= 8 (current version == {})", info().fwoct.version));
        }

        constexpr auto table_idx = 0;

        // reset table address
        _write_register(FWOCT_REG_ADDRRESET, 1);
        _write_register(FWOCT_REG_ADDRRESET, 0);

        // select background table
        _write_register(FWOCT_REG_TABLECONFIG, FWOCT_BACKGROUND_TABLE_BASE + table_idx);

        // populate table with supplied values or zeros
        for (size_t i = 0; i < info().fwoct.fft_capacity; i++) {
            int16_t data = 0;
            if (background != nullptr && i < record_length) {
                data = background[i];
            }
            // format is signed 16-bit integer
            _write_register(FWOCT_REG_TABLEDATA, data);
        }

        // reset table address
        _write_register(FWOCT_REG_ADDRRESET, 1);
        _write_register(FWOCT_REG_ADDRRESET, 0);

        // select window table
        _write_register(FWOCT_REG_TABLECONFIG, FWOCT_WINDOW_TABLE_BASE + table_idx);

        // populate table with supplied values or unity
        for (size_t i = 0; i < info().fwoct.fft_capacity; i++) {
            std::complex<float> value = { 1, 0 };
            if (window != nullptr && i < record_length) {
                value = window[i];
            }
            // format is signed fixed-point integer (2.14)
            auto im = _fixed_point<int16_t, 2, 14>(value.imag());
            auto re = _fixed_point<int16_t, 2, 14>(value.real());
            _write_register(FWOCT_REG_TABLEDATA, (im << 16) | re);
        }
    }

    // initialize other fields to sane defaults
    _write_register(FWOCT_REG_TRIGREP, 0);
    _write_register(FWOCT_REG_AVOIDZEROPADDING, 0);
}

void board_t::configure_hugepages(bool enable) {
    if(enable) {

#if !defined(VORTEX_PLATFORM_LINUX)
        throw exception("hugepages only supported on Linux");
#elif ADQAPI_VERSION_MAJOR < 10
        throw exception(fmt::format("external buffers not supported in the compiled ADQAPI {}.{}", ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR));
#else

        // shared parameters for mapping hugepages
        ADQBufferAddress hugepage;
        std::memset(&hugepage, 0, sizeof(hugepage));
        hugepage.size = info().hugepage_bytes;
        hugepage.action = ADQ_BUFFER_ACTION_HUGEPAGE_MMAP;

        // NOTE: capture configuration previously set the same record buffer size for all channels
        size_t offset = 0;
        auto record_buffer_size = _adq.transfer.channel[_first_active_channel()].record_buffer_size;

        // assign hugepage buffers for all channels and buffers
        // NOTE: iterate over buffers and then channels so that buffers are packed by channel
        for(size_t i = 0; i < info().buffer_count; i++) {

            // check if next hugepage required
            if(!hugepage.virtual_address || offset + _channels.size() * record_buffer_size > hugepage.size) {

                // check that contiguous record buffer is possible
                if(_channels.size() * record_buffer_size > hugepage.size) {
                    throw exception(fmt::format("single record buffer size for all channels exceeds hugepage size in bytes: {} > {}", record_buffer_size, _info.hugepage_bytes));
                }

                // map new hugepage
                auto result = ADQ_GetParameters(handle(), info().device_number, ADQ_PARAMETER_ID_BUFFER_ADDRESS, &hugepage);
                if(result != sizeof(hugepage)) {
                    throw exception(fmt::format("failed to allocate hugepage for buffer {} with record buffer size {}: {}", i, record_buffer_size, result));
                }

                // start assigning at beginning of hugepage
                offset = 0;
            }

            for(auto& ch : _channels) {
                auto& channel = _adq.transfer.channel[ch];

                // compute the buffer addresses
                channel.record_buffer_bus_address[i] = hugepage.bus_address + offset;
                channel.record_buffer[i] = static_cast<uint8_t*>(hugepage.virtual_address) + offset;
                offset += channel.record_buffer_size;
            }
        }

        _adq.transfer.common.record_buffer_memory_owner = ADQ_MEMORY_OWNER_USER;

#endif
    } else {
#if ADQAPI_VERSION_MAJOR >= 10
        _adq.transfer.common.record_buffer_memory_owner = ADQ_MEMORY_OWNER_API;
#endif
    }
}

void board_t::configure_capture(
    uint32_t channel_mask,
    int64_t samples_per_record,
    int32_t records_per_buffer,
    int64_t records_per_acquisition,
    bool test_pattern_signal,
    int64_t sample_skip_factor)
{
    _channels.clear();

    for (uint32_t ch = 0; ch < info().parameters.nof_channels; ++ch) {
        if (channel_mask & (1 << ch)) {
            _channels.push_back(ch);

            _adq.acquisition.channel[ch].nof_records = records_per_acquisition;
            _adq.acquisition.channel[ch].record_length = samples_per_record;

            // Reset DC offset parameter to default value (zero).
            _adq.afe.channel[ch].dc_offset = 0.0;

            // set up sample skipping
            _adq.signal_processing.sample_skip.channel[ch].skip_factor = sample_skip_factor;

            _adq.transfer.channel[ch].record_size = bytes_per_sample * samples_per_record;
#if ADQAPI_VERSION_MAJOR < 10
            _adq.transfer.channel[ch].record_length_infinite_enabled = 0;
#else
            _adq.transfer.channel[ch].infinite_record_length_enabled = 0;
#endif
            _adq.transfer.channel[ch].record_buffer_size = bytes_per_sample * records_per_buffer * samples_per_record;
            _adq.transfer.channel[ch].metadata_enabled = 0;
            _adq.transfer.channel[ch].nof_buffers = info().buffer_count;

            // Set up test signal pattern generator.
            _adq.test_pattern.channel[ch].source = test_pattern_signal ? ADQ_TEST_PATTERN_SOURCE_TRIANGLE : ADQ_TEST_PATTERN_SOURCE_DISABLE;
        }
        else {
            // Disable recording for this channel.
            _adq.acquisition.channel[ch].nof_records = 0;
        }
    }
}

void board_t::commit_configuration() {
    std::unique_lock<std::mutex> lock(_mutex);

    if (ADQ_SetParameters(handle(), info().device_number, &_adq) != sizeof(_adq)) {
        throw exception("ADQ SetParameters() failed");
    }
}

void board_t::start_capture() {

    std::unique_lock<std::mutex> lock(_mutex);

    auto result = ADQ_StartDataAcquisition(handle(), info().device_number);
    if (result != ADQ_EOK) {
        throw exception(fmt::format("start acquisition failed: {}", result));
    }

    _started = true;
}

void board_t::stop_capture() {
    std::unique_lock<std::mutex> lock(_mutex);

    auto result = ADQ_StopDataAcquisition(handle(), info().device_number);
    _started = false;

    if (result != ADQ_EOK && result != ADQ_EINTERRUPTED) {
        throw exception(fmt::format("stop acquisition failed: {}", result));
    }
}

size_t board_t::wait_and_copy_buffer(void* ptr, size_t index, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(_mutex);

    // wait for buffers to complete and lock them
    void* src;
    auto record_count = _wait_and_lock_buffer(&src, index, timeout);

    // copy all records and channels to the user buffer
    // NOTE: setting packed_buffers_enabled=1 above ensures that all channels are packed
    // NOTE: all channels share the same buffer size as per the acquisition configuration above
    auto& info = _adq.transfer.channel[_first_active_channel()];
    // TODO: consider faster copy methods
    std::memcpy(src, ptr, record_count * info.record_size * _channels.size() * bytes_per_sample);

    // return the channel buffers
    unlock_buffer(index);

    return record_count;
}

size_t board_t::wait_and_lock_buffer(void** ptr, size_t index, const std::chrono::milliseconds& timeout) {
    std::unique_lock<std::mutex> lock(_mutex);
    return _wait_and_lock_buffer(ptr, index, timeout);
}

size_t board_t::_wait_and_lock_buffer(void** ptr, size_t index, const std::chrono::milliseconds& timeout) {
    if(index >= _info.buffer_count) {
        throw std::runtime_error(fmt::format("index exceeds buffer count: {} > {}", index, _info.buffer_count));
    }

    // wait for all channels for this buffer to complete
    _wait_for_buffer(index, timeout);

    // extract the pointer for the first channel in the buffer
    // NOTE: setting packed_buffers_enabled=1 above ensures that all channels are packed
    // NOTE: all channels share the same buffer size as per the acquisition configuration above
    auto& info = _adq.transfer.channel[_first_active_channel()];
    *ptr = const_cast<void*>(info.record_buffer[index]);
    return info.record_buffer_size / info.record_size;
}

uint32_t board_t::_first_active_channel() {
    if(_channels.empty()) {
        throw std::runtime_error("no channels are active");
    } else {
        return *std::min_element(_channels.begin(), _channels.end());
    }
}

void board_t::_wait_for_buffer(size_t index, const std::chrono::milliseconds& timeout) {
    ADQP2pStatus status;

    auto deadline = std::chrono::steady_clock::now() + timeout;

    bool ready = false;
    while (!ready) {

        // wait up to remaining time for status update
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        auto result = ADQ_WaitForP2pBuffers(handle(), info().device_number, &status, downcast<uint32_t>(remaining.count()));

        // check for errors
        if (result == ADQ_EAGAIN) {
            // A timeout has occurred.

            // Check for device overflow situation.
            ADQOverflowStatus overflowStatus;
            if (ADQ_GetStatus(handle(), info().device_number, ADQ_STATUS_ID_OVERFLOW, &overflowStatus) != sizeof(overflowStatus)) {
                throw exception(fmt::format("unable to query device overflow status after a timeout for buffer {}", index));
            }
            if (overflowStatus.overflow) {
                throw buffer_overflow(fmt::format("hardware buffer overflow at buffer {}", index));
            }

            throw wait_timeout(fmt::format("acquisition timeout for buffer {}", index));
        }
        if (result != ADQ_EOK) {
            throw exception(fmt::format("unable to query buffer status: {}", result));
        }

        // check that all channels have a buffer ready
        ready = std::all_of(_channels.begin(), _channels.end(), [&](auto ch) {
            auto begin = status.channel[ch].completed_buffers;
            auto end = begin + status.channel[ch].nof_completed_buffers;
            return std::find(begin, end, index) != end;
        });
    }
}

void board_t::unlock_buffer(size_t index) const {
    if(index >= _info.buffer_count) {
        throw std::runtime_error(fmt::format("index exceeds buffer count: {} > {}", index, _info.buffer_count));
    }

    for(auto& ch : _channels) {
        auto result = ADQ_UnlockP2pBuffers(handle(), info().device_number, ch, 1llu << index);
        if (result != ADQ_EOK) {
            throw exception(fmt::format("failed to unlock buffer {} for channel {}: {}", index, ch, result));
        }
    }
}

std::vector<device_list_entry_t> vortex::teledyne::enumerate() {
    validate_ADQAPI_version();

    void* handle = CreateADQControlUnit();
    if (handle == nullptr) {
        throw exception("failed to create ADQ API handle");
    }

    // List the available devices connected to the host computer.
    ADQInfoListEntry* adq_list = nullptr;
    unsigned int n_devices = 0;
    auto result = ADQControlUnit_ListDevices(handle, &adq_list, &n_devices);

    std::vector<device_list_entry_t> devices;
    if(result) {
        devices.assign(adq_list, adq_list + n_devices);
    }

    // cleanup
    DeleteADQControlUnit(handle);

    if(!result) {
        throw exception("ADQ ListDevices() failed.");
    }

    return devices;
}

void vortex::teledyne::validate_ADQAPI_version() {
    // Validate ADQAPI version.
    auto result = ADQAPI_ValidateVersion(ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR);
    if(result == 0) {
        // ADQAPI is compatible
    } else if (result == -2) {
        // ADQAPI is backward compatible
    } else if(result == -1) {
        throw exception(fmt::format("system ADQAPI version is incompatible with compiled version {}.{}", ADQAPI_VERSION_MAJOR, ADQAPI_VERSION_MINOR));
    } else {
        throw std::invalid_argument(fmt::format("unexpected value in ADQAPI verison check: {}", result));
    }
}
