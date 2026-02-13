#include <vortex/driver/alazar/board.hpp>

#include <unordered_map>
#include <type_traits>
#include <algorithm>

#include <fmt/format.h>

#if defined(VORTEX_ENABLE_ALAZAR_DAC)
#  include <AlazarGalvo.h>
#  define NUMBER_OF_DAC_SLOTS 5
#endif

#include <vortex/driver/alazar/core.hpp>
#include <vortex/driver/alazar/db.hpp>

#include <vortex/util/cast.hpp>

using namespace vortex::alazar;

template<typename T>
static T _clip(T v, T lower, T upper) {
    return std::min(std::max(v, lower), upper);
}

channel_t vortex::alazar::operator~(const channel_t& a) {
    return channel_t(~cast(a));
}
channel_t vortex::alazar::operator&(const channel_t& a, const channel_t& b) {
    return channel_t(cast(a) & cast(b));
}
channel_t vortex::alazar::operator|(const channel_t& a, const channel_t& b) {
    return channel_t(cast(a) | cast(b));
}
channel_t vortex::alazar::operator^(const channel_t& a, const channel_t& b) {
    return channel_t(cast(a) ^ cast(b));
}
channel_t& vortex::alazar::operator&=(channel_t& a, const channel_t& b) {
    a = a & b;
    return a;
}
channel_t& vortex::alazar::operator|=(channel_t& a, const channel_t& b) {
    a = a | b;
    return a;
}
channel_t& vortex::alazar::operator^=(channel_t& a, const channel_t& b) {
    a = a ^ b;
    return a;
}

std::string vortex::alazar::to_string(const channel_t& v) {
    switch (v) {
    case channel_t::A: return "A";
    case channel_t::B: return "B";
    case channel_t::C: return "C";
    case channel_t::D: return "D";
    case channel_t::E: return "E";
    case channel_t::F: return "F";
    case channel_t::G: return "G";
    case channel_t::H: return "H";
    case channel_t::I: return "I";
    case channel_t::J: return "J";
    case channel_t::K: return "K";
    case channel_t::L: return "L";
    case channel_t::M: return "M";
    case channel_t::N: return "N";
    case channel_t::O: return "O";
    case channel_t::P: return "P";
    default:
        std::string s;
        for (size_t i = 0; i < std::numeric_limits<std::underlying_type_t<channel_t>>::digits; i++) {
            auto x = vortex::cast(v) & (1 << i);
            if (x == 0) {
                continue;
            }
            if (s.size() > 0) {
                s += "+";
            }
            s += to_string(channel_t(x));
        }
        if (s.empty()) {
            return "None";
        } else {
            return s;
        }
    }
}

std::string vortex::alazar::to_string(const coupling_t& v) {
    switch (v) {
    case coupling_t::AC: return "AC";
    case coupling_t::DC: return "DC";
    default:
        throw std::invalid_argument(fmt::format("invalid coupling value: {}", cast(v)));
    }
}

std::string vortex::alazar::to_string(const clock_edge_t& v) {
    switch (v) {
    case clock_edge_t::rising: return "rising";
    case clock_edge_t::falling: return "falling";
    default:
        throw std::invalid_argument(fmt::format("invalid clock edge value: {}", cast(v)));
    }
}

std::string vortex::alazar::to_string(const trigger_slope_t& v) {
    switch (v) {
    case trigger_slope_t::positive: return "positive";
    case trigger_slope_t::negative: return "negative";
    default:
        throw std::invalid_argument(fmt::format("invalid trigger slope value: {}", cast(v)));
    }
}
size_t board_t::board_t::info_t::max_sampling_rate() const {
    return *std::max_element(supported_sampling_rates.begin(), supported_sampling_rates.end());
}

U32 board_t::board_t::info_t::nearest_aligned_samples_per_record(double target_samples_per_record) const {
    return downcast<U32>(size_t(std::round(target_samples_per_record / sample_alignment_divisor) * sample_alignment_divisor));
}
U32 board_t::board_t::info_t::smallest_aligned_samples_per_record(double target_samples_per_record) const {
    return downcast<U32>(size_t(std::ceil(target_samples_per_record / sample_alignment_divisor) * sample_alignment_divisor));
}

const board_t::info_t::dsp_t& board_t::info_t::find_fft_info() const {
    auto it = std::find_if(dsp.begin(), dsp.end(), [](auto& d) { return d.type == DSP_MODULE_FFT; });
    if (it == dsp.end()) {
        throw alazar::unsupported_operation("FFT module is not available");
    }
    return *it;
}

double board_t::info_t::dsp_t::max_trigger_rate_per_fft_length(U32 fft_length) const {
#if ATSAPI_VERSION >= 70500 || ATSGPU_VERSION >= 40001
    double out;
    auto rc = AlazarFFTGetMaxTriggerRepeatRate(handle, fft_length, &out);
    detail::handle_error(rc, "unable to query maximum trigger rate for FFT length {}", fft_length);

    return out;
#else
    throw std::runtime_error("maximum FFT trigger rate is not available in the compiled " ALAZAR_LIBRARY_VERSION_STRING);
#endif
}

board_t::board_t() {

}

board_t::~board_t() {
    if(_started) {
        _dsp_aware_abort_capture();
    }
}

board_t::board_t(U32 system_index, U32 board_index) {
    RETURN_CODE rc;

    _handle = AlazarGetBoardBySystemID(system_index, board_index);
    if(!valid()) {
        throw std::runtime_error(fmt::format("unable to open Alazar board {}-{}: board not found", system_index, board_index));
    }

    _info.system_index = system_index;
    _info.board_index = board_index;

    AlazarQueryCapability(_handle, GET_SERIAL_NUMBER, 0, &_info.serial_number);
    AlazarQueryCapability(_handle, GET_LATEST_CAL_DATE, 0, &_info.calibration_date);
    AlazarGetChannelInfo(_handle, &_info.onboard_memory_size, &_info.bits_per_sample);
    AlazarQueryCapability(_handle, BOARD_TYPE, 0, &_info.type.id);
    _info.type.model = detail::lookup_board_kind(_info.type.id, "UNKNOWN");
    AlazarQueryCapability(_handle, GET_PCIE_LINK_SPEED, 0, &_info.pcie.speed);
    _info.pcie.speed_gbps = _info.pcie.speed * 2.5; // value is units of 2.5 Gbps
    AlazarQueryCapability(_handle, GET_PCIE_LINK_WIDTH, 0, &_info.pcie.width);

    AlazarQueryCapability(_handle, GET_MAX_PRETRIGGER_SAMPLES, 0, &_info.max_pretrigger_samples);
    // default to most conservative alignment
    const auto& alignment = detail::lookup_alignment(_info.type.model, { 512, 128, 128 });
    _info.min_samples_per_record = alignment.min_record_size;
    _info.sample_alignment_divisor = alignment.resolution;

    _info.features = detail::lookup_features(_info.type.model, {});

    U32 flags = 0;
    rc = AlazarQueryCapability(_handle, GET_BOARD_OPTIONS_LOW, 0, &flags);
    if (rc == ApiSuccess) {
        _info.features.dual_port_memory = (flags & OPTION_DUAL_PORT_MEMORY);
        // XXX: workaround since dual edge sampling is reported as absent via the API even for boards that explicitly support it
        _info.features.dual_edge_sampling |= (flags & OPTION_DUAL_EDGE_SAMPLING);
    }

    long channel_count = 0;
    rc = AlazarGetParameter(_handle, CHANNEL_ALL, GET_CHANNELS_PER_BOARD, &channel_count);
    if (rc == ApiSuccess) {
        std::copy(all_channels.begin(), all_channels.begin() + channel_count, std::back_inserter(_info.supported_channels));
    }

    _info.supported_sampling_rates = detail::lookup_supported_sampling_rate(_info.type.model, {});
    for (auto& [impedance_ohms, input_ranges_millivolts] : detail::lookup_supported_impedance_ohms_input_range_millivolts(_info.type.model, {})) {
        for (auto& input_range_millivolts : input_ranges_millivolts) {
            _info.supported_input_combinations.push_back({ impedance_ohms, input_range_millivolts });
        }
    }

    // DSP module enumeration
    U32 n;
    rc = AlazarDSPGetModules(_handle, 0, NULL, &n);
    detail::handle_error(rc, "unable to query DSP module count");

    std::vector<dsp_module_handle> dsps(n);
    rc = AlazarDSPGetModules(_handle, dsps.size(), dsps.data(), NULL);
    detail::handle_error(rc, "unable to open DSP modules");

    _info.dsp.reserve(dsps.size());
    for (auto& handle : dsps) {
        _info.dsp.push_back({});
        auto& dsp = _info.dsp.back();

        dsp.handle = handle;

        U32 channels;
        rc = AlazarDSPGetInfo(handle, &dsp.type, &dsp.version.major, &dsp.version.minor, &dsp.max_record_length, &channels, NULL);
        if (rc == ApiSuccess) {
            dsp.supported_channels = static_cast<channel_t>(channels);
        }

        if (dsp.type == DSP_MODULE_FFT) {
#if ATSAPI_VERSION >= 70500 || ATSGPU_VERSION >= 40001
            U32 out;

            AlazarDSPGetParameterU32(handle, DSP_RAW_PLUS_FFT_SUPPORTED, &out);
            if (rc == ApiSuccess) {
                dsp.fft_time_domain_supported = out != 0;
            }

            AlazarDSPGetParameterU32(handle, DSP_FFT_SUBTRACTOR_SUPPORTED, &out);
            if (rc == ApiSuccess) {
                dsp.fft_subtractor_supported = out != 0;
            }
#else
            dsp.fft_time_domain_supported = false;
            dsp.fft_subtractor_supported = false;
#endif
        }
    }

#if defined(VORTEX_ENABLE_ALAZAR_DAC)
    // DAC module
    {
        int32_t val;

        rc = AlazarGalvoSequenceGetCount(_handle, &val);
        if (rc == ApiSuccess) {
            _info.dac.sequence_count = val;
        }

        for (size_t slot_idx = 1; slot_idx < NUMBER_OF_DAC_SLOTS + 1; slot_idx++) {
            rc = AlazarGalvoSlotGetSize(_handle, static_cast<GALVO_PATTERN_SLOT>(slot_idx), &val);
            if (rc == ApiSuccess) {
                _info.dac.slot_sizes[slot_idx] = val;
            }
        }
    }
#endif
}

const board_t::info_t& board_t::info() const {
    return _info;
}

const HANDLE& board_t::handle() const {
    return _handle;
}

void board_t::configure_clock_external(float level_ratio, coupling_t coupling, clock_edge_t clock_edge, U32 decimation) {
    RETURN_CODE rc;

    rc = AlazarSetCaptureClock(_handle,
        coupling == coupling_t::DC ? EXTERNAL_CLOCK_DC : EXTERNAL_CLOCK_AC,
        SAMPLE_RATE_USER_DEF,
        cast(clock_edge),
        decimation
    );
    detail::handle_error(rc, "failed to configure external {} clock on edge {} with decimation {}", cast(coupling), cast(clock_edge), decimation);

    if (_info.features.set_external_clock_level) {
        rc = AlazarSetExternalClockLevel(_handle, 100 * level_ratio);
        detail::handle_error(rc, "failed to set external clock level to {}", level_ratio);
    }
}

void board_t::configure_clock_internal(size_t samples_per_second, U32 decimation) {
    auto rc = AlazarSetCaptureClock(_handle, INTERNAL_CLOCK, detail::lookup_sampling_rate(samples_per_second), 0, decimation);
    detail::handle_error(rc, "failed to configure internal clock at {} samples/s and decimation {}", samples_per_second, decimation);
}

void board_t::configure_input(channel_t channel, coupling_t coupling, size_t range_millivolts, size_t impedance_ohms) {
    auto rc = AlazarInputControlEx(
        _handle, cast(channel),
        cast(coupling),
        detail::lookup_input_range_millivolts(range_millivolts),
        detail::lookup_impedance_ohms(impedance_ohms)
    );
    detail::handle_error(rc, "failed to configure channel {} with coupling {}, range {}, and impedance {}", to_string(channel), to_string(coupling), range_millivolts, impedance_ohms);
}

static auto _trigger_level(float level) {
    return static_cast<U32>(_clip<float>((level + 1) / 2 * 255, 0, 255));
}

void board_t::configure_single_trigger_external(size_t range_millivolts, float level_ratio, size_t delay_samples, trigger_slope_t slope, coupling_t coupling) {
    auto rc = AlazarSetTriggerOperation(
        _handle,
        TRIG_ENGINE_OP_J,
        TRIG_ENGINE_J, TRIG_EXTERNAL, cast(slope), _trigger_level(level_ratio),
        TRIG_ENGINE_K, TRIG_DISABLE, TRIGGER_SLOPE_POSITIVE, 0                    // disabled
    );
    detail::handle_error(rc, "failed set single external trigger on engine J with level {}% and slope {}", 100 * level_ratio, cast(slope));

    _configure_external_trigger(range_millivolts, delay_samples, coupling);
}
void board_t::configure_dual_trigger_external(size_t range_millivolts, float level_ratio_first, float level_ratio_second, size_t delay_samples, trigger_slope_t slope_first, coupling_t coupling) {
    trigger_slope_t slope_second = (slope_first == trigger_slope_t::positive) ? trigger_slope_t::negative : trigger_slope_t::positive;
    auto rc = AlazarSetTriggerOperation(
        _handle,
        TRIG_ENGINE_OP_J_OR_K,
        TRIG_ENGINE_J, TRIG_EXTERNAL, cast(slope_first), _trigger_level(level_ratio_first),
        TRIG_ENGINE_K, TRIG_EXTERNAL, cast(slope_second), _trigger_level(level_ratio_second)
    );
    detail::handle_error(rc, "failed set dual external trigger on engine (J or K) with levels {}% / {}% and slope {} / {}", 100 * level_ratio_first, 100 * level_ratio_second, cast(slope_first), cast(slope_second));

    _configure_external_trigger(range_millivolts, delay_samples, coupling);
}
void board_t::_configure_external_trigger(size_t range_millivolts, size_t delay_samples, coupling_t coupling) {
    RETURN_CODE rc;

    rc = AlazarSetExternalTrigger(_handle, cast(coupling), detail::lookup_trigger_range_volts(range_millivolts));
    detail::handle_error(rc, "failed set external trigger range {} mV (0 mV means TTL) and coupling {}", range_millivolts, cast(coupling));

    rc = AlazarSetTriggerDelay(_handle, downcast<U32>(delay_samples));
    detail::handle_error(rc, "failed set trigger delay to {} samples (check trigger delay alignment)", delay_samples);
}

void board_t::configure_auxio_trigger_out() {
    auto rc = AlazarConfigureAuxIO(_handle, AUX_OUT_TRIGGER, 0);
    detail::handle_error(rc, "failed to set auxiliary I/O to trigger output");
}
void board_t::configure_auxio_clock_out() {
    auto rc = AlazarConfigureAuxIO(_handle, AUX_OUT_CLOCK, 0);
    detail::handle_error(rc, "failed to set auxiliary I/O to clock output");
}
void board_t::configure_auxio_pacer_out(U32 divider) {
    auto rc = AlazarConfigureAuxIO(_handle, AUX_OUT_PACER, divider);
    detail::handle_error(rc, "failed to set auxiliary I/O to pacer output with divider {}", divider);
}

void board_t::configure_fft_window(U32 samples_per_record, const std::complex<float>* window) {
    if (window) {
        std::vector<float> real(samples_per_record), imaginary(samples_per_record);

        std::transform(window, window + samples_per_record, real.begin(), [](auto& v) { return std::real(v); });
        std::transform(window, window + samples_per_record, imaginary.begin(), [](auto& v) { return std::imag(v); });

        configure_fft_window(samples_per_record, real.data(), imaginary.data());
    } else {
        configure_fft_window(samples_per_record, nullptr, nullptr);
    }
}
void board_t::configure_fft_window(U32 samples_per_record, float* real, float* imaginary) {
    auto rc = AlazarFFTSetWindowFunction(_info.find_fft_info().handle, samples_per_record, real, imaginary);
    detail::handle_error(rc, "unable to set FFT window function of length {}", samples_per_record);
}

void board_t::configure_fft_background_subtraction(U32 samples_per_record, const S16* background_record) {
    RETURN_CODE rc;
    auto handle = _info.find_fft_info().handle;

    bool enable = samples_per_record > 0 && background_record;

#if ATSAPI_VERSION >= 70500 || ATSGPU_VERSION >= 40001
    if (enable) {
        rc = AlazarFFTBackgroundSubtractionSetRecordS16(handle, background_record, samples_per_record);
        detail::handle_error(rc, "unable to set FFT background record with {} samples", samples_per_record);
    }

    rc = AlazarFFTBackgroundSubtractionSetEnabled(handle, enable);
    detail::handle_error(rc, "unable to {} FFT background subtraction", enable ? "enable" : "disable");
# else
    if (enable) {
        throw std::runtime_error("FFT background subtraction is not supported in the compiled ATS-SDK version");
    }
#endif
}

void board_t::configure_sample_skipping(U32 clocks_per_record, U16* sample_bitmap) {
    bool enable = clocks_per_record > 0 && sample_bitmap;
    auto rc = AlazarConfigureSampleSkipping(_handle, enable ? SSM_ENABLE : SSM_DISABLE, clocks_per_record, sample_bitmap);
    if (enable) {
        detail::handle_error(rc, "unable to enable sample skipping with {} clocks per record", clocks_per_record);
    } else {
        detail::handle_error(rc, "unable to disable sample skipping");
    }
}

U32 board_t::_capture_flags() const {
    auto flags = U32(ADMA_NPT) | U32(ADMA_EXTERNAL_STARTCAPTURE) | ADMA_INTERLEAVE_SAMPLES;
    if (info().onboard_memory_size == 0) {
        flags |= ADMA_FIFO_ONLY_STREAMING;
    }
    return flags;
}

void board_t::configure_capture_fft(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 fft_length, U32 output_format, long transfer_offset) {
    RETURN_CODE rc;

    auto fft = _info.find_fft_info();

    // AlazarFFTSetup(...) takes the place of AlazarSetRecordSize(...)
    U32 bytes_per_record;
    rc = AlazarFFTSetup(fft.handle,
        cast(channels),
        samples_per_record, fft_length,
        output_format,
        FFT_FOOTER_NONE, 0, &bytes_per_record
    );
    detail::handle_error(rc, "failed to configure FFT for channels {} with samples per record {}, FFT length {}, and output format {}", to_string(channels), samples_per_record, fft_length, output_format);

    // use bytes_per_record instead of samples_per_record and infinite acquisition per documentation
    auto flags = _capture_flags() | ADMA_DSP;
    rc = AlazarBeforeAsyncRead(_handle,
        cast(channels),
        transfer_offset,
        bytes_per_record, records_per_buffer, infinite_acquisition,
        flags
    );
    detail::handle_error(rc, "failed to configure capture for channels {} with offset {}, bytes per record {}, records per buffer {}, and records per acquisition {} (flags = 0x{:x})", to_string(channels), transfer_offset, bytes_per_record, records_per_buffer, flags);

   _dsp_active = true;
}

void board_t::configure_capture(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 records_per_acquisition, long transfer_offset) {
    RETURN_CODE rc;

    // NOTE: must set the record size even though it is set again below
    rc = AlazarSetRecordSize(_handle, 0, samples_per_record);
    detail::handle_error(rc, "failed to set record size to zero pre-trigger samples and {} post-trigger samples", samples_per_record);

    auto flags = _capture_flags();
    rc = AlazarBeforeAsyncRead(_handle,
        cast(channels),
        transfer_offset,
        samples_per_record, records_per_buffer, records_per_acquisition,
        flags
    );
    detail::handle_error(rc, "failed to configure capture for channels {} with offset {}, samples per record {}, records per buffer {}, and records per acquisition {} (flags = 0x{:x})", to_string(channels), transfer_offset, samples_per_record, records_per_buffer, records_per_acquisition, flags);

    _dsp_active = false;
}

RETURN_CODE board_t::_dsp_aware_abort_capture() {
    if (_dsp_active) {
        return AlazarDSPAbortCapture(_handle);
    } else {
        return AlazarAbortAsyncRead(_handle);
    }
}

void board_t::start_capture() {
    auto rc = AlazarStartCapture(_handle);
    detail::handle_error(rc, "failed to start capture");
    _started = true;
}
void board_t::stop_capture() {
    auto rc = _dsp_aware_abort_capture();
    _started = false;

    // NOTE: handle error after updating internal state
    detail::handle_error(rc, "failed to stop capture");
}

RETURN_CODE board_t::_dsp_aware_wait_buffer(void* buffer, U32 timeout_ms) {
    if (_dsp_active) {
        return AlazarDSPGetBuffer(_handle, buffer, timeout_ms);
    } else {
        return AlazarWaitAsyncBufferComplete(_handle, buffer, timeout_ms);
    }
}

void board_t::post_buffer(void* ptr, size_t size_in_bytes) {
    auto rc = AlazarPostAsyncBuffer(_handle, ptr, downcast<U32>(size_in_bytes));
    detail::handle_error(rc, "failed to post buffer {} size {} bytes", ptr, size_in_bytes);
}
void board_t::wait_buffer(void* ptr) {
    RETURN_CODE rc;
    do {
        rc = _dsp_aware_wait_buffer(ptr, std::numeric_limits<U32>::max());
    } while(rc == ApiWaitTimeout);
    detail::handle_error(rc, "failed to wait for buffer {}", ptr);
}
void board_t::wait_buffer(void* ptr, const std::chrono::milliseconds& timeout) {
    auto rc = _dsp_aware_wait_buffer(ptr, downcast<U32>(timeout.count()));
    detail::handle_error(rc, "failed to wait for buffer {}", ptr);
}

void board_t::set_ignore_bad_clock(bool enable, double good_seconds, double bad_seconds) {
    double unused;
    auto rc = AlazarOCTIgnoreBadClock(_handle, static_cast<U32>(enable), good_seconds, bad_seconds, &unused, &unused);
    detail::handle_error(rc, "failed to set OCT ignore bad clock {} with good {} s and bad {} s", enable, good_seconds, bad_seconds);
}

// causes BSOD...
//void board_t::set_bits_per_sample(U8 channel, size_t bits) {
//    auto rc = AlazarSetParameterUL(_handle, channel, DATA_WIDTH, downcast<U32>(bits));
//    detail::handle_error(rc, "failed to set bits per sample (data width) {} on channel {}", bits, channel);
//}

//void board_t::set_packing_mode(size_t bits, U8 channel) {
//    auto rc = AlazarSetParameterUL(_handle, channel, PACK_MODE, detail::lookup_packing_mode(bits));
//    detail::handle_error(rc, "failed to set packing mode {} on channel {}", bits, channel);
//}

void board_t::set_dual_edge_sampling(U8 channel, bool enable) {
    auto rc = AlazarSetParameterUL(_handle, channel, SET_ADC_MODE, enable ? ADC_MODE_DES : ADC_MODE_DEFAULT);
    detail::handle_error(rc, "failed to set dual edge sampling {} on channel {}", enable, channel);
}

double board_t::buffer_bytes_per_sample() const {
    long packing;
    try {
        auto rc = AlazarGetParameter(_handle, CHANNEL_ALL, PACK_MODE, &packing);
        detail::handle_error(rc, "failed to get packing mode");
    } catch(const unsupported_operation&) {
        packing = PACK_DEFAULT;
    }

    if(packing == PACK_DEFAULT) {
        return std::ceil(_info.bits_per_sample / double(8));
    } else if(_info.bits_per_sample == 12 && packing == PACK_12_BITS_PER_SAMPLE) {
        return 1.5;
    } else {
        throw std::runtime_error(fmt::format("unknown combination of bits per sample {} and packing {}", _info.bits_per_sample, packing));
    }
}

//size_t board_t::bits_per_sample(U8 channel) const {
//    U32 bits;
//    auto rc = AlazarGetParameterUL(_handle, channel, DATA_WIDTH, &bits);
//    detail::handle_error(rc, "failed to get bits per sample (data width) on channel {}", channel);
//    return bits;
//}

#define _ACCESSOR_HELPER(name, parameter, display) \
    size_t board_t::name() const { \
        U32 value; \
        auto rc = AlazarGetParameterUL(_handle, CHANNEL_ALL, parameter, &value); \
        detail::handle_error(rc, "failed to get " display); \
        return value; \
    }

_ACCESSOR_HELPER(samples_per_record, RECORD_LENGTH, "samples per record (record length)");
_ACCESSOR_HELPER(record_capture_count, GET_RECORDS_CAPTURED, "record capture count");
_ACCESSOR_HELPER(pending_buffer_count, GET_ASYNC_BUFFERS_PENDING, "pending count");

bool board_t::valid() const {
    return _handle != NULL;
}

bool board_t::running() const {
    return _started;
}

std::vector<board_t> vortex::alazar::enumerate() {
    std::vector<board_t> boards;

    auto n_systems = AlazarNumOfSystems();
    for (U32 sid = 1; sid <= n_systems; sid++) {
        auto n_boards = AlazarBoardsInSystemBySystemID(sid);
        for (U32 bid = 1; bid <= n_boards; bid++) {
            boards.emplace_back(sid, bid);
        }
    }

    return boards;
}

#if defined(VORTEX_ENABLE_ALAZAR_GPU)

gpu_board_t::gpu_board_t() {

}
gpu_board_t::gpu_board_t(U32 system_index, U32 board_index, U32 gpu_device_index)
    : board_t(system_index, board_index) {

    static_cast<board_t::info_t&>(_info) = board_t::_info;

    auto rc = ATS_GPU_SetCUDAComputeDevice(_handle, gpu_device_index);
    detail::handle_error(rc, "failed to set GPU device {}", gpu_device_index);
    _info.gpu_device_index = gpu_device_index;
}

gpu_board_t::~gpu_board_t() {
    if (_started) {
        ATS_GPU_AbortCapture(_handle);
    }
}

const gpu_board_t::info_t& gpu_board_t::info() const {
    return _info;
}

void gpu_board_t::configure_capture(channel_t channels, U32 samples_per_record, U32 records_per_buffer, U32 records_per_acquisition, long transfer_offset, U32 gpu_flags) {
    RETURN_CODE rc;

    // NOTE: must set the record size even though it is set again below
    rc = AlazarSetRecordSize(_handle, 0, samples_per_record);
    detail::handle_error(rc, "failed to set record size to zero pre-trigger samples and {} post-trigger samples", samples_per_record);

    auto flags = U32(ADMA_NPT) | U32(ADMA_EXTERNAL_STARTCAPTURE | ADMA_INTERLEAVE_SAMPLES); // to satisfy clang
    if (info().onboard_memory_size == 0) {
        flags |= ADMA_FIFO_ONLY_STREAMING;
    }
    rc = ATS_GPU_Setup(_handle,
        cast(channels),
        transfer_offset,
        samples_per_record, records_per_buffer, records_per_acquisition,
        flags, gpu_flags
    );
    detail::handle_error(rc, "failed to configure GPU capture for channels {} with offset {}, samples per record {}, records per buffer {}, and records per acquisition {} (flags = 0x{:x}, GPU flags = 0x{:x})", to_string(channels), transfer_offset, samples_per_record, records_per_buffer, records_per_acquisition, flags, gpu_flags);
}

void gpu_board_t::start_capture() {
    auto rc = ATS_GPU_StartCapture(_handle);
    detail::handle_error(rc, "failed to start GPU capture");
    _started = true;
}
void gpu_board_t::stop_capture() {
    auto rc = ATS_GPU_AbortCapture(_handle);
    _started = false;

    // NOTE: handle error after updating internal state
    detail::handle_error(rc, "failed to stop GPU capture");
}

void gpu_board_t::post_buffer(void* ptr, size_t size_in_bytes) {
    auto rc = ATS_GPU_PostBuffer(_handle, ptr, downcast<U32>(size_in_bytes));
    detail::handle_error(rc, "failed to post GPU buffer {} size {} bytes", ptr, size_in_bytes);
}
void gpu_board_t::wait_buffer(void* ptr) {
    RETURN_CODE rc;
    do {
        rc = ATS_GPU_GetBuffer(_handle, ptr, std::numeric_limits<U32>::max(), nullptr);
    } while (rc == ApiWaitTimeout);
    detail::handle_error(rc, "failed to wait for buffer {}", ptr);
}
void gpu_board_t::wait_buffer(void* ptr, const std::chrono::milliseconds& timeout) {
    auto rc = ATS_GPU_GetBuffer(_handle, ptr, downcast<U32>(timeout.count()), nullptr);
    detail::handle_error(rc, "failed to wait for buffer {}", ptr);
}

#endif

#if defined(VORTEX_ENABLE_ALAZAR_DAC)

void board_t::start_dac() {
    auto rc = AlazarGalvoPlaybackStart(_handle);
    detail::handle_error(rc, "failed to start DAC");
}

void board_t::stop_dac() {
    auto rc = AlazarGalvoPlaybackStop(_handle);
    detail::handle_error(rc, "failed to stop DAC");
}

void board_t::configure_dac_mode(int32_t ascans_per_bscan, bool enable_bscan_mode) {
    RETURN_CODE rc;

    rc = AlazarGalvoAlinesPerBscanSet(_handle,
        ascans_per_bscan == 0 ? GALVO_ALINES_PER_BSCAN_MODE_CUSTOM : GALVO_ALINES_PER_BSCAN_MODE_STANDARD,
        ascans_per_bscan
    );
    detail::handle_error(rc, "failed to set DAC ascans per bscans to {}", ascans_per_bscan);

    rc = AlazarGalvoBscanModeSet(_handle, enable_bscan_mode ? GALVO_BSCAN_MODE_ON : GALVO_BSCAN_MODE_OFF);
    detail::handle_error(rc, "failed to set DAC bscan mode to {}", enable_bscan_mode);
}

void board_t::configure_dac_sequence(size_t sequence_idx, size_t slot_idx, int32_t repetitions, int32_t start_idx, int32_t end_idx) {
    auto rc = AlazarGalvoSequenceWrite(_handle, sequence_idx, static_cast<GALVO_PATTERN_SLOT>(slot_idx), repetitions, start_idx, start_idx, end_idx, 0, false, false, false);
    detail::handle_error(rc, "failed to configure DAC sequence {} for slot {}: {} -> {}, repeat {}x", sequence_idx, slot_idx, start_idx, end_idx, repetitions);
}

void board_t::write_dac_slot(size_t slot_idx, const uint32_t* data, int32_t count, int32_t offset) {
    auto rc = AlazarGalvoPatternSlotWrite(_handle, static_cast<GALVO_PATTERN_SLOT>(slot_idx), data, count, offset);
    detail::handle_error(rc, "failed to write DAC pattern slot {} with {} samples", slot_idx, count);
}

void board_t::set_dac_park_position(uint16_t x, uint16_t y) {
    auto rc = AlazarGalvoSetParkPosition(_handle, x, y);
    detail::handle_error(rc, "failed to set DAC park position to x={}, y={}", x, y);
}

#endif
