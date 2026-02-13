#include <vortex/driver/daqmx.hpp>

#include <vortex/util/cast.hpp>

using namespace vortex::daqmx;

std::string vortex::daqmx::to_string(int32 error) {
    auto n = DAQmxGetErrorString(error, nullptr, 0);
    std::string msg;
    msg.resize(n);
    DAQmxGetErrorString(error, msg.data(), downcast<uInt32>(msg.length()));
    return msg;
}

template<typename... Args> const
static void _handle_error(int32 error, const std::string& msg, Args... args) {
    if (!DAQmxFailed(error)) {
        return;
    }

    // emit an exception
#if FMT_VERSION >= 80000
    auto user_msg = fmt::format(fmt::runtime(msg), args...);
#else
    auto user_msg = fmt::format(msg, args...);
#endif
    auto error_msg = fmt::format("{}: ({})", user_msg, error);
    if (error == DAQmxErrorTimeoutExceeded) {
        throw wait_timeout(error_msg);
    } else if (error == DAQmxErrorSamplesNoLongerAvailable) {
        error_msg += " buffer overflow";
        throw buffer_overflow(error_msg);
    } else if (error == DAQmxErrorGenStoppedToPreventRegenOfOldSamples || error == DAQmxErrorSamplesNotYetAvailable) {
        error_msg += " buffer underflow";
        throw buffer_underflow(error_msg);
    } else if (error == DAQmxErrorMultiChanTypesInTask) {
        error_msg += " unsupported operation";
        throw unsupported_operation(error_msg);
    } else {
        error_msg += "\n" + to_string(error);
        throw exception(error_msg);
    }
}

daqmx_t::daqmx_t() {

}

daqmx_t::daqmx_t(std::string task_name)
    : _name(std::move(task_name)) {
    auto error = DAQmxCreateTask(_name.c_str(), &_task);
    _handle_error(error, "task creation failed");
}

daqmx_t::daqmx_t(daqmx_t&& other) {
    *this = std::move(other);
}
daqmx_t& daqmx_t::operator=(daqmx_t&& other) {
    // destroy self
    clear_task();

    // trade with other
    std::swap(_task, other._task);
    std::swap(_name, other._name);
    std::swap(_started, other._started);

    return *this;
}

daqmx_t::~daqmx_t() {
    clear_task();
}

void daqmx_t::create_digital_output(const std::string& line_name) {
    auto error = DAQmxCreateDOChan(_task, line_name.c_str(), "", DAQmx_Val_ChanForAllLines);
    _handle_error(error, "digital output channel creation failed");
}

void daqmx_t::create_digital_input(const std::string& line_name) {
    auto error = DAQmxCreateDOChan(_task, line_name.c_str(), "", DAQmx_Val_ChanForAllLines);
    _handle_error(error, "digital input channel creation failed");
}

void daqmx_t::create_analog_voltage_output(const std::string& port_name, float64 min, float64 max) {
    auto error = DAQmxCreateAOVoltageChan(_task, port_name.c_str(), "", min, max, DAQmx_Val_Volts, "");
    _handle_error(error, "analog voltage output channel creation failed");
}
void daqmx_t::create_analog_voltage_input(const std::string & port_name, float64 min, float64 max, terminal_t terminal) {
    auto error = DAQmxCreateAIVoltageChan(_task, port_name.c_str(), "", cast(terminal), min, max, DAQmx_Val_Volts, "");
    _handle_error(error, "analog voltage input channel creation failed");
}

void daqmx_t::configure_sample_clock(const std::string& source, sample_mode_t sample_mode, size_t samples_per_second, size_t samples_per_channel, size_t divisor, edge_t edge) {
    int32 error;

    if (divisor > 1) {

        // configure the timebase source, rate, and divisor
        error = DAQmxSetSampClkTimebaseSrc(_task, source.c_str());
        _handle_error(error, "timebase source from {} failed", source);
        error = DAQmxSetSampClkTimebaseRate(_task, float64(samples_per_second));
        _handle_error(error, "timebase rate set to {} samples/second failed", samples_per_second);
        error = DAQmxSetSampClkTimebaseDiv(_task, downcast<uInt32>(divisor));
        _handle_error(error, "timebase divisor set to {} failed", divisor);

        // configure clock for continuous generation at the divided rate
        // NOTE: a source of NULL means the onboard clock driven by the timebase source, which is configured above
        error = DAQmxCfgSampClkTiming(_task, NULL, float64(samples_per_second / divisor), cast(edge), cast(sample_mode), samples_per_channel / divisor);
        _handle_error(error, "clock mode {} from {} at {} samples/second and {} samples/channel failed", cast(sample_mode), source, samples_per_second / divisor, samples_per_channel / divisor);

    } else {

        // configure clock for continuous generation
        error = DAQmxCfgSampClkTiming(_task, source.c_str(), float64(samples_per_second), cast(edge), cast(sample_mode), samples_per_channel);
        _handle_error(error, "clock mode {} from {} at {} samples/second and {} samples/channel failed", cast(sample_mode), source, samples_per_second, samples_per_channel);

    }

    // configure trigger same as clock
    error = DAQmxCfgDigEdgeStartTrig(_task, source.c_str(), cast(edge));
    _handle_error(error, "trigger from {} failed", source);
}

void daqmx_t::set_output_buffer_size(size_t samples_per_channel) {
    auto error = DAQmxCfgOutputBuffer(_task, downcast<uInt32>(samples_per_channel));
    _handle_error(error, "set output buffer to {} samples/channel failed", samples_per_channel);
}
void daqmx_t::set_input_buffer_size(size_t samples_per_channel) {
    auto error = DAQmxCfgInputBuffer(_task, downcast<uInt32>(samples_per_channel));
    _handle_error(error, "set input buffer to {} samples/channel failed", samples_per_channel);
}

void daqmx_t::set_regeneration(bool enable) {
    auto error = DAQmxSetWriteRegenMode(_task, enable ? DAQmx_Val_AllowRegen : DAQmx_Val_DoNotAllowRegen);
    _handle_error(error, "set regeneration {} failed", enable ? "on" : "off");
}

void daqmx_t::write_analog(size_t samples_per_channel, const xt::xtensor<float64, 2>& buffer, const seconds& timeout) {
    int32 written;
    auto error = DAQmxWriteAnalogF64(_task, downcast<int32>(samples_per_channel), false, timeout.count(), DAQmx_Val_GroupByScanNumber, buffer.data(), &written, nullptr);
    _handle_error(error, "analog write failed");
    if (written != samples_per_channel){
        throw incomplete_operation(fmt::format("partial analog write: {} != {}", written, samples_per_channel));
    }
}
void daqmx_t::write_digital(size_t samples_per_channel, const xt::xtensor<uInt32, 2>& buffer, const seconds& timeout) {
    int32 written;
    auto error = DAQmxWriteDigitalU32(_task, downcast<int32>(samples_per_channel), false, timeout.count(), DAQmx_Val_GroupByScanNumber, buffer.data(), &written, nullptr);
    _handle_error(error, "digital write failed");
    if (written != samples_per_channel){
        throw incomplete_operation(fmt::format("partial digital write: {} != {}", written, samples_per_channel));
    }
}
void daqmx_t::read_analog(size_t samples_per_channel, xt::xtensor<float64, 2>& buffer, const seconds& timeout) {
    int32 read;
    auto error = DAQmxReadAnalogF64(_task, downcast<int32>(samples_per_channel), timeout.count(), DAQmx_Val_GroupByScanNumber, buffer.data(), downcast<uInt32>(buffer.size()), &read, nullptr);
    _handle_error(error, "analog read failed");
    if (read != samples_per_channel){
        throw incomplete_operation(fmt::format("partial analog read: {} != {}", read, samples_per_channel));
    }
}
void daqmx_t::read_digital(size_t samples_per_channel, xt::xtensor<uInt32, 2>& buffer, const seconds& timeout) {
    int32 read;
    auto error = DAQmxReadDigitalU32(_task, downcast<int32>(samples_per_channel), timeout.count(), DAQmx_Val_GroupByScanNumber, buffer.data(), downcast<uInt32>(buffer.size()), &read, nullptr);
    _handle_error(error, "digital read failed");
    if (read != samples_per_channel){
        throw incomplete_operation(fmt::format("partial digital read: {} != {}", read, samples_per_channel));
    }
}

void daqmx_t::start_task() {
    if (!_started) {
        auto error = DAQmxStartTask(_task);
        _handle_error(error, "start failed");
        _started = true;
    }
}

void daqmx_t::stop_task() {
    if (_started) {
        auto error = DAQmxStopTask(_task);
        _started = false;

        // NOTE: handle error after updating internal state
        _handle_error(error, "stop task");
    }
}

void daqmx_t::clear_task() {
    if (valid()) {
        auto error = DAQmxClearTask(_task);

        // erase self
        _task = nullptr;
        _name.clear();
        _started = false;

        // NOTE: handle error after updating internal state
        _handle_error(error, "clear task");
    }
}

const std::string& daqmx_t::name() const {
    return _name;
}
bool daqmx_t::valid() const {
    return _task != nullptr;
}
TaskHandle daqmx_t::handle() const {
    return _task;
}

bool daqmx_t::running() const {
    return _started;
}
