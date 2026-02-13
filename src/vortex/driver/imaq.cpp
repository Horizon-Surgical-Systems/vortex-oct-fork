#include <vortex/driver/imaq.hpp>

#include <shared_mutex>
#include <unordered_map>

#include <vortex/util/cast.hpp>

using namespace vortex::imaq;

static std::shared_mutex _callback_mutex;
static std::unordered_map<SESSION_ID, imaq_t::callback_t> _callback_map;

// string buffers of at least 256 bytes required per documentation
#define IMAQ_MIN_BUFFER_LENGTH 256

std::string vortex::imaq::to_string(const signal_t& v) {
    switch (v) {
    case signal_t::none: return "none";
    case signal_t::external: return "external";
    case signal_t::rtsi: return "RTSI";
    case signal_t::iso_in: return "ISO in";
    case signal_t::iso_out: return "ISO out";
    case signal_t::status: return "status";
    case signal_t::scaled_encoder: return "scaled encoder";
    case signal_t::software_trigger: return "software trigger";
    default:
        throw std::invalid_argument(fmt::format("invalid signal value: {}", cast(v)));
    }
}

std::string vortex::imaq::to_string(const polarity_t& v) {
    switch (v) {
    case polarity_t::low: return "low";
    case polarity_t::high: return "high";
    default:
        throw std::invalid_argument(fmt::format("invalid polarity value: {}", cast(v)));
    }
}

std::string vortex::imaq::to_string(const source_t& v) {
    switch (v) {
    case source_t::disabled: return "disabled";
    case source_t::acquisition_in_progress: return "acquisition in progress";
    case source_t::acquisition_done: return "acquisition done";
    case source_t::pixel_clock: return "pixel clock";
    case source_t::unasserted: return "unasserted";
    case source_t::asserted: return "asserted";
    case source_t::hsync: return "hsync";
    case source_t::vsync: return "vsync";
    case source_t::frame_start: return "frame start";
    case source_t::frame_done: return "frame done";
    case source_t::scaled_encoder: return "scaled encoder";
    default:
        throw std::invalid_argument(fmt::format("invalid source value: {}", cast(v)));
    }
}

std::string vortex::imaq::to_string(IMG_ERR error) {
    char buffer[IMAQ_MIN_BUFFER_LENGTH];
    auto error2 = imgShowError(error, buffer);

    if (error2 == IMG_ERR_GOOD) {
        return std::string(buffer);
    } else {
        return "unknown error";
    }
}

template<typename... Args> const
static void _handle_error(IMG_ERR error, const std::string& msg, Args... args) {
    if (error == IMG_ERR_GOOD) {
        return;
    }

    // emit an exception
#if FMT_VERSION >= 80000
    auto user_msg = fmt::format(fmt::runtime(msg), args...);
#else
    auto user_msg = fmt::format(msg, args...);
#endif
    auto error_msg = fmt::format("{}: ({})", user_msg, error);
    if (error == IMG_ERR_DEVICE_IN_USE) {
        error_msg += " device in use";
        throw device_in_use(error_msg);
    } else if (error == IMG_ERR_BAD_USER_RECT) {
        error_msg += " incompatible ROI (check alignment and size)";
        throw incompatible_region(error_msg);
    } else if (error == IMG_ERR_TIMEOUT) {
        error_msg += " timeout";
        throw timeout(error_msg);
    } else {
        error_msg += "\n" + to_string(error);
        throw exception(error_msg);
    }
}

imaq_t::imaq_t() {

}

imaq_t::imaq_t(std::string interface_name)
    : _name(std::move(interface_name)) {
    {
        auto error = imgInterfaceOpen(_name.c_str(), &_interface);
        _handle_error(error, "interface \"{}\" open failed", _name);
    }
    {
        auto error = imgSessionOpen(_interface, &_session);
        _handle_error(error, "session open failed");
    }

    uInt32 out;

    imgGetAttribute(_session, IMG_ATTR_INTERFACE_TYPE, &_info.device);
    imgGetAttribute(_session, IMG_ATTR_GETSERIAL, &_info.serial);

    imgGetAttribute(_session, IMG_ATTR_CALIBRATION_DATE, &out);
    if (out > 0) {
        // seconds since Jan 1 1970
        _info.calibration = std::chrono::utc_clock::time_point{} + std::chrono::seconds(out);
    }

    imgGetAttribute(_session, IMG_ATTR_HORZ_RESOLUTION, &_info.resolution.horizontal);
    imgGetAttribute(_session, IMG_ATTR_VERT_RESOLUTION, &_info.resolution.vertical);

    imgGetAttribute(_session, IMG_ATTR_LINESCAN, &out);
    _info.line_scan = out;

    imgGetAttribute(_session, IMG_ATTR_ACQWINDOW_LEFT, &_info.acquisition_window.left);
    imgGetAttribute(_session, IMG_ATTR_ACQWINDOW_TOP , &_info.acquisition_window.top);
    imgGetAttribute(_session, IMG_ATTR_ACQWINDOW_WIDTH, &_info.acquisition_window.width);
    imgGetAttribute(_session, IMG_ATTR_ACQWINDOW_HEIGHT, &_info.acquisition_window.height);
    _info.acquisition_window.pixels_per_row = _info.acquisition_window.width;

    imgGetAttribute(_session, IMG_ATTR_BITSPERPIXEL, &_info.bits_per_pixel);
    imgGetAttribute(_session, IMG_ATTR_BYTESPERPIXEL, &_info.bytes_per_pixel);
}


imaq_t::imaq_t(imaq_t&& other) {
    *this = std::move(other);
}
imaq_t& imaq_t::operator=(imaq_t&& other) {
    // destroy self
    _destroy();

    // trade with other
    std::swap(_interface, other._interface);
    std::swap(_session, other._session);
    std::swap(_name, other._name);
    std::swap(_started, other._started);
    std::swap(_info, other._info);

    return *this;
}

imaq_t::~imaq_t() {
    _destroy();
}

void imaq_t::configure_ring(std::vector<void*>& buffers, uInt32 skip) {
    auto error = imgRingSetup(_session, buffers.size(), buffers.data(), skip, false);
    _handle_error(error, "ring buffer setup of size {} failed", buffers.size());
}
imaq_t::locked_frame_t imaq_t::lock_frame(uInt32 index, bool requested_only) const {
    locked_frame_t frame(*this);
    frame.target_index = index;

    auto error = imgSessionExamineBuffer2(_session, index, &frame.actual_index, &frame.ptr);
    _handle_error(error, "locking frame {}", index);

    if (requested_only && frame.target_index != current_frame) {
        if (frame.target_index != frame.actual_index) {
            throw buffer_overflow(fmt::format("frame {} has been overwritten", frame.target_index));
        }
    }

    return frame;
}

uInt32 imaq_t::required_buffer_size() const {
    uInt32 size;
    auto error = imgSessionGetBufferSize(_session, &size);
    _handle_error(error, "get buffer size failed");
    return size;
}

imaq_t::roi_t imaq_t::fit_region(const imaq_t::roi_t& roi) {
    roi_t roi2;
    auto error = imgSessionFitROI(_session, IMG_ROI_FIT_LARGER, roi.top, roi.left, roi.height, roi.width, &roi2.top, &roi2.left, &roi2.height, &roi2.width);
    _handle_error(error, "fitting ROI failed (t={}, l={}) and shape [h={} x w={}] failed", roi.top, roi.left, roi.height, roi.width);
    roi2.pixels_per_row = 0;
    return roi2;
}

void imaq_t::configure_region(const imaq_t::roi_t& roi) {
    IMG_ERR error;

    error = imgSessionConfigureROI(_session, roi.top, roi.left, roi.height, roi.width);
    _handle_error(error, "setting ROI offset (t={}, l={}) and shape [h={} x w={}] failed", roi.top, roi.left, roi.height, roi.width);

    auto ppr = roi.pixels_per_row > 0 ? roi.pixels_per_row : roi.width;
    error = imgSetAttribute2(_session, IMG_ATTR_ROWPIXELS, ppr);
    _handle_error(error, "setting pixels per row to {} failed", ppr);
}
imaq_t::roi_t imaq_t::query_region() const {
    IMG_ERR error;

    roi_t roi;
    error = imgSessionGetROI(_session, &roi.top, &roi.left, &roi.height, &roi.width);
    _handle_error(error, "retrieving ROI failed");

    error = imgGetAttribute(_session, IMG_ATTR_ROWPIXELS, &roi.pixels_per_row);
    _handle_error(error, "retrieving pixels per row failed");
    return roi;
}

void imaq_t::configure_line_trigger(uInt32 line, uInt32 skip, polarity_t polarity, signal_t signal) {
    auto error = imgSessionLineTrigSource2(_session, cast(signal), line, cast(polarity), skip);
    _handle_error(error, "external line trigger on line {} failed", line);
}
void imaq_t::configure_frame_trigger(uInt32 line, polarity_t polarity, signal_t signal) {
    auto error = imgSessionTriggerConfigure2(_session, cast(signal), line, cast(polarity), IMG_TIMEOUT_INFINITE, IMG_TRIG_ACTION_BUFFER);
    _handle_error(error, "external frame trigger on line {} failed", line);
}
void imaq_t::configure_frame_trigger(uInt32 line, const std::chrono::milliseconds& timeout, polarity_t polarity, signal_t signal) {
    auto error = imgSessionTriggerConfigure2(_session, cast(signal), line, cast(polarity), downcast<uInt32>(timeout.count()), IMG_TRIG_ACTION_BUFFER);
    _handle_error(error, "external frame trigger on line {} failed", line);
}
void imaq_t::configure_trigger_output(uInt32 line, source_t source, polarity_t polarity, signal_t signal) {
    auto error = imgSessionTriggerDrive2(_session, cast(signal), line, cast(polarity), cast(source));
    _handle_error(error, "drive trigger on line {} from {} failed", line, cast(source));
}

void imaq_t::configure_frame_timeout(const std::chrono::milliseconds& timeout) {
    auto error = imgSetAttribute2(_session, IMG_ATTR_FRAMEWAIT_MSEC, downcast<uInt32>(timeout.count()));
    _handle_error(error, "set frame timeout of {} msec", timeout.count());
}

void imaq_t::start_capture() {
    if (!_started) {
        auto error = imgSessionStartAcquisition(_session);
        _handle_error(error, "start failed");
        _started = true;
    }
}
void imaq_t::start_capture(imaq_t::callback_t&& callback) {
    {
        // lock global state
        std::unique_lock<std::shared_mutex> lock(_callback_mutex);

        // store callback
        _callback_map[_session] = std::forward<callback_t>(callback);
    }

    auto error = imgSessionAcquire(_session, true, &imaq_t::_callback_wrapper);
    _handle_error(error, "start with callback start failed");
}

void imaq_t::stop_capture() {
    if (_started) {
        auto error = imgSessionStopAcquisition(_session);
        _started = false;

        // NOTE: handle error after updating internal state
        _handle_error(error, "stop failed");
    }
}

void imaq_t::_destroy() {
    if (_started) {
        stop_capture();
    }

    if (valid()) {
        // ignore error
        imgClose(_interface, false);
    }
}

uInt32 imaq_t::_callback_wrapper(SESSION_ID session, IMG_ERR error, uInt32 signal, void* data) {
    callback_t callback;

    {
        // lock global state
        std::shared_lock<std::shared_mutex> lock(_callback_mutex);

        // lookup the correct callback
        const auto& it = _callback_map.find(session);
        if (it != _callback_map.end()) {
            callback = it->second;
        }
    }

    if (callback) {
        // decode error
        std::exception_ptr ex;
        try {
            _handle_error(error, "error reported via callback");
        } catch (const std::exception&) {
            ex = std::current_exception();
        }

        // invoke callback
        std::invoke(callback, ex, data);
    }

    // receive future events only if there is a callback
    return !!callback;
}

const std::string& imaq_t::name() const {
    return _name;
}
bool imaq_t::valid() const {
    return _interface != 0;
}
INTERFACE_ID imaq_t::handle() const {
    return _interface;
}

bool imaq_t::running() const {
    return _started;
}

imaq_t::locked_frame_t::locked_frame_t(const imaq_t& imaq_)
    : imaq(&imaq_) { }

imaq_t::locked_frame_t::~locked_frame_t() {
    if (imaq && imaq->valid()) {
        auto error = imgSessionReleaseBuffer(imaq->_session);
        _handle_error(error, "unlocking frame {}", actual_index);
    }
}

imaq_t::locked_frame_t::locked_frame_t(imaq_t::locked_frame_t&& other) {
    *this = std::move(other);
}
imaq_t::locked_frame_t& imaq_t::locked_frame_t::operator=(imaq_t::locked_frame_t&& other) {
    std::swap(target_index, other.target_index);
    std::swap(actual_index, other.actual_index);
    std::swap(ptr, other.ptr);
    std::swap(imaq, other.imaq);

    return *this;
}

std::vector<std::string> vortex::imaq::enumerate() {
    std::vector<std::string> result;
    IMG_ERR error;

    while(true) {
        // buffer requires at least
        char buffer[IMAQ_MIN_BUFFER_LENGTH];
        error = imgInterfaceQueryNames(downcast<uInt32>(result.size()), buffer);

        if (error == IMG_ERR_GOOD) {
            result.push_back(std::string(buffer));
        } else {
            break;
        }
    }

    return result;
}
