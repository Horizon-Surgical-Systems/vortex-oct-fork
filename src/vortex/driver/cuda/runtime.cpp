#include <vortex/driver/cuda/runtime.hpp>
#include <vortex/driver/cuda/types.hpp>

using namespace vortex::cuda;

device_t vortex::cuda::device(device_t index) {
    auto orig = device();
    auto error = cudaSetDevice(index);
    detail::handle_error(error, "unable to set CUDA device");
    return orig;
}

device_t vortex::cuda::device() {
    device_t index;
    auto error = cudaGetDevice(&index);
    detail::handle_error(error, "unable to query CUDA device");
    return index;
}

void vortex::cuda::peer_access(cuda::device_t device_visible_to, cuda::device_t device_visible_from, bool enable) {
    auto prior_device = device();
    device(device_visible_to);
    if (enable) {
        auto error = cudaDeviceEnablePeerAccess(device_visible_from, 0);
        detail::handle_error(error, "unable to enable peer access of device {} on device {}", device_visible_from, device_visible_to);
    } else {
        auto error = cudaDeviceDisablePeerAccess(device_visible_from);
        detail::handle_error(error, "unable to disable peer access of device {} on device {}", device_visible_from, device_visible_to);
    }
    cuda::device(prior_device);
}

bool vortex::cuda::peer_access(cuda::device_t device_visible_to, cuda::device_t device_visible_from) {
    int accessible;
    auto error = cudaDeviceCanAccessPeer(&accessible, device_visible_to, device_visible_from);
    detail::handle_error(error, "could not determine peer accessibility {} <-> {}", device_visible_to, device_visible_from);
    return accessible != 0;
}

event_t::event_t(unsigned int flags) {
    auto error = cudaEventCreateWithFlags(&_event, flags);
    detail::handle_error(error, "unable to create event");
}

event_t::~event_t() {
    // ignore error
    cudaEventDestroy(_event);
}

void event_t::sync() const {
    auto error = cudaEventSynchronize(_event);
    detail::handle_error(error, "unable to synchronize event {}", (void*)_event);
}

bool event_t::done() const {
    auto error = cudaEventQuery(_event);
    if(error == cudaErrorNotReady) {
        return false;
    } else {
        detail::handle_error(error, "unable to query event {}", (void*)_event);
    }

    return true;
}

void event_t::record() {
    auto error = cudaEventRecord(_event);
    detail::handle_error(error, "unable to record event {}", (void*)_event);
}

void event_t::record(const stream_t& stream) {
    auto error = cudaEventRecord(_event, stream.handle());
    detail::handle_error(error, "unable to record event {} on stream {}", (void*)_event, (void*)stream.handle());
}

float event_t::elapsed(const event_t& start) const {
    float result;
    auto error = cudaEventElapsedTime(&result, start._event, _event);
    detail::handle_error(error, "unable to compute elapsed time from event {} to event {}", (void*)start._event, (void*)_event);

    return result * 1000;
}

const cudaEvent_t& event_t::handle() const {
    return _event;
}

texture_t::texture_t() {
    _texture = 0;
}

texture_t::~texture_t() {
    if(valid()) {
        // ignore error
        cudaDestroyTextureObject(_texture);
    }
}

texture_t::texture_t(texture_t&& o) : texture_t() {
    *this = std::move(o);
}
texture_t& texture_t::operator=(texture_t&& o) {
    std::swap(_texture, o._texture);

    return *this;
}

bool texture_t::valid() const {
    return _texture != 0;
}

void texture_t::reset() {
    *this = texture_t();
}

const cudaTextureObject_t& texture_t::handle() const {
    return _texture;
}

surface_t::surface_t() {
    _surface = 0;
}

surface_t::~surface_t() {
    if(valid()) {
        // ignore error
        cudaDestroySurfaceObject(_surface);
    }
}

surface_t::surface_t(surface_t&& o) : surface_t() {
    *this = std::move(o);
}
surface_t& surface_t::operator=(surface_t&& o) {
    std::swap(_surface, o._surface);

    return *this;
}

bool surface_t::valid() const {
    return _surface != 0;
}

const cudaSurfaceObject_t& surface_t::handle() const {
    return _surface;
}

stream_t::stream_t(unsigned int flags) {
    auto error = cudaStreamCreateWithFlags(&_stream, flags);
    detail::handle_error(error, "unable to create stream {}: {}", (void*)_stream);
}

stream_t::~stream_t() {
    if(_stream != 0) {
        // ignore error
        cudaStreamDestroy(_stream);
    }
}

stream_t::stream_t(stream_t&& o) : stream_t() {
    *this = std::move(o);
}
stream_t& stream_t::operator=(stream_t&& o) {
    std::swap(_stream, o._stream);

    return *this;
}

const cudaStream_t& stream_t::handle() const {
    return _stream;
}

void stream_t::sync() const {
    auto error = cudaStreamSynchronize(_stream);
    detail::handle_error(error, "unable to synchronize stream {}", (void*)_stream);
}

void stream_t::wait(const event_t& event) const {
    auto error = cudaStreamWaitEvent(_stream, event.handle(), 0);
    detail::handle_error(error, "unable to wait for event {} on stream {}", (void*)event.handle(), (void*)_stream);
}

bool stream_t::ready() const {
    auto error = cudaStreamQuery(_stream);
    if(error == cudaErrorNotReady) {
        return false;
    } else if(error != cudaSuccess) {
        detail::handle_error(error, "unable to query stream {}", (void*)_stream);
    }

    return true;
}

stream_t stream_t::default_() {
    return stream_t(cudaStreamDefault);
}
