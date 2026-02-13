#include <vortex/util/sync.hpp>

using namespace vortex::sync;

event_t::event_t(bool set)
    : _set(set), _finished(false) {
}

void event_t::set() {
    std::unique_lock<std::mutex> lock(_mutex);
    _set = true;
    _cv.notify_all();
}

void event_t::unset() {
    std::unique_lock<std::mutex> lock(_mutex);
    _set = false;
}

void event_t::change(bool s) {
    if (s) {
        set();
    } else {
        unset();
    }
}

bool event_t::wait() const {
    if (!_set && !_finished) {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [&]() { return _set || _finished; });
    }
        
    return _set;
}

bool event_t::wait_for(const std::chrono::high_resolution_clock::duration& timeout) const {
    if (!_set && !_finished) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (!_cv.wait_for(lock, timeout, [&]() { return _set || _finished; })) {
            return false;
        }
    }

    return _set;
}

bool event_t::status() const {
    return _set;
}

void event_t::finish() {
    _finished = true;
    _cv.notify_all();
}


counter_t::counter_t(size_t initial)
    : _count(initial) {

}

void counter_t::set(size_t value) {
    std::unique_lock<std::mutex> lock(_mutex);
    _count = value;
    _cv.notify_all();
}

void counter_t::operator++(int) {
    increment();
}
void counter_t::operator--(int) {
    decrement();
}

void counter_t::increment() {
    std::unique_lock<std::mutex> lock(_mutex);
    _count++;
    _cv.notify_all();
}

bool counter_t::decrement() {
    std::unique_lock<std::mutex> lock(_mutex);
    while (_count == 0 && !_finished) {
        _cv.wait(lock, [&]() { return _count > 0 || _finished; });
    }
    if (_count == 0) {
        return false;
    } else {
        _count--;
        _cv.notify_all();
        return true;
    }
}

size_t counter_t::query() const {
    std::unique_lock<std::mutex> lock(_mutex);
    return _count;
}

bool counter_t::wait(size_t value) const {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_count != value && !_finished) {
        _cv.wait(lock, [&]() { return _count == value || _finished; });
    }
    return _count == value;
}

void counter_t::finish() {
    _finished = true;
    _cv.notify_all();
}