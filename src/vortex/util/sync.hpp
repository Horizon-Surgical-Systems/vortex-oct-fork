#pragma once

#include <atomic>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <chrono>

namespace vortex::sync {

    template<typename T>
    class queue_t {
    public:

        void push(T o, size_t limit = 0, bool drop = true) {
            std::unique_lock<std::mutex> lock(_mutex);

            if (limit > 0) {
                if (drop) {
                    while (_queue.size() >= limit) {
                        _queue.pop();
                    }
                } else if (_queue.size() >= limit) {
                    _pop_cv.wait(lock, [&]() { return _queue.size() < limit || _finished; });
                }
            }

            _queue.emplace(std::move(o));
            _push_cv.notify_one();
        }

        bool pop(T& o, bool wait = true) {
            std::unique_lock<std::mutex> lock(_mutex);
            if (wait && _queue.empty() && !_finished) {
                _push_cv.wait(lock, [&]() { return !_queue.empty() || _finished; });
            }

            if (_queue.empty()) {
                return false;
            } else {
                o = std::move(_queue.front());
                _queue.pop();
                _pop_cv.notify_one();
                return true;
            }
        }

        bool peek(const std::function<bool(T&)>& accessor, bool wait = true) {
            std::unique_lock<std::mutex> lock(_mutex);
            if (wait && _queue.empty() && !_finished) {
                _push_cv.wait(lock, [&]() { return !_queue.empty() || _finished; });
            }

            if (_queue.empty()) {
                return false;
            } else {
                if (accessor(_queue.front())) {
                    _queue.pop();
                    _pop_cv.notify_one();
                }
                return true;
            }
        }

        void clear() {
            std::unique_lock<std::mutex> lock(_mutex);
            while (!_queue.empty()) {
                _queue.pop();
            }
            _pop_cv.notify_one();
        }

        void finish() {
            std::unique_lock<std::mutex> lock(_mutex);
            _finished = true;
            _push_cv.notify_all();
            _pop_cv.notify_all();
        }

    protected:

        std::atomic_bool _finished = false;

        std::condition_variable _push_cv, _pop_cv;
        std::mutex _mutex;
        std::queue<T> _queue;

    };

    template<typename T>
    class aggregator_t {
    public:

        aggregator_t(size_t n)
            : _queues(n) {}

        void push(size_t index, T o, size_t limit = 0, bool drop = true) {
            std::unique_lock<std::mutex> lock(_mutex);
            auto& q = _queues.at(index);

            if (limit > 0) {
                if (drop) {
                    while (q.size() >= limit) {
                        q.pop();
                    }
                } else if (q.size() >= limit) {
                    _pop_cv.wait(lock, [&]() { return q.size() < limit || _finished; });
                }
            }

            q.emplace(std::move(o));
            _push_cv.notify_one();
        }

        bool pop(std::vector<T>& os, bool wait = true) {
            std::unique_lock<std::mutex> lock(_mutex);
            if (wait && _empty() && !_finished) {
                _push_cv.wait(lock, [&]() { return !_empty() || _finished; });
            }

            if (_empty()) {
                return false;
            } else {
                os.resize(_queues.size());
                for (size_t i = 0; i < _queues.size(); i++) {
                    os[i] = std::move(_queues[i].front());
                    _queues[i].pop();
                }
                _pop_cv.notify_one();
                return true;
            }

        }

        void clear() {
            std::unique_lock<std::mutex> lock(_mutex);
            for (auto& q : _queues) {
                while (!q.empty()) {
                    q.pop();
                }
            }
            _pop_cv.notify_one();
        }

        void finish() {
            std::unique_lock<std::mutex> lock(_mutex);
            _finished = true;
            _push_cv.notify_all();
            _pop_cv.notify_all();
        }

    protected:

        bool _empty() {
            return std::any_of(_queues.begin(), _queues.end(), [](auto& q) { return q.empty(); });
        }

        std::atomic_bool _finished = false;

        std::condition_variable _push_cv, _pop_cv;
        std::mutex _mutex;
        std::vector<std::queue<T>> _queues;

    };

    class counter_t {
    public:

        counter_t(size_t initial = 0);

        void set(size_t value);

        void operator++(int);
        void operator--(int);

        void increment();
        bool decrement();

        size_t query() const;

        bool wait(size_t value = 0) const;

        void finish();

    protected:

        std::atomic_bool _finished = false;
        std::atomic<size_t> _count = 0;

        mutable std::mutex _mutex;
        mutable std::condition_variable _cv;

    };

    class event_t {
    public:

        event_t(bool s = false);

        void set();
        void unset();
        void change(bool s);

        bool status() const;
        bool wait() const;
        bool wait_for(const std::chrono::high_resolution_clock::duration& timeout) const;

        void finish();

        struct scoped_set_t {
            scoped_set_t(event_t& event_)
                : event(event_) {}
            ~scoped_set_t() {
                event.set();
            }

            event_t& event;
        };

    protected:

        std::atomic_bool _set;
        std::atomic_bool _finished;

        mutable std::mutex _mutex;
        mutable std::condition_variable _cv;

    };

    template<typename T>
    struct lockable : T {

        using T::T;
        
        lockable(lockable&& o) {
            *this = std::move(o);
        }
        lockable& operator=(lockable&& o) {
            // NOTE: the caller is responsible for locking the mutex if needed
            T::operator=(std::move(o));
            return *this;
        }

        auto& mutex() const { return _mutex; }

    private:
        mutable std::shared_mutex _mutex;
    };

}