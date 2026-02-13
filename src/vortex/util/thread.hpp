#pragma once

#include <thread>
#include <functional>

#include <spdlog/logger.h>

#include <vortex/util/sync.hpp>
#include <vortex/util/platform.hpp>

namespace vortex::util {

    class worker_pool_t {
    public:

        using setup_task_t = std::function<void(size_t)>;
        using task_t = std::function<void()>;

        worker_pool_t() {}
        worker_pool_t(const std::string& name, size_t n = 0, setup_task_t&& setup_task = {}, std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)), _setup_task(std::forward<setup_task_t>(setup_task)) {

            if (n == 0) {
                n = std::thread::hardware_concurrency();
            }

            _threads.resize(n);
            for (size_t i = 0; i < n; i++) {
                _threads[i] = std::thread(&worker_pool_t::_loop, this, i, n == 1 ? name : fmt::format("{} {}", name, i));
            }
        }

        ~worker_pool_t() {
            wait_finish();
        }

        void post(task_t&& task) {
            _tasks.push(std::forward<task_t>(task));
        }

        void wait_finish() {
            _tasks.finish();

            for (auto& thread : _threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

    protected:

        void _loop(size_t id, std::string name) {
            set_thread_name(name);

            if (_log) { _log->debug("{} thread entered", name); }

            if (_setup_task) {
                std::invoke(_setup_task, id);
            }

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif
                task_t task;
                while (_tasks.pop(task)) {
                    std::invoke(task);
                }
#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in {} thread: {}\n{}", name, to_string(e), check_trace(e)); }
            }
#endif

            if (_log) { _log->debug("{} thread exited", name); }
        }

        std::vector<std::thread> _threads;
        vortex::sync::queue_t<task_t> _tasks;

        setup_task_t _setup_task;

        std::shared_ptr<spdlog::logger> _log;

    };

    template<typename... Args>
    class completion_worker_pool_t {
    public:

        using setup_task_t = std::function<void(size_t)>;

        using work_t = std::function<std::tuple<Args...>()>;
        using callback_t = std::function<void(Args...)>;

        completion_worker_pool_t() {}
        completion_worker_pool_t(const std::string& prefix, size_t n = 0, setup_task_t&& setup_task = {}, std::shared_ptr<spdlog::logger> log = nullptr)
            : _work_pool(prefix + " Worker", n, std::forward<setup_task_t>(setup_task), log),
              _done_pool(prefix + " Completion", 1, {}, log) {
        }

        ~completion_worker_pool_t() {
            wait_finish();
        }

        void post(work_t&& work, callback_t&& callback) {
            // create job to share between thread pools
            auto job = std::make_shared<job_t>(std::forward<work_t>(work), std::forward<callback_t>(callback));

            // post work to work pool
            _work_pool.post([job]() {
                // NOTE: use scoped set in case of exception
                sync::event_t::scoped_set_t set_on_exit(job->done);
                // NOTE: the work must catch any exceptions that it wants relayed to the callback
                job->result = std::invoke(job->work);
            });

            // post callback to done pool
            _done_pool.post([job]() {
                job->done.wait();
                std::apply(job->callback, job->result);
            });
        }

        void wait_finish() {
            _work_pool.wait_finish();
            _done_pool.wait_finish();
        }

    protected:

        struct job_t {
            work_t work;
            std::invoke_result_t<work_t> result;

            sync::event_t done;
            callback_t callback;

            job_t() {};
            job_t(work_t&& work_, callback_t&& callback_)
                : work(std::forward<work_t>(work_)), callback(std::forward<callback_t>(callback_)) {}
        };

        worker_pool_t _work_pool, _done_pool;

    };

}
