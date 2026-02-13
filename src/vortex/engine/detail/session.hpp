#pragma once

#include <vector>
#include <map>
#include <set>
#include <thread>
#include <fstream>
#include <filesystem>

#include <xtensor/core/xnoalias.hpp>

#include <vortex/core.hpp>

#include <vortex/engine/common.hpp>

#include <vortex/driver/cuda/runtime.hpp>

#include <vortex/memory/cuda.hpp>

#include <vortex/storage/detail/raw.hpp>

#include <vortex/version.hpp>

#include <vortex/util/cast.hpp>
#include <vortex/util/sync.hpp>
#include <vortex/util/variant.hpp>
#include <vortex/util/ring.hpp>
#include <vortex/util/platform.hpp>

namespace vortex::engine::detail {

    template<typename config_t, typename master_plan_t, typename scan_queue_t, typename block_t>
    class session_t {
    protected:

        struct job_t {
            size_t id = 0;
            block_t* block;

            size_t count = 0;
            std::exception_ptr error;
            bool last = false;

            size_t format_ascan_index, format_spectra_index;

            timing_t timing;
        };

    public:

        using clock_t = std::chrono::high_resolution_clock;

        session_t(config_t& config, master_plan_t& master_plan, std::vector<block_t>& blocks, scan_queue_t& scan_queue, event_callback_t event_callback, job_callback_t job_callback, std::exception_ptr& shutdown_exception, std::shared_ptr<spdlog::logger> log)
            : _config(config), _master_plan(master_plan), _scan_queue(scan_queue), _log(log),
              _acquired_jobs(_number_of_acquire_tasks()), finished_jobs(_number_of_format_tasks()),
              _launch(_config.preload_count),
              _event_callback(std::move(event_callback)), _job_callback(std::move(job_callback)),
              _shutdown_exception(shutdown_exception) {

            // unrestrict launching
            if (_config.preload_count == 0) {
                _launch.finish();
            }

            // load blocks
            for (auto& block : blocks) {
                _available_blocks.push(&block);
            }

            // create chain and processed queues for each formatter
            for (auto& [f, _] : _config.formatters) {
                processed_jobs_per_format[f];
            }

            // determine initial scan sample
            // NOTE: need to move the scan queue far enough ahead in the future to allow for leading
            auto max_lead = _max_io_lead_delay();
            _scan_queue.rebase(max_lead);

            // determine initial scan position
            auto [sample, position, velocity] = _scan_queue.state();
            // use zeros if start scan position is undefined
            auto zeros = xt::zeros<typename block_t::analog_element_t>({ _config.galvo_output_channels });
            xt::xtensor<typename block_t::analog_element_t, 2> hold = xt::view(position.value_or(zeros), xt::newaxis(), xt::all());

            for (const auto& lead : _master_plan.io_lead_samples) {
                // determine how much this signal needs to be delayed from the largest lead
                size_t n = max_lead - lead;

                if (n == 0) {
                    // ring buffer will not be used
                } else {
                    // initialize with the scan start position
                    _galvo_target_ring_buffers[lead].buffer = xt::repeat(hold, n, 0);

                    // initialize with default strobes
                    _strobe_ring_buffers[lead].buffer.resize({ n, size_t(1) });
                    _strobe_ring_buffers[lead].buffer.fill(_config.lead_strobes);
                }
            }

            // configure profiling
            {
                auto var = envvar(VORTEX_PROFILER_ENVVAR);
                if (var) {
                    auto path = std::filesystem::path(*var);
                    if (_log) { _log->debug("open profiler log at \"{}\"", path.string()); }

                    // ensure directories exist
                    if (!path.parent_path().empty()) {
                        std::filesystem::create_directories(path.parent_path());
                    }
                    _profiler_log.open(path, std::ios::binary);

                    // write version information
                    _profiler_queue.push({ VORTEX_PROFILER_META, VORTEX_VERSION, VORTEX_PROFILER_VERSION, {} });

                    _profiler_thread = std::thread(&session_t::_profiler_loop, this);
                }
            }

            // start the session
            _main_thread = std::thread(&session_t::_main_loop, this);
        }

    public:

        ~session_t() {
            quit();

            if (_main_thread.joinable()) {
                _main_thread.join();
            }

            _profiler_queue.finish();
            if (_profiler_thread.joinable()) {
                _profiler_thread.join();
            }
            if (_profiler_log.is_open()) {
                if (_log) { _log->debug("closing profiler log"); }
                _profiler_log.close();
            }
        }

        auto done() const {
            return _session_complete.status();
        }

        void wait() const {
            _session_complete.wait();
        }
        bool wait_for(const std::chrono::high_resolution_clock::duration& timeout) const {
            return _session_complete.wait_for(timeout);
        }

        void quit(bool interrupt = false) {
            _quit(interrupt, {});
        };

        auto status() const {
            return session_status_t{
                _config.blocks_to_allocate, _inflight.query(),
                _dispatch_index, _config.blocks_to_acquire
            };
        }

    protected:

        //
        // thread functions
        //

        void _quit(bool interrupt, const std::exception_ptr& error) {
            // check if shutdown already in progress
            if (_shutdown) {
                return;
            }

            // record exception that triggers shutdown
            _shutdown_exception = error;

            if (interrupt) {
                if (_log) { _log->warn("shutdown requested with interruption"); }
                _interrupt = true;
            } else {
                if (_log) { _log->warn("shutdown requested"); }
            }
            _available_blocks.finish();

            _shutdown = true;
            _emit(event_t::shutdown, error);
        }

        void _abort() {
            // capture error
            auto error = std::current_exception();

            if (_log) { _log->critical("aborting with {} inflight tasks", _inflight.query()); }

            // notify and trigger shutdown
            _emit(event_t::abort, error);
            _quit(true, error);

            // hard stop which potentially ignores remaining work
            _launch.finish();
            _inflight.finish();
        }

        void _main_loop() {
            vortex::set_thread_name("Session Worker");
            if (_log) { _log->debug("session starting"); }
            _emit(event_t::launch);

            // prepare acquisitions
            _prepare();

            // launch workers
            _workers.emplace_back(&session_t::_recycle_loop, this);
            size_t n = 0;
            for (auto& [f, _] : _config.formatters) {
                _workers.emplace_back(&session_t::_format_loop, this, n++, f);
            }
            _workers.emplace_back(&session_t::_process_loop, this);
            _workers.emplace_back(&session_t::_acquire_loop, this);

            // wait for job dispatch to finish
            _emit(event_t::run);
            _dispatch_complete.wait();

            // wait for pending jobs to finish
            if (_log) { _log->debug("waiting for pending jobs to finish"); }
            _emit(event_t::stop);
            _inflight.wait();

            // stop all acquisitions and IO
            _stop(false);
            // stop the masters last
            _stop(true);

            // shutdown the remainder of the pipeline
            if (_log) { _log->debug("shutting down session"); }
            _acquired_jobs.finish();
            for (auto& [_, q] : processed_jobs_per_format) {
                q.finish();
            }
            finished_jobs.finish();

            // join all threads
            for (auto& worker : _workers) {
                worker.join();
            }

            // ready to clean up
            _session_complete.set();
            if (_log) { _log->debug("session exited"); }
            _emit(event_t::exit);

            // avoid superfluous quit call in destructor
            _shutdown = true;
        }

        void _acquire_loop() {
            vortex::set_thread_name("Acquire Loop");
            vortex::setup_realtime();
            if (_log) { _log->debug("acquire loop entered"); }

            auto post_scan_records = _config.post_scan_records;

            // use the slowest (largest lead) signal as reference to minimize copies
            auto ref_lead = _max_io_lead_delay();

            // resting state of strobes
            auto strobe_idle = _strobe_idle();

            // markers that were generated too far in advance for a given block due to leading
            std::vector<typename block_t::marker_t> future_markers;
            future_markers.push_back(_config.lead_marker);
            future_markers.push_back(marker::inactive_lines{});

            struct deferred_preload_t {
                job_t job;
                size_t task_index;
            };
            std::map<std::variant<typename config_t::adapter::acquisition, typename config_t::adapter::io>, std::vector<deferred_preload_t>> deferred;

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif

                while (!_interrupt) {
                    // start up a new job
                    job_t job;
                    job.id = _dispatch_index++;
                    job.last = _shutdown || (_config.blocks_to_acquire > 0 && _dispatch_index >= _config.blocks_to_acquire);
                    job.timing.create = _profile(profiler_job_mark_t::create, job.id);

                    // wait for launch clearance
                    _launch--;

                    // wait for next available block to service job
                    if (!_available_blocks.pop(job.block)) {
                        break;
                    }
                    job.timing.service = _profile(profiler_job_mark_t::clearance, job.id);
                    if (_log) { _log->trace("created job {} (block {})", job.id, job.block->id); }

                    // reset the block header
                    job.block->timestamp = {};
                    job.block->sample = _dispatch_sample;
                    // load with future markers
                    job.block->markers = std::move(future_markers);

                    // generate scan pattern
                    // NOTE: store into slowest (highest lead) signal because all others will be delayed from it
                    auto [scan_records, total_records] = _scan_queue.generate(job.block->markers, view(job.block->galvo_target[ref_lead]), _config.records_per_block);
                    _profile(profiler_job_mark_t::generate_scan, job.id);

                    // update block header
                    job.block->length = total_records;
                    _dispatch_sample += job.block->length;

                    {
                        // store future markers for later
                        auto is_present = [&](auto& marker) {
                            return std::visit([&](auto& m) { return m.sample < job.block->sample + job.block->length; }, marker);
                        };
                        auto it = std::partition_point(job.block->markers.begin(), job.block->markers.end(), is_present);
                        future_markers.assign(std::make_move_iterator(it), std::make_move_iterator(job.block->markers.end()));
                        job.block->markers.erase(it, job.block->markers.end());
                    }

                    // generate record index
                    view(job.block->counter).to_xt() = job.block->sample + xt::arange(job.block->length);

                    // generate strobes
                    _generate_strobes(*job.block, strobe_idle, ref_lead);
                    _profile(profiler_job_mark_t::generate_strobe, job.id);

                    // generate leading waveforms
                    for (auto lead : _master_plan.io_lead_samples) {
                        // skip the reference as it has been loaded directly
                        if (lead == ref_lead) {
                            continue;
                        }

                        // delay signals through ring buffers
                        _delay_signals_with_ring(lead, ref_lead, total_records, _galvo_target_ring_buffers, job.block->galvo_target);
                        _delay_signals_with_ring(lead, ref_lead, total_records, _strobe_ring_buffers, job.block->strobes);
                    }

                    // generate sample target waveform from the zero-lead signal
                    {
                        auto in = view(job.block->galvo_target[0]).to_xt();
                        auto out = view(job.block->sample_target).to_xt();
                        CALL_CONST(_config.scanner_warp, forward, in, out);
                    }

                    if (scan_records < total_records) {
                        auto overscan = total_records - scan_records;
                        if (overscan >= post_scan_records) {
                            // sufficient post-scan records obtained
                            if (_log) { _log->info("scan is complete"); }
                            job.last = true;
                        } else {
                            // reduce the post-scan requirement
                            post_scan_records -= overscan;
                        }
                    }
                    job.timing.scan = _profile(profiler_job_mark_t::acquire_dispatch_begin, job.id);

                    // send for acquisition
                    _inflight++;

                    auto preload = job.id + 1 <= _config.preload_count;
                    auto start = (job.id + 1 == _config.preload_count) || (job.last && preload) || (job.id == 0 && !preload);

                    if (_log) { _log->trace("sending job {} (block {}) for {}", job.id, job.block->id, preload ? "preload" : "acquisition"); }
                    size_t task_idx = 0;
                    _dispatch(_config.acquisitions, job, preload, deferred, task_idx);
                    _dispatch(_config.ios, job, preload, deferred, task_idx);

                    // start after preload has completed
                    if (start) {

                        // start non-masters before masters
                        for (auto master : { false, true }) {
                            _start(_config.acquisitions, master, deferred);
                            _start(_config.ios, master, deferred);
                        }

                        _emit(event_t::start);
                    }
                    _profile(profiler_job_mark_t::acquire_dispatch_end, job.id);

                    if (job.last) {
                        if (_log) { _log->info("dispatching of blocks completed"); }
                        _emit(event_t::complete);
                        break;
                    }
                }

#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in acquire loop: {}\n{}", to_string(e), check_trace(e)); }
                _abort();
            }
#endif

            // signal that dispatch is done
            _dispatch_complete.set();

            // wait for inflight blocks to finish
            // NOTE: if this wait is not performed, then waits on Alazar buffers report cancellation
            // NOTE: this likely due to some interaction of the async operations with how Alazar handle buffers (e.g., posting thread must stay alive until wait is complete)
            if (_log) { _log->debug("acquire loop waiting for inflight tasks to finish"); }
            _inflight.wait();

            // signal no further jobs
            if (_log) { _log->debug("acquire loop exited"); }
        }

        void _process_loop() {
            vortex::set_thread_name("Process Loop");
            vortex::setup_realtime();
            if (_log) { _log->debug("process loop entered"); }

            struct state_t {
                size_t process_set_index = 0;
            };
            std::map<typename config_t::adapter::acquisition, state_t> states;

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif

                while (true) {
                    std::vector<job_t> jobs;

                    // wait for acquire job to complete
                    if (!_acquired_jobs.pop(jobs)) {
                        break;
                    }
                    auto& job = jobs.front();
                    job.timing.acquire = _profile(profiler_job_mark_t::acquire_join, job.id);
                    if (_log) { _log->trace("acquired job {} (block {})", job.id, job.block->id); }

                    // renew launch clearance
                    _launch++;

                    // check for errors
                    std::exception_ptr first_error;
                    for (auto& job : jobs) {
                        if (job.error) {
                            if (_log) { _log->error("during acquisition of job {} (block {}): {}", job.id, job.block->id, vortex::to_string(job.error)); }
                            _emit(job.error);
                            if (!first_error) {
                                first_error = job.error;
                            }
                        }
                    }

                    // check for empty acquisition, which signals graceful completion
                    auto n = std::min_element(jobs.begin(), jobs.end(), [](auto& a, auto& b) { return a.count < b.count; })->count;
                    if (_interrupt || first_error || n == 0) {
                        _quit(false, first_error);
                        _inflight--;
                        continue;
                    }

                    // generate the sample actual waveforms
                    {
                        auto in = view(job.block->galvo_actual).to_xt();
                        auto out = view(job.block->sample_actual).to_xt();
                        CALL_CONST(_config.scanner_warp, forward, in, out);
                    }

                    // send for processing
                    _profile(profiler_job_mark_t::process_dispatch_begin, job.id);
                    if (_log) { _log->trace("sending job {} (block {}) for processing", job.id, job.block->id); }

                    // determine which processors to invoke
                    std::set<size_t> updated_buffer;
                    for (auto& [acquire, acquire_plan] : _master_plan.acquire) {

                        // update state
                        auto& state = states[acquire];
                        auto& process_set = acquire_plan.rotation[state.process_set_index];
                        state.process_set_index = (state.process_set_index + 1) % acquire_plan.rotation.size();

                        for (auto& process : process_set) {
                            auto& process_plan = _master_plan.process[process];

                            // check if buffer has already been updated because processors may share the same buffers
                            auto [_, absent] = updated_buffer.insert(process_plan.input_index);

                            // transfer the data as needed
                            if (absent && process_plan.input_index != acquire_plan.output_index) {

                                // a transfer is required
                                std::visit([&](auto& src, auto& dst) {
                                    _transfer(process.channel(), _choose_stream(view(src), view(dst)), view(src), view(dst), process_plan.start);
                                }, job.block->spectra[acquire_plan.output_index], job.block->spectra[process_plan.input_index]);
                            }

                            // assemble queues for pushing results
                            std::vector<sync::queue_t<job_t>*> queues;
                            for (auto& f : _config.processors[process].graph) {
                                queues.push_back(&processed_jobs_per_format[f]);
                            }

                            // launch processing
                            job.format_spectra_index = process_plan.input_index;
                            job.format_ascan_index = process_plan.output_index;
                            process.next_async(*job.block, job.block->spectra[process_plan.input_index], job.block->ascans[process_plan.output_index], &process_plan.start, &process_plan.done, _push_unsized(std::move(queues), process_plan.output_index, job, profiler_task_mark_t::process_complete));
                        }
                    }
                    _profile(profiler_job_mark_t::process_dispatch_end, job.id);

                    if (job.last) {
                        if (_log) { _log->info("acquired all blocks"); }
                        break;
                    }
                }

#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in process loop: {}\n{}", to_string(e), check_trace(e)); }
                _abort();
            }
#endif

            // signal no further jobs
            if (_log) { _log->debug("process loop exited"); }
        }

        void _format_loop(size_t index, typename config_t::adapter::formatter format) {
            vortex::set_thread_name(fmt::format("Format Loop {}", index));
            vortex::setup_realtime();
            if (_log) { _log->debug("format loop {} entered", index); }

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif

                auto& processed_jobs = processed_jobs_per_format[format];
                auto& format_plan = _master_plan.format[format];

                while (true) {
                    job_t job;

                    // wait for acquire job to complete
                    if (!processed_jobs.pop(job)) {
                        break;
                    }
                    job.timing.process = _profile(profiler_task_mark_t::format_begin, index, job.id);
                    if (_log) { _log->trace("processed job {} (block {})", job.id, job.block->id); }

                    // check for errors
                    if (job.error) {
                        if (_log) { _log->error("during processing of job {} (block {}): {}", job.id, job.block->id, vortex::to_string(job.error)); }
                        _emit(job.error);
                    }

                    if (_interrupt || job.error) {
                        _quit(false, job.error);
                        finished_jobs.push(format_plan.format_index, std::move(job));
                        continue;
                    }

                    if (_log) { _log->trace("sending job {} (block {}) for formatting", job.id, job.block->id); }

                    // generate format plan
                    auto plan = format.next(*job.block);
                    if (job.last) {
                        format.finish(plan);
                    }
                    _profile(profiler_task_mark_t::format_plan, index, job.id);

                    // find buffers corresponding to this endpoint
                    auto& spectra_stream = job.block->spectra[job.format_spectra_index];
                    auto& ascan_stream = job.block->ascans[job.format_ascan_index];

#if defined(VORTEX_EXCEPTION_GUARDS)
                    try {
#endif
                        // apply plan to endpoints
                        for (auto& endpoint : _config.formatters[format].graph) {
                            endpoint.handle(plan, *job.block, spectra_stream, ascan_stream);
                        }
#if defined(VORTEX_EXCEPTION_GUARDS)
                    } catch (const std::exception& e) {
                        if (_log) { _log->error("error during formatting job {} (block {}): {}", job.id, job.block->id, to_string(e)); }
                        _emit();
                    }
#endif

                    // send block for recycling
                    job.timing.format = _profile(profiler_task_mark_t::format_end, index, job.id);
                    finished_jobs.push(format_plan.format_index, std::move(job));

                    if (job.last) {
                        if (_log) { _log->info("processed all blocks"); }
                        break;
                    }
                }

#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in format loop {}: {}\n{}", index, to_string(e), check_trace(e)); }
                _abort();
            }
#endif

            // no signaling
            if (_log) { _log->debug("format loop {} exited", index); }
        }

        void _recycle_loop() {
            vortex::set_thread_name("Recycle Loop");
            vortex::setup_realtime();
            if (_log) { _log->debug("recycle loop entered"); }

#if defined(VORTEX_EXCEPTION_GUARDS)
            try {
#endif

                while (true) {
                    std::vector<job_t> jobs;

                    // wait for job to finish
                    if (!finished_jobs.pop(jobs)) {
                        break;
                    }
                    if (jobs.empty()) {
                        raise(_log, "received empty job list to recycle (configuration error?)");
                    }
                    auto& job = jobs.front();
                    job.timing.recycle = _profile(profiler_job_mark_t::format_join, job.id);

                    // cleanup
                    if (_log) { _log->trace("formatted job {} and recycling block {}", job.id, job.block->id); }
                    for (auto& [a, cfg] : _config.acquisitions) {
                        auto output_index = _master_plan.acquire[a].output_index;
                        a.recycle(*job.block, job.block->spectra[output_index]);
                    }
                    _available_blocks.push(job.block);
                    _inflight--;

                    // handle job callback
                    if (_job_callback) {
                        try {
                            std::invoke(_job_callback, job.id, status(), job.timing);
                        } catch (const std::exception& e) {
                            if (_log) { _log->error("error during job callback: {}", to_string(e)); }
                            _emit();
                        }
                    }
                    _profile(profiler_job_mark_t::recycle, job.id);

                    // check for completions
                    if (job.last) {
                        if (_log) { _log->debug("formatted all blocks"); }
                        break;
                    }
                }

#if defined(VORTEX_EXCEPTION_GUARDS)
            } catch (const std::exception& e) {
                if (_log) { _log->critical("unhandled exception in recycle loop: {}\n{}", to_string(e), check_trace(e)); }
                _abort();
            }
#endif

            // signal no further jobs
            if (_log) { _log->debug("recycle loop exited"); }
        }

        //
        // profiler
        //

        void _profiler_loop() {
            vortex::set_thread_name("Profiler Worker");
            vortex::setup_realtime();
            if (_log) { _log->debug("profiler worker entered"); }

            profiler_entry_t entry;
            while (_profiler_queue.pop(entry)) {
                storage::detail::write_raw(_profiler_log, entry.code);
                storage::detail::write_raw(_profiler_log, entry.index);
                storage::detail::write_raw(_profiler_log, entry.job);
                storage::detail::write_raw(_profiler_log, entry.timestamp.time_since_epoch().count());
            }

            if (_log) { _log->debug("profiler worker exited"); }
        }

        auto _profile(const event_t& code) const {
            return _profile(cast(code), 0, 0);
        }
        auto _profile(const profiler_job_mark_t& code, size_t job) const {
            return _profile(cast(code) + 0x100, 0, job);
        }
        auto _profile(const profiler_task_mark_t& code, size_t index, size_t job) const {
            return _profile(cast(code) + 0x10000, index, job);
        }
        auto _profile(size_t code, size_t index, size_t job) const {
            auto timestamp = std::chrono::high_resolution_clock::now();
            if (_profiler_log.is_open()) {
                _profiler_queue.push({ code, index, job, timestamp });
            }
            return timestamp;
        }

        //
        // helpers for callbacks that push onto queues/aggregators
        //

        template<typename T>
        auto _aggregate_sized(sync::aggregator_t<T>& aggregator, size_t index, T job, const profiler_task_mark_t& code) {
            return[this, &aggregator, job = std::move(job), index, code](size_t n, std::exception_ptr error) mutable {
                job.count = n;
                job.error = error;
                _profile(code, index, job.id);
                aggregator.push(index, std::move(job));
            };
        }
        template<typename T>
        auto _push_unsized(sync::queue_t<T>& queue, size_t index, T job, const profiler_task_mark_t& code) {
            return[this, &queue, job = std::move(job), index, code](std::exception_ptr error) mutable {
                job.error = error;
                _profile(code, index, job.id);
                queue.push(std::move(job));
            };
        }
        template<typename T>
        auto _push_unsized(std::vector<sync::queue_t<T>*> queues, size_t index, T job, const profiler_task_mark_t& code) {
            return[this, queues = std::move(queues), job = std::move(job), index, code](std::exception_ptr error) mutable {
                job.error = error;
                _profile(code, index, job.id);
                for (auto& queue : queues) {
                    queue->push(job);
                }
            };
        }

        //
        // acquire and I/O helpers
        //

        void _prepare() {
            for (const auto& [a, v] : _config.acquisitions) {
                a.prepare();
            }
            for (const auto& [io, v] : _config.ios) {
                io.prepare();
            }
        }

        template<typename object_map_t, typename deferred_map_t>
        void _dispatch(const object_map_t& objects, const job_t& job, bool preload, deferred_map_t& deferred, size_t& task_idx) {
            for (auto& [obj, cfg] : objects) {
                // only defer if preloading but object does not support it
                if (preload && !cfg.preload) {
                    deferred[obj].push_back({ job, task_idx });
                } else {
                    _dispatch(obj, cfg, job, task_idx);
                }

                task_idx++;
            }
        }
        void _dispatch(const typename config_t::adapter::acquisition& a, const typename config_t::acquire_config& config, const job_t& job, size_t task_index) {
            auto output_index = _master_plan.acquire[a].output_index;
            a.next_async(*job.block, job.block->spectra[output_index], _aggregate_sized(_acquired_jobs, task_index, job, profiler_task_mark_t::acquire_complete));
        }
        void _dispatch(const typename config_t::adapter::io& io, const typename config_t::io_config& config, const job_t& job, size_t task_index) {
            io.next_async(*job.block, job.block->streams(config.lead_samples), _aggregate_sized(_acquired_jobs, task_index, job, profiler_task_mark_t::acquire_complete));;
        }

        template<typename object_map_t, typename deferred_map_t>
        void _start(const object_map_t& objects, bool master, deferred_map_t& deferred) {
            for (const auto& [obj, cfg] : objects) {
                // check master status
                if (cfg.master != master) {
                    continue;
                }

                // start the object
                obj.start();

                // look for deferred preloads
                auto it = deferred.find(obj);
                if (it == deferred.end()) {
                    continue;
                }

                // consume the deferred list
                auto preloads = std::move(it->second);
                deferred.erase(it);

                // dispatch the deferred preloads
                if (_log) { _log->debug("releasing {} deferred preload jobs", preloads.size()); }
                for (auto& preload : preloads) {
                    _dispatch(obj, cfg, preload.job, preload.task_index);
                }
            }
        }

        void _stop(bool master) {
            for (const auto& [a, v] : _config.acquisitions) {
                if (v.master == master) {
                    a.stop();
                }
            }
            for (const auto& [io, v] : _config.ios) {
                if (v.master == master) {
                    io.stop();
                }
            }
        }

        //
        // CUDA copy helper
        //

        template<typename V1, typename V2>
        cuda::stream_t& _choose_stream(const cpu_viewable<V1>& src_, const cuda::cuda_viewable<V2>& dst_) {
            auto& dst = dst_.derived_cast();

            return _master_plan.transfer_streams[dst.device()];
        }
        template<typename V1, typename V2>
        cuda::stream_t& _choose_stream(const cuda::cuda_viewable<V1>& src_, const viewable<V2>& dst_) {
            auto& src = src_.derived_cast();

            return _master_plan.transfer_streams[src.device()];
        }
        template<typename V1, typename V2>
        [[ noreturn ]] cuda::stream_t& _choose_stream(const cpu_viewable<V1>& src_, const cpu_viewable<V2>& dst_) {
            raise(_log, "unexpected need to transfer between host buffers (probably an implementation error)");
        }

        template<typename V1, typename V2>
        void _transfer(size_t channel, cuda::stream_t& stream, const viewable<V1>& src_, const viewable<V2>& dst_, cuda::event_t& done) {
            auto& src = src_.derived_cast();
            auto& dst = dst_.derived_cast();

            if (src.dimension() != 3) {
                raise(_log, "transfer source must have dimension 3: {}", shape_to_string(src.shape()));
            }

            // if (src.shape()[2] > 1) {
            //     // deinterleaving required

            //     auto src_begin = strided_offset(src.stride(), 0, 0, channel);
            //     auto dst_begin = 0;

            //     cuda::copy(
            //         src, src_begin, src.stride(2),   // stride across all channels in the source
            //         dst, dst_begin, dst.stride(2),   // contiguous channels in the destination
            //         src.shape(0) * src.shape(1), 1,  // each sample is a "row" so the pitch is the offset between them during copying
            //         &stream
            //     );
            // } else {
                // simply copy
                cuda::copy(src, dst, &stream);
            // }

            // mark completion
            done.record(stream);
        }

        //
        // ring buffer management for I/O leading
        //

        template<typename R, typename S>
        void _delay_signals_with_ring(size_t lead, size_t ref_lead, size_t total_records, R& rings, S& signals) {
            // number of samples that ring buffer needs to store
            auto n = std::min(ref_lead - lead, total_records);

            // load records from past in buffer
            {
                auto& buf = rings[lead].buffer;
                auto dst = view(signals[lead]).to_xt();
                rings[lead].ring.load(buf, dst, n);
            }

            // time shift current records
            if (total_records > n) {
                auto src = view(signals[ref_lead]).range(0, total_records - n).to_xt();
                auto dst = view(signals[lead]).range(n, total_records).to_xt();
                xt::noalias(dst) = src;
            }

            // store records for future in buffer
            {
                auto& buf = rings[lead].buffer;
                auto src = view(signals[ref_lead]).to_xt();
                rings[lead].ring.store(buf, src, n, total_records - n);
            }
        }

        //
        // strobe generation
        //

        auto _strobe_idle() const {
            // start all lines at digital low
            typename block_t::digital_element_t idle = 0;

            // inspect each strobe polarity
            for_each(_config.strobes, [&](const auto& s) {
                if (s.polarity == strobe::polarity_t::low) {
                    // set bit to digital high
                    idle |= (1 << s.line);
                }
            });

            return idle;
        }

        template<typename marker_t, typename buffer_t, typename strobe_t>
        void _generate_flagged_strobe(block_t& block, const std::set<typename block_t::marker_t>& markers, buffer_t& buf, const strobe_t& s) {
            for_each(markers, overloaded{
                [&](const marker_t& m) {
                    if (!m.flags.matches(s.flags)) {
                        return;
                    }

                    // determine the range of the strobe for this marker
                    auto start = m.sample + s.delay;
                    auto stop = start + s.duration;

                    // for strobes that outlast this block, save the marker for next block
                    if (stop > block.sample + block.length) {
                        _ongoing_strobe_markers.insert(m);
                    }

                    // shift relative to this block
                    auto local_start = std::max(start, block.sample) - block.sample;
                    auto local_stop = std::min(stop, block.sample + block.length) - block.sample;

                    // check if any output in this block
                    if (local_start >= local_stop) {
                        return;
                    }

                    if (s.polarity == strobe::polarity_t::high) {
                        // set all selected bits to digital high
                        xt::view(buf, xt::range(local_start, local_stop), xt::all()) |= (1 << s.line);
                    } else {
                        // set all selected bits to digital low
                        xt::view(buf, xt::range(local_start, local_stop), xt::all()) &= ~(1 << s.line);
                    }
                },
                [&](const auto&) {} // ignore
            });
        }

        void _generate_strobes(block_t& block, typename block_t::digital_element_t idle, size_t ref_lead) {
            // build ongoing strobe markers
            std::set<typename block_t::marker_t> markers = std::move(_ongoing_strobe_markers);
            std::copy(block.markers.begin(), block.markers.end(), std::inserter(markers, markers.begin()));

            // NOTE: store into slowest (highest lead) signal because all others will be delayed from it
            auto buf = view(block.strobes[ref_lead]).to_xt();

            // start by clearing out prior signals
            buf = xt::broadcast(idle, block.strobes[ref_lead].shape());

            // sample indices for this block
            auto n = xt::expand_dims(xt::arange(block.sample, block.sample + block.length), 1);

            // create each requested strobe
            for_each(_config.strobes, overloaded{
                [&](const strobe::sample& s) {
                    // compute strobe level over time as Boolean
                    auto b = (n + s.phase) % s.divisor < s.duration;
                    // convert to word and shift strobe to correct bit
                    auto w = xt::cast<typename block_t::digital_element_t>(b) << s.line;

                    if (s.polarity == strobe::polarity_t::high) {
                        // set all selected bits to digital high
                        buf |= w;
                    } else {
                        // set all selected bits to digital low
                        buf &= ~w;
                    }
                },
                [&](const strobe::segment& s) { _generate_flagged_strobe<marker::segment_boundary>(block, markers, buf, s); },
                [&](const strobe::volume& s)  { _generate_flagged_strobe<marker::volume_boundary> (block, markers, buf, s); },
                [&](const strobe::scan& s)    { _generate_flagged_strobe<marker::scan_boundary>   (block, markers, buf, s); },
                [&](const strobe::event& s)   { _generate_flagged_strobe<marker::event>           (block, markers, buf, s); },
            });
        }

        //
        // other helpers
        //

        auto _number_of_acquire_tasks() const {
            return _config.acquisitions.size() + _config.ios.size();
        }
        auto _number_of_format_tasks() const {
            return _master_plan.format.size();
        }

        void _emit() const {
            _emit(std::current_exception());
        }
        void _emit(const std::exception_ptr& error) const {
            _emit(event_t::error, error);
        }
        void _emit(event_t event, const std::exception_ptr& error = {}) const {
            _profile(event);
            if (_event_callback) {
                try {
                    std::invoke(_event_callback, event, error);
                } catch (const std::exception& e) {
                    if (_log) { _log->error("error during event callback: {}", to_string(e)); }
                }
            }
        }

        auto _max_io_lead_delay() const {
            // the list is sorted from smallest to largest
            return _master_plan.io_lead_samples.back();
        }

        //
        // shared state with engine
        //

        config_t& _config;
        master_plan_t& _master_plan;
        scan_queue_t& _scan_queue;
        event_callback_t _event_callback;
        job_callback_t _job_callback;
        std::exception_ptr& _shutdown_exception;

        std::shared_ptr<spdlog::logger> _log;

        //
        // session status
        //

        std::atomic_bool _interrupt = false, _shutdown = false;
        sync::event_t _dispatch_complete, _session_complete;

        //
        // threads
        //

        std::thread _main_thread;
        std::vector<std::thread> _workers;

        //
        // block tracking and generation
        //

        std::atomic<counter_t> _dispatch_sample = 0;
        std::atomic<size_t> _dispatch_index = 0;

        sync::queue_t<block_t*> _available_blocks;

        //
        // job tracking
        //

        sync::aggregator_t<job_t> _acquired_jobs;
        std::map<typename config_t::adapter::formatter, sync::queue_t<job_t>> processed_jobs_per_format;
        sync::aggregator_t<job_t> finished_jobs;

        sync::counter_t _inflight, _launch;

        //
        // IO lead handling
        //

        template<typename T>
        struct ring_buffer {
            xt::xtensor<T, 2> buffer;
            ring_buffer_xt ring;
        };
        std::unordered_map<size_t, ring_buffer<typename block_t::analog_element_t>> _galvo_target_ring_buffers;
        std::unordered_map<size_t, ring_buffer<typename block_t::digital_element_t>> _strobe_ring_buffers;

        std::set<typename block_t::marker_t> _ongoing_strobe_markers;

        //
        // profiling
        //

        mutable sync::queue_t<profiler_entry_t> _profiler_queue;
        std::ofstream _profiler_log;
        std::thread _profiler_thread;

    };

}
