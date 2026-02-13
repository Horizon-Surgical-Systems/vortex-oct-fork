#pragma once

#include <optional>
#include <functional>
#include <memory>

#include <vortex/acquire.hpp>

#include <vortex/io.hpp>

#include <vortex/process.hpp>

#include <vortex/engine.hpp>
#include <vortex/engine/clock.hpp>
#include <vortex/engine/dispersion.hpp>
#include <vortex/engine/source.hpp>

#include <vortex/endpoint/common.hpp>
#include <vortex/endpoint/cuda.hpp>
#include <vortex/endpoint/storage.hpp>

#include <vortex/storage.hpp>

namespace vortex::simple {

    struct simple_engine_config_t {

        using source_t = engine::source_t<double>;

        alazar::channel_t input_channel = alazar::channel_t::B;
        alazar::channel_t clock_channel = alazar::channel_t::A;
        bool internal_clock = false;

        std::string broct_save_path, sample_target_save_path, sample_actual_save_path;

        source_t swept_source = { 100'000, 1376, 0.50 };

        double galvo_delay = 0.0;

        std::array<double, 2> dispersion = { {0, 0} };

        size_t samples_per_ascan = 2752;
        size_t ascans_per_bscan = 500;
        size_t bscans_per_volume = 500;

        size_t blocks_to_acquire = 0;
        size_t blocks_to_allocate = 128;
        size_t ascans_per_block = 500;

        size_t preload_count = 4;
        size_t process_slots = 2;

    };

    class simple_engine_t {
    public:

        using acquisition_t = alazar_acquisition_t;
        using processor_t = cuda_processor_t<acquisition_t::output_element_t, int8_t>;
        using io_t = daqmx_io_t;
        using engine_t = vortex::engine_t<acquisition_t::output_element_t, processor_t::output_element_t, io_t::analog_element_t, io_t::digital_element_t>;

        simple_engine_t(std::string log_scope = "")
            : _log_scope(std::move(log_scope)) {}

        void initialize(simple_engine_config_t config) {
            std::swap(_config, config);

            //
            // acquisition
            //

            auto& ac = _acquire_config;
            auto& pc = _process_config;

            ac.device = { 1, 1 };

            if (_config.internal_clock) {
                // TODO: make the clock rate an option
                ac.clock = alazar::clock::internal_t{ 800'000'000 };
            } else {
                ac.clock = alazar::clock::external_t{};
            }

            ac.inputs.push_back(alazar::input_t{ _config.input_channel });
            ac.options.push_back(alazar::option::auxio_trigger_out_t{});

            ac.records_per_block() = _config.ascans_per_block; // ac.recommended_minimum_records_per_block();

            //
            // clocking
            //

            alazar::board_t board(ac.device.system_index, ac.device.board_index);
            if (_config.internal_clock) {
                auto [clock_samples_per_second, clock] = acquire_alazar_clock<acquisition_t>(_config.swept_source, ac, _config.clock_channel);
                _config.swept_source.clock_edges_seconds() = engine::find_rising_edges(clock, clock_samples_per_second, _config.swept_source.clock_edges_seconds().size());
                auto resampling = engine::compute_resampling(_config.swept_source, ac.samples_per_second(), _config.samples_per_ascan);

                // resampling
                pc.resampling_samples = xt::cast<processor_t::float_element_t>(resampling);

                // acquire enough samples to obtain the required ones
                ac.samples_per_record() = board.info().smallest_aligned_samples_per_record(xt::amax(resampling, xt::evaluation_strategy::immediate)(0));
            } else {
                ac.samples_per_record() = board.info().smallest_aligned_samples_per_record(double(_config.swept_source.clock_rising_edges_per_trigger));
            }

            _acquire = std::make_shared<acquisition_t>(spdlog::get(_log_scope + "acquire"));
            _acquire->initialize(ac);

            //
            // OCT processing setup
            //

            pc.samples_per_record() = ac.samples_per_record();
            pc.ascans_per_block() = ac.records_per_block();

            pc.slots = _config.process_slots;

            // spectral filter
            auto window = 0.5 - 0.5 * xt::cos(2 * pi * xt::arange<ptrdiff_t>(pc.samples_per_ascan()) / double(pc.samples_per_ascan() - 1)); // Hamming window
            auto phasor = engine::dispersion_phasor(window.shape(0), _config.dispersion);
            pc.spectral_filter = xt::cast<std::complex<processor_t::float_element_t>>(window * phasor);

            // DC subtraction per block
            // TODO: make this an option
            pc.average_window = 2 * pc.ascans_per_block();

            _process = std::make_shared<processor_t>(spdlog::get(_log_scope + "process"));
            _process->initialize(pc);

            //
            // galvo control
            //

            auto& ioc_out = _io_out_config;
            auto& ioc_in = _io_in_config;

            ioc_out.samples_per_block() = ac.records_per_block();
            ioc_out.samples_per_second() = _config.swept_source.triggers_per_second;
            ioc_out.blocks_to_buffer = _config.preload_count;
            ioc_in = ioc_out;
            ioc_in.clock.divisor = 4;

            ioc_out.name = "output";
            ioc_in.name = "input";

            daqmx::channel::analog_voltage_output_t out;
            out.logical_units_per_physical_unit = 15.0 / 10.0;
            out.stream = cast(engine_t::block_t::stream_index_t::galvo_target);

            out.port_name = "Dev1/ao0";
            out.channel = 0;
            ioc_out.channels.push_back(out);

            out.port_name = "Dev1/ao1";
            out.channel = 1;
            ioc_out.channels.push_back(out);

            daqmx::channel::analog_voltage_input_t in;
            in.logical_units_per_physical_unit = 24.30 / 10.0;
            in.stream = cast(engine_t::block_t::stream_index_t::galvo_actual);

            in.port_name = "Dev1/ai0";
            in.channel = 0;
            ioc_in.channels.push_back(in);

            in.port_name = "Dev1/ai1";
            in.channel = 1;
            ioc_in.channels.push_back(in);

            _io_out = std::make_shared<io_t>(spdlog::get(_log_scope + "output"));
            _io_out->initialize(ioc_out);
            _io_in = std::make_shared<io_t>(spdlog::get(_log_scope + "input"));
            _io_in->initialize(ioc_in);

            //
            // output setup
            //

            format_planner_t::config_t fc;
            fc.segments_per_volume() = _config.bscans_per_volume;
            fc.records_per_segment() = _config.ascans_per_bscan;
            fc.adapt_shape = false;

            // TODO: make this constant an option
            fc.mask = { 0x01 };
            _stack_format = std::make_shared<format_planner_t>(spdlog::get(_log_scope + "format"));
            _stack_format->initialize(fc);

            // TODO: make this constant an option
            fc.mask = { 0x02 };
            _radial_format = std::make_shared<format_planner_t>(spdlog::get(_log_scope + "format"));
            _radial_format->initialize(fc);

            auto samples_to_save = pc.samples_per_ascan() / 2;
            _volume = std::make_shared<sync::lockable<cuda::cuda_device_tensor_t<int8_t>>>();

            auto& cfec = _stack_executor_config;
            cfec.sample_slice = copy::slice::simple_t(samples_to_save);

            _stack_tensor_endpoint = std::make_shared<endpoint::ascan_stack_cuda_device_tensor<processor_t::output_element_t>>(
                std::make_shared<stack_format_executor_t>(), _volume, spdlog::get(_log_scope + "endpoint")
                );
            _stack_tensor_endpoint->executor()->initialize(cfec);

            auto& rfec = _radial_executor_config;
            rfec.sample_slice = copy::slice::simple_t(samples_to_save);
            rfec.volume_xy_extent = { { {-5, 5}, {-5, 5} } };
            rfec.segment_rt_extent = { {{ -5, 5 }, {0, pi}} };
            rfec.radial_segments_per_volume() = _config.bscans_per_volume;
            rfec.radial_records_per_segment() = _config.ascans_per_bscan;

            _radial_tensor_endpoint = std::make_shared<endpoint::ascan_radial_cuda_device_tensor<processor_t::output_element_t>>(
                std::make_shared<radial_format_executor_t>(), _volume, spdlog::get(_log_scope + "endpoint")
                );
            _radial_tensor_endpoint->executor()->initialize(rfec);

            _volume->resize({ rfec.radial_records_per_segment(), rfec.radial_records_per_segment(), samples_to_save });

            if (_config.broct_save_path.size() > 0) {
                _broct_storage_endpoint = std::make_shared<endpoint::broct_storage>(
                    std::make_shared<broct_format_executor_t>(), std::make_shared<broct_storage_t>(spdlog::get(_log_scope + "storage")), spdlog::get(_log_scope + "endpoint")
                );

                auto& bfec = _broct_executor_config;
                bfec.sample_slice = copy::slice::simple_t(samples_to_save);
                _broct_storage_endpoint->executor()->initialize(bfec);

                auto& bc = _broct_storage_config;
                bc.path = _config.broct_save_path;
                bc.shape = { fc.segments_per_volume(), fc.records_per_segment(), samples_to_save };
                _broct_storage_endpoint->storage()->open(bc);
            }

            if (_config.sample_target_save_path.size() > 0) {
                auto& dc = _sample_target_dump_config;
                dc.path = _config.sample_target_save_path;
                dc.stream = cast(engine_t::block_t::stream_index_t::sample_target);

                _sample_target_endpoint = std::make_shared<endpoint::stream_dump_storage>(std::make_shared<stream_dump_t>(spdlog::get(_log_scope + "dump")));
                _sample_target_endpoint->storage()->open(dc);
            }

            if (_config.sample_actual_save_path.size() > 0) {
                auto& dc = _sample_actual_dump_config;
                dc.path = _config.sample_actual_save_path;
                dc.stream = cast(engine_t::block_t::stream_index_t::sample_actual);

                _sample_actual_endpoint = std::make_shared<endpoint::stream_dump_storage>(std::make_shared<stream_dump_t>(spdlog::get(_log_scope + "dump")));
                _sample_actual_endpoint->storage()->open(dc);
            }

            //
            // engine setup
            //

            auto& ec = _engine_config;
            ec.add_acquisition(_acquire, true, true, _process);
            ec.add_processor(_process, _stack_format, _radial_format);
            ec.add_io(_io_in, false);
            ec.add_io(_io_out, true, false, static_cast<size_t>(std::round(_config.galvo_delay * ioc_out.samples_per_second())));

            ec.add_formatter(_stack_format, _stack_tensor_endpoint);
            ec.add_formatter(_radial_format, _radial_tensor_endpoint);
            if (_broct_storage_endpoint) {
                ec.add_formatter(_stack_format, _broct_storage_endpoint);
            }
            if (_sample_target_endpoint) {
                ec.add_formatter(_stack_format, _sample_target_endpoint);
            }
            if (_sample_actual_endpoint) {
                ec.add_formatter(_stack_format, _sample_actual_endpoint);
            }

            ec.preload_count = _config.preload_count;
            ec.records_per_block = _config.ascans_per_block;
            ec.blocks_to_allocate = _config.blocks_to_allocate;
            ec.blocks_to_acquire = _config.blocks_to_acquire;

            ec.galvo_input_channels = _io_in->config().channels.size();
            ec.galvo_output_channels = _io_out->config().channels.size();

            _engine.initialize(ec);
            _engine.prepare();
        }

        template<typename S>
        void append_scan(std::shared_ptr<S>& scan) {
            _engine.scan_queue()->append(scan);
        }
        template<typename S>
        void interrupt_scan(std::shared_ptr<S>& scan) {
            _engine.scan_queue()->interrupt(scan);
        }

        using callback_t = std::function<void(bool, std::vector<size_t>)>;
        void start(callback_t callback = {}) {
            if (callback) {
                _stack_tensor_endpoint->aggregate_segment_callback = [this, callback](std::vector<size_t> bscans) {
                    _stack_tensor_endpoint->stream().sync();
                    callback(false, std::move(bscans));
                };
                _radial_tensor_endpoint->aggregate_segment_callback = [this, callback](std::vector<size_t> bscans) {
                    _radial_tensor_endpoint->stream().sync();
                    callback(true, std::move(bscans));
                };
            }

            _engine.start();
        }

        void wait() const {
            _engine.wait();
        }
        bool wait_for(const std::chrono::high_resolution_clock::duration& timeout) const {
            return _engine.wait_for(timeout);
        }

        void stop() {
            _engine.stop();
            wait();
            if (_broct_storage_endpoint) {
                _broct_storage_endpoint->storage()->close();
            }
            if (_sample_target_endpoint) {
                _sample_target_endpoint->storage()->close();
            }
            if (_sample_actual_endpoint) {
                _sample_actual_endpoint->storage()->close();
            }
        }

        const auto& volume() const {
            return _volume;
        }

    protected:

        acquisition_t::config_t _acquire_config;
        std::shared_ptr<acquisition_t> _acquire;

        processor_t::config_t _process_config;
        std::shared_ptr<processor_t> _process;

        io_t::config_t _io_out_config, _io_in_config;
        std::shared_ptr<io_t> _io_out, _io_in;

        std::shared_ptr<format_planner_t> _stack_format, _radial_format;

        stack_format_executor_t::config_t _stack_executor_config;
        radial_format_executor_t::config_t _radial_executor_config;
        std::shared_ptr<endpoint::ascan_stack_cuda_device_tensor<processor_t::output_element_t>> _stack_tensor_endpoint;
        std::shared_ptr<endpoint::ascan_radial_cuda_device_tensor<processor_t::output_element_t>> _radial_tensor_endpoint;

        broct_storage_t::config_t _broct_storage_config;
        broct_format_executor_t::config_t _broct_executor_config;
        std::shared_ptr<endpoint::broct_storage> _broct_storage_endpoint;

        stream_dump_t::config_t _sample_target_dump_config, _sample_actual_dump_config;
        std::shared_ptr<endpoint::stream_dump_storage> _sample_target_endpoint, _sample_actual_endpoint;

        std::shared_ptr<sync::lockable<cuda::cuda_device_tensor_t<int8_t>>> _volume;

        engine_t::config_t _engine_config;
        engine_t _engine;

        simple_engine_config_t _config;
        std::string _log_scope;

    };

}
