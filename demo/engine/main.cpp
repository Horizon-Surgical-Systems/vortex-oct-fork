#include <fstream>

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <xtensor/io/xnpy.hpp>
#include <xtensor/core/xnoalias.hpp>
#include <xtensor/misc/xcomplex.hpp>

#include <fmt/printf.h>

#include <lyra/lyra.hpp>

#include <vortex/acquire.hpp>
#include <vortex/process.hpp>

#include <vortex/io.hpp>

#include <vortex/format.hpp>
#include <vortex/format/broct.hpp>

#include <vortex/engine/clock.hpp>
#include <vortex/engine/dispersion.hpp>
#include <vortex/engine/source.hpp>
#include <vortex/endpoint/cuda.hpp>
#include <vortex/endpoint/storage.hpp>
#include <vortex/engine.hpp>

#include <vortex/scan.hpp>

#include <vortex/storage.hpp>

#include <vortex/util/sync.hpp>
#include <vortex/util/platform.hpp>
#include <vortex/util/exception.hpp>

using sink_t = spdlog::sinks::stderr_color_sink_mt;
#if defined(NDEBUG)
#   define make_logger(name) spdlog::create_async<sink_t>(name)
#else
#   define make_logger(name) spdlog::create<sink_t>(name)
#endif

static auto logger           = make_logger("main");
static auto acquire_logger   = make_logger("acquire");
static auto process_logger   = make_logger("process");
static auto scan_logger      = make_logger("scan");
static auto input_logger     = make_logger("input");
static auto output_logger    = make_logger("output");
static auto format_logger    = make_logger("format");
static auto storage_logger   = make_logger("storage");
static auto engine_logger    = make_logger("engine");

using acquire_t = vortex::alazar_acquisition_t;
using processor_t = vortex::cuda_processor_t<acquire_t::output_element_t, int8_t>;
using io_t = vortex::daqmx_io_t;
using source_t = vortex::engine::source_t<double>;
using engine_t = vortex::engine_t<acquire_t::output_element_t, processor_t::output_element_t, io_t::analog_element_t, io_t::digital_element_t>;

template<typename scan_t, typename source_t, typename format_config_t, typename engine_t>
void setup_scan(typename scan_t::config_t& sc, source_t& source, format_config_t& fc, engine_t& engine, typename engine_t::config_t& ec) {
    sc.samples_per_second = source.triggers_per_second;

    fc.segments_per_volume() = sc.bscans_per_volume();
    fc.records_per_segment() = sc.ascans_per_bscan();

    ec.scanner_warp = sc.warp;

    auto scan = std::make_shared<scan_t>();
    scan->initialize(sc);
    engine.scan_queue()->append(scan);
}

int run(int argc, char* argv[]) {
    //
    // defaults
    //

    size_t internal_clock = 0;

    size_t blocks_to_acquire = 10;

    size_t buffer_count = 4;
    size_t preload_count = 2;
    size_t process_slots = 2;

    size_t samples_per_ascan = 2752;
    size_t ascans_per_block = 500;

    double galvo_delay = 0.0;

    std::array<double, 2> dispersion = { {0, 0} };

    std::map<std::string, source_t> sources = {
        { "Axsun100k",    { 100'000, 1376, 0.50 } },
        { "Axsun200k",    { 200'000, 1024, 0.61 } },
        { "Thorlabs400k", { 400'000, 1024, 0.54 } }
    };
    std::string source_name = sources.begin()->first;
    std::string source_choices;
    for (auto& [k, _] : sources) {
        if (!source_choices.empty()) {
            source_choices += ", ";
        }
        source_choices += k;
    }

    std::string output_path, output_conj_path, format_path, scan_type = "raster";

    size_t log_level = spdlog::level::info;

    //
    // argument parsing
    //

    bool help = false;
    auto cli = lyra::help(help)
        | lyra::opt(source_name, "source")["--source"].choices([&](const std::string& v) { return sources.find(v) != sources.end(); })("laser source: " + source_choices)
        | lyra::opt(samples_per_ascan, "sample")["--samples"]("number of samples in an A-scan")
        | lyra::opt(ascans_per_block, "block-width")["--block-width"]("number of A-scans in a block")
        | lyra::opt(dispersion[0], "dispersion")["--dispersion"]("dispersion correction coefficient")
        | lyra::opt(galvo_delay, "galvo-delay")["--galvo-delay"]("galvo delay in seconds")
        | lyra::opt(buffer_count, "buffers")["--buffers"]("number of pipelined buffers")
        | lyra::opt(preload_count, "preload")["--preload"]("number of buffers to preload")
        | lyra::opt(process_slots, "slots")["--slots"]("number of processing slots")
        | lyra::opt(blocks_to_acquire, "blocks")["--blocks"]("number of blocks to acquire")
        | lyra::opt(internal_clock, "internal-clock")["--internal-clock"]("activate internal clocking at given samples per second")
        | lyra::opt(output_path, "output")["--output"]("path to save acquired data")
        | lyra::opt(output_conj_path, "output-conjugate")["--output-conjugate"]("path to save acquired conjugate data")
        | lyra::opt(format_path, "output-format")["--output-format"]("path to formatted data")
        | lyra::opt(scan_type, "scan")["--scan"].choices("raster", "radial") //, "spiral")
        | lyra::opt(log_level, "log-level")["--log-level"](fmt::format("log level ({} = trace, {} = off)", spdlog::level::trace, spdlog::level::off));
    ;

    auto result = cli.parse({ argc, argv });
    if (!result) {
        std::cout << "invalid arguments: " << result.errorMessage() << std::endl;
        return -1;
    } else if (help) {
        std::cout << cli << std::endl;
        return 0;
    }

    spdlog::set_level(spdlog::level::level_enum(log_level));

    acquire_t::config_t ac;
    processor_t::config_t pc;
    io_t::config_t ioc_in, ioc_out;
    auto& swept_source = sources[source_name];

    //
    // Alazar acquisition setup
    //

    ac.device = { 1, 1 };

    if (internal_clock) {
        ac.clock = vortex::acquire::clock::internal_t{ internal_clock };
    } else {
        ac.clock = vortex::acquire::clock::external_t{};
    }

    ac.inputs.push_back(vortex::acquire::input_t{ vortex::alazar::channel_t::B });
    ac.options.push_back(vortex::acquire::option::auxio_trigger_out_t{});

    ac.records_per_block() = ascans_per_block; // ac.recommended_minimum_records_per_block();

    //
    // clocking
    //

    vortex::alazar::board_t board(ac.device.system_index, ac.device.board_index);
    if (internal_clock) {
        auto [clock_samples_per_second, clock] = acquire_alazar_clock<acquire_t>(swept_source, ac, vortex::alazar::channel_t::A);
        swept_source.clock_edges_seconds() = vortex::engine::find_rising_edges(clock, clock_samples_per_second, swept_source.clock_edges_seconds().size());
        auto resampling = vortex::engine::compute_resampling(swept_source, ac.samples_per_second(), samples_per_ascan);

        // resampling
        pc.resampling_samples = xt::cast<processor_t::float_element_t>(resampling);

        // acquire enough samples to obtain the required ones
        ac.samples_per_record() = board.info().smallest_aligned_samples_per_record(xt::amax(resampling, xt::evaluation_strategy::immediate)(0));
    } else {
        ac.samples_per_record() = board.info().smallest_aligned_samples_per_record(double(swept_source.clock_rising_edges_per_trigger));
    }
    // adhere to minimum record limits
    ac.samples_per_record() = std::max<size_t>(ac.samples_per_record(), board.info().min_samples_per_record);

    //
    // OCT processing setup
    //

    pc.samples_per_record() = ac.samples_per_record();
    pc.ascans_per_block() = ac.records_per_block();

    pc.slots = process_slots;

    // spectral filter
    auto window = 0.5 - 0.5 * xt::cos(2 * vortex::pi * xt::arange<ptrdiff_t>(pc.samples_per_ascan()) / double(pc.samples_per_ascan() - 1)); // Hamming window
    auto phasor = vortex::engine::dispersion_phasor(window.shape(0), dispersion);
    pc.spectral_filter = xt::cast<std::complex<processor_t::float_element_t>>(window * phasor);

    // DC subtraction per B-scan
    pc.average_window = 2 * pc.ascans_per_block();

    //
    // galvo driving
    //

    ioc_out.samples_per_block() = ac.records_per_block();
    ioc_out.samples_per_second() = swept_source.triggers_per_second;
    ioc_out.blocks_to_buffer = preload_count;
    ioc_in = ioc_out;

    ioc_out.name = "output";
    ioc_in.name = "input";

    vortex::io::channel::analog_voltage_output_t out;
    out.logical_units_per_physical_unit = 15.0 / 10.0;
    out.stream = vortex::cast(engine_t::block_t::stream_index_t::galvo_target);

    out.port_name = "Dev1/ao0";
    out.channel = 0;
    ioc_out.channels.push_back(out);

    out.port_name = "Dev1/ao1";
    out.channel = 1;
    ioc_out.channels.push_back(out);

    vortex::io::channel::analog_voltage_input_t in;
    in.logical_units_per_physical_unit = 24.30 / 10.0;
    in.stream = vortex::cast(engine_t::block_t::stream_index_t::galvo_actual);

    in.port_name = "Dev1/ai0";
    in.channel = 0;
    ioc_in.channels.push_back(in);

    in.port_name = "Dev1/ai1";
    in.channel = 1;
    ioc_in.channels.push_back(in);

    //
    // output setup
    //

    vortex::format_planner_t::config_t fc;

    //
    // scan pattern
    //

    engine_t::config_t ec;
    engine_t engine(spdlog::get("engine"));

    if (scan_type == "raster") {
        vortex::raster_scan_t::config_t sc;

        // use a higher-density scan than is default
        sc.bscans_per_volume() = 500;
        sc.ascans_per_bscan() = 500;
        sc.bscan_extent() = { -3, 3 };
        sc.volume_extent() = { -3, 3 };
        sc.bidirectional_segments = false;
        sc.bidirectional_volumes = false;

        setup_scan<vortex::raster_scan_t>(sc, swept_source, fc, engine, ec);

    } else if (scan_type == "radial") {
        vortex::radial_scan_t::config_t sc;

        // use a higher-density scan than is default
        sc.bscans_per_volume() = 500;
        sc.ascans_per_bscan() = 500;
        sc.bscan_extent() = { -3, 3 };
        sc.volume_extent() = { -3, 3 };
        sc.bidirectional_segments = false;
        sc.bidirectional_volumes = false;

        setup_scan<vortex::radial_scan_t>(sc, swept_source, fc, engine, ec);

    //} else if (scan_type == "spiral") {
    //    vortex::spiral_scan_t::config_t sc;

    //    setup_scan<vortex::spiral_scan_t>(sc, swept_source, fc, engine, ec);
    }

    //
    // initialize components and setup engine
    //

    std::shared_ptr<vortex::sync::lockable<vortex::cuda::cuda_device_tensor_t<int8_t>>> formatted_volume;

    auto samples_to_save = pc.samples_per_ascan() / 2;

    auto acquire = std::make_shared<acquire_t>(spdlog::get("acquire"));
    acquire->initialize(ac);

    {
        auto processor = std::make_shared<processor_t>(spdlog::get("process"));
        processor->initialize(pc);
        ec.add_acquisition(acquire, true, true, processor);

        auto format = std::make_shared<vortex::format_planner_t>(spdlog::get("format"));
        format->initialize(fc);
        ec.add_processor(processor, format);

        if (output_path.size() > 0) {
            auto bse = std::make_shared<vortex::endpoint::broct_storage>(
                std::make_shared<vortex::broct_format_executor_t>(),
                std::make_shared<vortex::broct_storage_t>(spdlog::get("storage")),
                spdlog::get("endpoint")
            );
            ec.add_formatter(format, bse);

            vortex::format::stack_format_executor_config_t bfec;
            bfec.sample_slice = vortex::copy::slice::simple_t(samples_to_save);
            bse->executor()->initialize(bfec);

            vortex::broct_storage_t::config_t  bc;
            bc.path = output_path;
            bc.shape = { fc.segments_per_volume(), fc.records_per_segment(), samples_to_save };
            bse->storage()->open(bc);
        }

        // add a device tensor formatter just for testing
        if (scan_type == "raster") {

            vortex::stack_format_executor_t::config_t cfec;
            cfec.sample_slice = vortex::copy::slice::simple_t(samples_to_save);

            auto endpoint = std::make_shared<vortex::endpoint::ascan_stack_cuda_device_tensor<processor_t::output_element_t>>(
                std::make_shared<vortex::stack_format_executor_t>(),
                std::vector<size_t>{ fc.segments_per_volume(), fc.records_per_segment(), samples_to_save },
                spdlog::get("endpoint")
            );
            endpoint->executor()->initialize(cfec);
            formatted_volume = endpoint->tensor();

            ec.add_formatter(format, endpoint);

        } else if (scan_type == "radial") {

            vortex::radial_format_executor_t::config_t rfec;
            rfec.sample_slice = vortex::copy::slice::simple_t(samples_to_save);
            rfec.volume_xy_extent = { { {-5, 5}, {-5, 5} } };
            rfec.segment_rt_extent = { { { -5, 5 }, {0, vortex::pi} } };
            rfec.radial_segments_per_volume() = fc.segments_per_volume();
            rfec.radial_records_per_segment() = fc.records_per_segment();

            auto endpoint = std::make_shared<vortex::endpoint::ascan_radial_cuda_device_tensor<processor_t::output_element_t>>(
                std::make_shared<vortex::radial_format_executor_t>(),
                std::vector<size_t>{ fc.segments_per_volume() , fc.segments_per_volume() , samples_to_save },
                spdlog::get("endpoint")
            );
            endpoint->executor()->initialize(rfec);
            formatted_volume = endpoint->tensor();

            ec.add_formatter(format, endpoint);


        //} else if (scan_type == "spiral") {

        //    vortex::spiral_format_executor_t::config_t spec;
        //    spec.sample_slice = vortex::copy::slice::simple_t(samples_to_save);
        //    spec.volume_xy_extent = { { {-5, 5}, {-5, 5} } };
        //    //spec.segment_rt_extent = { { { -5, 5 }, {0, vortex::pi} } };
        //    //spec.spiral_segments_per_volume() = fc.segments_per_volume();
        //    //spec.spiral_records_per_segment() = fc.records_per_segment();

        //    auto endpoint = std::make_shared<vortex::endpoint::spiral_galvo_cuda_device_tensor<processor_t::output_element_t>>(
        //        std::make_shared<vortex::spiral_format_executor_t>(),
        //        std::vector<size_t>{ fc.segments_per_volume() , fc.segments_per_volume() , samples_to_save },
        //        spdlog::get("endpoint")
        //    );
        //    endpoint->executor()->initialize(spec);
        //    formatted_volume = endpoint->tensor();

        //    ec.add_formatter(format, endpoint);

        }

    }

    if (output_conj_path.size() > 0) {
        auto processor = std::make_shared<processor_t>(spdlog::get("process"));
        xt::noalias(pc.spectral_filter) = xt::conj(pc.spectral_filter);
        processor->initialize(pc);
        ec.add_acquisition(acquire, true, true, processor);

        auto format = std::make_shared<vortex::format_planner_t>(spdlog::get("format"));
        format->initialize(fc);
        ec.add_processor(processor, format);

        if (output_path.size() > 0) {
            auto bse = std::make_shared<vortex::endpoint::broct_storage>(
                std::make_shared<vortex::broct_format_executor_t>(),
                std::make_shared<vortex::broct_storage_t>(spdlog::get("storage")),
                spdlog::get("endpoint")
            );
            ec.add_formatter(format, bse);

            vortex::format::stack_format_executor_config_t bfec;
            bfec.sample_slice = vortex::copy::slice::simple_t(samples_to_save, pc.samples_per_ascan());
            bse->executor()->initialize(bfec);

            vortex::broct_storage_t::config_t  bc;
            bc.path = output_conj_path;
            bc.shape = { fc.segments_per_volume(), fc.records_per_segment(), samples_to_save };
            bse->storage()->open(bc);
        } else {
            // add a null formatter so the engine can execute, even if the data is not saved
            ec.add_formatter(format, std::make_shared<vortex::endpoint::null>());
        }
    }

    auto io_out = std::make_shared<io_t>(spdlog::get("output"));
    io_out->initialize(ioc_out);
    ec.add_io(io_out, true, false, static_cast<size_t>(std::round(galvo_delay * ioc_out.samples_per_second())));
    auto io_in = std::make_shared<io_t>(spdlog::get("input"));
    io_in->initialize(ioc_in);
    // NOTE: DAQmx input is not preloadable
    ec.add_io(io_in, false);

    //
    // initialize engine
    //

    ec.preload_count = preload_count;
    ec.records_per_block = ascans_per_block;
    ec.blocks_to_allocate = buffer_count;
    ec.blocks_to_acquire = blocks_to_acquire;

    ec.galvo_input_channels = io_in->config().channels.size();
    ec.galvo_output_channels = io_out->config().channels.size();

    engine.initialize(ec);
    engine.prepare();

    vortex::setup_keyboard_interrupt([&]() { engine.stop(); });
    engine.start();

    while (!engine.wait_for(std::chrono::milliseconds(100))) {
        auto status = engine.status();
        logger->info("acquired = {:6d} ( {:4.1f}% )    inflight = {:6d} ( {:4.1f}% )", status.dispatched_blocks, 100.0 * status.dispatch_completion, status.inflight_blocks, 100.0 * status.block_utilization);
    }

    if (format_path.size() > 0) {
        vortex::broct_storage_t::config_t bc;
        bc.path = format_path;
        bc.shape = vortex::to_array<3>(formatted_volume->shape());

        // allocate intermediate storage
        vortex::cuda::cuda_device_tensor_t<int8_t> transposed;
        transposed.resize(bc.broct_volume_shape());

        vortex::cuda::cuda_host_tensor_t<int8_t> host;
        host.resize(transposed.shape());

        std::array<ptrdiff_t, 3> src_stride = { formatted_volume->stride(0), formatted_volume->stride(1), formatted_volume->stride(2) };
        std::array<ptrdiff_t, 3> dst_stride = { transposed.stride(0), transposed.stride(2), transposed.stride(1) };

        // transpose and copy volume to host
        vortex::cuda::stream_t stream;
        vortex::copy::detail::linear_transform(
            stream, 1, 0,
            vortex::to_array<3>(formatted_volume->shape()),
            formatted_volume->data(), 0, src_stride,
            transposed.data(), 0, dst_stride
        );
        vortex::cuda::copy(view(transposed), view(host), &stream);
        stream.sync();

        // save out formatted volume
        vortex::broct_storage_t bs(spdlog::get("storage"));
        bs.open(bc);
        bs.write_volume(view(host));
    }

    return 0;
}

int main(int argc, char* argv[]) {
    vortex::set_thread_name("Main");
    vortex::setup_realtime();

    spdlog::set_pattern("[%d-%b-%Y %H:%M:%S.%f] %-10n %^(%L) %v%$");
    spdlog::enable_backtrace(20);

#if defined(VORTEX_EXCEPTION_GUARDS)
    try {
#endif
        run(argc, argv);
#if defined(VORTEX_EXCEPTION_GUARDS)
    } catch (const std::exception& e) {
        logger->critical("unhandled exception in main thread: {}\n{}", vortex::to_string(e), vortex::check_trace(e));
    }
#endif

    spdlog::shutdown();

    // flush data to profiler
    cudaDeviceReset();
}
