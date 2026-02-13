#include <fstream>

#include <fmt/printf.h>

#include <xtensor/io/xnpy.hpp>

#include <vortex/scan.hpp>
#include <vortex/storage.hpp>
#include <vortex/version.hpp>

using namespace std::literals::string_literals;

template<typename marker_t>
void dump_markers(const std::string& path, const std::vector<marker_t>& markers) {
    vortex::marker_log_t::config_t mlc;
    mlc.path = path;

    vortex::marker_log_t ml;
    ml.open(mlc);

    ml.write(markers);
}

template<typename scan_t>
void generate_scan_full(const std::string& base_path, scan_t& scan) {

    // calling scan_buffer() or scan_markers() will automatically generate the complete scan
    xt::dump_npy(base_path + ".pattern.npy", scan.scan_buffer());
    dump_markers(base_path + ".markers.txt", scan.scan_markers());

}

template<typename scan_t>
void generate_scan_blocks(const std::string& base_path, scan_t& scan, size_t block_count, size_t block_length) {

    // allocate a block to store the scan
    vortex::cpu_tensor_t<double> scan_buffer;
    scan_buffer.resize({ block_length, scan.config().channels_per_sample });
    std::vector<vortex::default_marker_t> markers;

    size_t n = 0;
    while (n < block_count) {

        // generate the next block
        auto released = scan.next(markers, view(scan_buffer));

        // save out the block's scan pattern and markers
        xt::dump_npy(fmt::format("{}.{:06d}.pattern.npy", base_path, n), view(scan_buffer).to_xt());
        dump_markers(fmt::format("{}.{:06d}.markers.txt", base_path, n), markers);

        // check if scan has completed
        if (released < block_length) {
            break;
        }

        n++;

    }

}

void usage_and_exit(int argc, char* argv[]) {
    fmt::print("usage: generate-scan (raster | radial | spiral) (full | blocks) output_base_path");
    std::exit(-1);
}

template<typename scan_t, typename scan_config_t>
void run(int argc, char* argv[], scan_t& scan, scan_config_t& cfg) {
    try {
        scan.initialize(cfg);
    } catch (const std::exception& e) {
        fmt::print("scan initialization failed: {}", e.what());
        std::exit(-2);
    }

    bool full = (argv[2] == "full"s);
    bool blocks = (argv[2] == "blocks"s);

    if (full) {
        generate_scan_full(argv[3], scan);
    } else if (blocks) {
        generate_scan_blocks(argv[3], scan, 100, 100);
    } else {
        usage_and_exit(argc, argv);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        usage_and_exit(argc, argv);
    }

    bool raster = (argv[1] == "raster"s);
    bool radial = (argv[1] == "radial"s);
    bool spiral = (argv[1] == "spiral"s);

    if (raster) {

        vortex::raster_scan_t::config_t cfg;
        // TODO: adjust the scan options as needed

        // create the scan object
        vortex::raster_scan_t scan;
        run(argc, argv, scan, cfg);

    } else if (radial) {

        vortex::radial_scan_t::config_t cfg;
        // TODO: adjust the scan options as needed

        // create the scan object
        vortex::radial_scan_t scan;
        run(argc, argv, scan, cfg);

    } else if (spiral) {

        vortex::spiral_scan_t::config_t cfg;
        // TODO: adjust the scan options as needed

        // the default constant-angular velocity spiral exceeds the default scanner limits
        cfg.angular_velocity = 1;
        cfg.linear_velocity = 0;
        cfg.bypass_limits_check = true;

        // create the scan object
        vortex::spiral_scan_t scan;
        run(argc, argv, scan, cfg);

    } else {
        usage_and_exit(argc, argv);
    }

    return 0;
}
