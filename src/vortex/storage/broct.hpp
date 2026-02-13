#pragma once

#include <fstream>

#include <xtensor/containers/xtensor.hpp>

#include <vortex/core.hpp>

#include <vortex/storage/detail/raw.hpp>

#include <vortex/util/cast.hpp>

namespace vortex::storage {

    namespace broct {
        static constexpr int32_t meta = 70;
        static constexpr std::array<size_t, 3> header_axis_order = { 1, 2, 0 };
        static constexpr std::array<size_t, 3> data_axis_order = { 0, 2, 1 };
    }

    enum class broct_scan_t : int32_t {
        rectangular = 0,
        bscan = 1,
        aiming = 2,
        mscan = 3,
        radial = 4,
        ascan = 5,
        speckle = 6,
        mixed = 7,
        xfast_yfast = 8,
        xfast_yfast_speckle = 9,
        spiral = 10
    };

    struct broct_storage_config_t {
        std::string path;

        std::array<size_t, 3> shape = { {0, 0, 0} };
        std::array<double, 3> dimensions = { {1.0, 1.0, 1.0} };

        size_t& samples_per_ascan() { return shape[2]; }
        const size_t& samples_per_ascan() const { return shape[2]; }
        size_t& ascans_per_bscan() { return shape[1]; }
        const size_t& ascans_per_bscan() const { return shape[1]; }
        size_t& bscans_per_volume() { return shape[0]; }
        const size_t& bscans_per_volume() const { return shape[0]; }

        auto broct_volume_shape() const {
            std::array<size_t, 3> out;
            for (size_t i = 0; i < shape.size(); i++) {
                out[i] = shape[broct::data_axis_order[i]];
            }
            return out;
        }
        auto broct_bscan_shape() const {
            std::array<size_t, 2> out;
            for (size_t i = 1; i < shape.size(); i++) {
                out[i - 1] = shape[broct::data_axis_order[i]];
            }
            return out;
        }

        broct_scan_t scan_type = broct_scan_t::rectangular;
        std::string notes;

        bool buffering = false;
    };

    template<typename config_t_>
    class broct_storage_t {
    public:

        using config_t = config_t_;
        using element_t = int8_t;

        broct_storage_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~broct_storage_t() {
            close();
        }

        const auto& config() const {
            return _config;
        }

        virtual void open(config_t config) {
            // close any previously open file
            close();

            // accept configuration
            std::swap(_config, config);

            // open new file
            if (_log) { _log->trace("opening file \"{}\"", _config.path); }
            _out.exceptions(~std::ios::goodbit);
            if (!_config.buffering) {
                // NOTE: disable buffering before the file is open
                _out.rdbuf()->pubsetbuf(nullptr, 0);
            }
            _out.open(_config.path, std::ios::binary);

            // write out header
            if (_log) { _log->trace("writing out header"); }
            detail::write_raw<int32_t>(_out, broct::meta);                          // meta
            for (auto& d : broct::header_axis_order) {
                detail::write_raw(_out, downcast<int32_t>(_config.shape[d]));       // xdim, ydim, zdim
            }
            for (auto& d : broct::header_axis_order) {
                detail::write_raw<int32_t>(_out, 0);                                // xmin, ymin, zmin
                detail::write_raw(_out, downcast<int32_t>(_config.shape[d] - 1));   // xmax, ymax, zmax
            }
            detail::write_raw<int32_t>(_out, 0);                                    // inactive
            for (auto& d : _config.dimensions) {
                detail::write_raw(_out, d);                                         // xlength, ylength, zlength
            }
            detail::write_raw(_out, cast(_config.scan_type));                       // scan type
            detail::write_raw<int32_t>(_out, 0);                                    // big_xdim
            detail::write_raw<int32_t>(_out, 0);                                    // big_xmin
            detail::write_raw<int32_t>(_out, 0);                                    // big_xmax
            detail::write_raw<int32_t>(_out, 0);                                    // big_inactive
            detail::write_raw<int32_t>(_out, 0);                                    // roi
            for (size_t i = 0; i < _config.shape[broct::header_axis_order[2]]; i++) {
                detail::write_raw(_out, downcast<int32_t>(i));                      // scan_map
            }
            detail::write_raw(_out, downcast<int32_t>(_config.notes.size()));       // notes length
            _out.write(_config.notes.data(), _config.notes.size());                 // notes

            // record header end
            _header_position = _volume_position = _farthest_write_position = _out.tellp();
        }

        virtual void seek(size_t volume_index, size_t bscan_index) {
            _volume_position = _header_position + volume_index * _volume_size_in_bytes();
            _out.seekp(_volume_position + bscan_index * _bscan_size_in_bytes());
        }

        template<typename V>
        void write_bscan(size_t index, const cpu_viewable<V>& bscan_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            auto& bscan = bscan_.derived_cast();

            if (_log) { _log->trace("writing B-scan {}", index); }

            if (bscan.dimension() != 2) {
                raise(_log, "B-scan dimension mismatch: {} != {}", bscan.dimension(), 2);
            }
            if (!vortex::equal(bscan.shape(), _config.broct_bscan_shape())) {
                raise(_log, "B-scan shape mismatch: {} != {} (was the B-scan transposed as needed?)", shape_to_string(bscan.shape()), shape_to_string(_config.broct_bscan_shape()));
            }

            if (index >= _config.bscans_per_volume()) {
                raise(_log, "attempt to seek past B-scans per volume: {} >= {}", index, _config.bscans_per_volume());
            }

            if (!bscan.is_contiguous()) {
                raise(_log, "B-scan must be contiguous in memory");
            }

            // seek to correct position
            _out.seekp(_volume_position + index * _bscan_size_in_bytes());

            // write out the B-scan
            detail::write_raw(_out, bscan.data(), bscan.count());
            _farthest_write_position = std::max<size_t>(_farthest_write_position, _out.tellp());
        }

        template<typename V>
        void write_multi_bscan(size_t index, const cpu_viewable<V>& chunk_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            auto& chunk = chunk_.derived_cast();

            if (_log) { _log->trace("writing B-scans {}-{}", index, index + chunk.shape(0)); }

            if (chunk.dimension() != 3) {
                raise(_log, "chunk dimension mismatch: {} != {}", chunk.dimension(), 3);
            }
            if (!equal(tail<2>(chunk.shape()), _config.broct_bscan_shape())) {
                raise(_log, "B-scan shape mismatch: {} != {} (were the B-scans transposed as needed?)", shape_to_string(tail<2>(chunk.shape())), shape_to_string(_config.broct_bscan_shape()));
            }

            if (index >= _config.bscans_per_volume()) {
                raise(_log, "attempt to seek past B-scans per volume: {} >= {}", index, _config.bscans_per_volume());
            }
            if (index + chunk.shape(0) > _config.bscans_per_volume()) {
                raise(_log, "attempt to write past end of volume: {} >= {}", index + chunk.shape(0), _config.bscans_per_volume());
            }

            if (!chunk.is_contiguous()) {
                raise(_log, "B-scans must be contiguous in memory");
            }

            // seek to correct position
            _out.seekp(_volume_position + index * _bscan_size_in_bytes());

            // write out the B-scans
            detail::write_raw(_out, chunk.data(), chunk.count());
            _farthest_write_position = std::max<size_t>(_farthest_write_position, _out.tellp());
        }

        template<typename V>
        void write_partial_bscan(size_t bscan_index, size_t ascan_index, const cpu_viewable<V>& chunk_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            // FIXME
            throw std::runtime_error("BROCT partial B-scan write not yet supported");
            //auto& chunk = chunk_.derived_cast().to_xt();

            //if (_log) { _log->trace("writing B-scan {}, A-scans {}-{}", bscan_index, ascan_index + chunk.shape(0)); }

            //if (chunk.dimension() != 2) {
            //    raise(_log, "partial B-scan dimension mismatch: {} != {}", chunk.dimension(), 2);
            //}
            //if (chunk.shape(1) != _config.samples_per_ascan()) {
            //    raise(_log, "partial B-scan shape mismatch: {} != {}", chunk.shape(1), _config.samples_per_ascan());
            //}

            //if (bscan_index >= _config.bscans_per_volume()) {
            //    raise(_log, "attempt to seek past B-scans per volume: {} >= {}", bscan_index, _config.bscans_per_volume());
            //}
            //if (ascan_index >= _config.ascans_per_bscan()) {
            //    raise(_log, "attempt to seek past A-scans per B-scan: {} >= {}", ascan_index, _config.ascans_per_bscan());
            //}
            //if (ascan_index + chunk.shape(0) > _config.ascans_per_volume()) {
            //    raise(_log, "attempt to write past end of volume: {} >= {}", ascan_index + chunk.shape(0), _config.ascans_per_volume());
            //}

            //if (!chunk.is_contiguous()) {
            //    raise(_log, "A-scans must be contiguous in memory");
            //}

            //// seek to correct position
            //_out.seekp(_volume_position + bscan_index * _bscan_size_in_bytes() + ascan_index * _ascan_size_in_bytes());

            //// write out the A-scans
            //detail::write_raw(_out, &(*chunk.begin()), chunk.size());
            //_farthest_write_position = std::max<size_t>(_farthest_write_position, _out.tellp());
        }


        template<typename V>
        void write_volume(const cpu_viewable<V>& volume_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            auto& volume = volume_.derived_cast();

            if (_log) { _log->trace("writing volume"); }

            if (volume.dimension() != 3) {
                raise(_log, "chunk dimension mismatch: {} != {}", volume.dimension(), 3);
            }
            if (!equal(volume.shape(), _config.broct_volume_shape())) {
                raise(_log, "B-scan shape mismatch: {} != {} (were the B-scans transposed as needed?)", shape_to_string(volume.shape()), shape_to_string(_config.broct_volume_shape()));
            }

            if (!volume.is_contiguous()) {
                raise(_log, "B-scans must be contiguous in memory");
            }

            // seek to correct position (potentially overwriting other B-scans that have already been written)
            _out.seekp(_volume_position);

            // write out the volume
            detail::write_raw(_out, volume.data(), volume.count());
            _volume_position = _out.tellp();
            _farthest_write_position = std::max<size_t>(_farthest_write_position, _out.tellp());
        }

        virtual void advance_volume(bool allocate = true) {
            if (_log) { _log->trace("advancing volume"); }
            _volume_position += _volume_size_in_bytes();

            // only allocate if the file has not been written this far to avoid overwriting a B-scan byte
            if (allocate && _farthest_write_position < _volume_position) {
                if (_log) { _log->trace("allocating next volume on disk"); }

                _out.seekp(_volume_position - 1);
                // write a character to allocate the disk space now
                _out.put(0);
                _farthest_write_position = _out.tellp();
            } else {
                _out.seekp(_volume_position);
            }
        }

        virtual void close() {
            if (_out.is_open()) {
                // check if need to finish the final volume
                if ((_farthest_write_position - _header_position) % _volume_size_in_bytes() != 0) {
                    if (_log) { _log->debug("completing unfinished volume"); }
                    seek((_farthest_write_position - _header_position) / _volume_size_in_bytes(), 0);
                    advance_volume();
                }

                if (_log) { _log->trace("closing file \"{}\"", _config.path); }
                _out.close();
            }
        }

        virtual bool ready() const {
            return _out.is_open();
        }

    protected:

        size_t _volume_size_in_bytes() {
            return _config.bscans_per_volume() * _bscan_size_in_bytes();
        }

        size_t _bscan_size_in_bytes() {
            return _config.ascans_per_bscan() * _config.samples_per_ascan() * sizeof(element_t);
        }

        std::shared_ptr<spdlog::logger> _log;

        config_t _config;

        std::ofstream _out;
        size_t _header_position, _volume_position, _farthest_write_position;

    };

}
