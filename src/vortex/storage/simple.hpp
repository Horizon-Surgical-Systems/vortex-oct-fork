#pragma once

#include <fstream>

#include <xtensor/containers/xtensor.hpp>

#include <vortex/core.hpp>
#include <vortex/util/cast.hpp>

#include <vortex/storage/detail/header.hpp>

namespace vortex::storage {

    enum class simple_stream_header_t {
        none
    };

    struct simple_stream_config_t {
        std::string path;

        bool buffering = false;
        simple_stream_header_t header = simple_stream_header_t::none;
    };

    template<typename element_t_, typename config_t_>
    class simple_stream_t {
    public:

        using config_t = config_t_;
        using element_t = element_t_;

        simple_stream_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~simple_stream_t() {
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
            if (_log) { _log->debug("opening file \"{}\"", _config.path); }
            _out.exceptions(~std::ios::goodbit);
            if (!_config.buffering) {
                // NOTE: disable buffering before the file is open
                _out.rdbuf()->pubsetbuf(nullptr, 0);
            }
            _out.open(_config.path, std::ios::binary);

            // write out header
            if (_config.header == simple_stream_header_t::none) {
                // no header
            } else {
                raise(_log, "unsupported header format: {}", cast(_config.header));
            }
        }

        template<typename V, typename = std::enable_if_t<std::is_same_v<std::decay_t<typename V::element_t>, element_t>>>
        void write(const cpu_viewable<V>& data_) {
            auto data = data_.derived_cast();

            if (!data.is_contiguous()) {
                raise(_log, "data must be contiguous in memory");
            }

            // write out the data
            detail::write_raw(_out, data.data(), data.count());
        }

        virtual void close() {
            if (ready()) {
                if (_log) { _log->debug("closing file \"{}\"", _config.path); }
                _out.close();
            }
        }

        virtual bool ready() const {
            return _out.is_open();
        }

    protected:

        std::shared_ptr<spdlog::logger> _log;

        config_t _config;

        std::ofstream _out;

    };

    enum class simple_stack_header_t {
        none,
        numpy,
        matlab,
        nrrd,
        nifti
    };

    struct simple_stack_config_t {
        std::string path;

        std::array<size_t, 4> shape = { {0, 0, 0, 1} };

        auto& channels_per_sample() { return shape[3]; }
        const auto& channels_per_sample() const { return shape[3]; }
        size_t& samples_per_ascan() { return shape[2]; }
        const size_t& samples_per_ascan() const { return shape[2]; }
        size_t& ascans_per_bscan() { return shape[1]; }
        const size_t& ascans_per_bscan() const { return shape[1]; }
        size_t& bscans_per_volume() { return shape[0]; }
        const size_t& bscans_per_volume() const { return shape[0]; }

        auto volume_shape() const {
            return shape;
        }
        std::array<size_t, 3> bscan_shape() const {
            return { ascans_per_bscan(), samples_per_ascan(), channels_per_sample() };
        }
        std::array<size_t, 2> ascan_shape() const {
            return { samples_per_ascan(), channels_per_sample() };
        }

        bool buffering = false;
        simple_stack_header_t header = simple_stack_header_t::numpy;
    };

    template<typename element_t_, typename config_t_>
    class simple_stack_t {
    public:

        using config_t = config_t_;
        using element_t = element_t_;

        simple_stack_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~simple_stack_t() {
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
            if (_log) { _log->debug("opening file \"{}\"", _config.path); }
            _out.exceptions(~std::ios::goodbit);
            if (!_config.buffering) {
                // NOTE: disable buffering before the file is open
                _out.rdbuf()->pubsetbuf(nullptr, 0);
            }
            _out.open(_config.path, std::ios::binary);

            // write out header
            if (_config.header == simple_stack_header_t::none) {
                // no header
            } else if (_config.header == simple_stack_header_t::numpy ||
                _config.header == simple_stack_header_t::matlab ||
                _config.header == simple_stack_header_t::nrrd ||
                _config.header == simple_stack_header_t::nifti) {

                // placeholder for final shape (volume count is unknown)
                std::vector<size_t> shape;
                shape.push_back(0);
                std::copy(_config.shape.begin(), _config.shape.end(), std::back_inserter(shape));

                // write prototype header
                if (_config.header == simple_stack_header_t::numpy) {
                    if (_log) { _log->trace("writing out provisional NumPy header"); }
                    detail::numpy::write_header<element_t>(_out, shape);
                } else if(_config.header == simple_stack_header_t::matlab) {
                    if (_log) { _log->trace("writing out provisional MATLAB header"); }
                    detail::matlab::write_header<element_t>(_out, shape);
                } else if (_config.header == simple_stack_header_t::nrrd) {
                    if (_log) { _log->trace("writing out provisional NRRD header"); }
                    detail::nrrd::write_header<element_t>(_out, shape);
                } else if (_config.header == simple_stack_header_t::nifti) {
                    if (_log) { _log->trace("writing out provisional NIfTI header"); }
                    detail::nifti::write_header<element_t>(_out, shape);
                } else {
                    raise(_log, "unsupported header format: {} (internal logic error)", cast(_config.header));
                }

            } else {
                raise(_log, "unsupported header format: {}", cast(_config.header));
            }

            // record header end
            _header_position = _volume_position = _farthest_write_position = _out.tellp();
        }

        virtual void seek(size_t volume_index, size_t bscan_index) {
            _volume_position = _header_position + volume_index * _volume_size_in_bytes();
            _out.seekp(_volume_position + bscan_index * _bscan_size_in_bytes());
        }

        template<typename V>
        void write_partial_bscan(size_t bscan_index, size_t ascan_index, const cpu_viewable<V>& raw_chunk_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            auto& raw_chunk = raw_chunk_.derived_cast();

            if (_log) { _log->trace("writing B-scan {}, A-scans {}-{}", bscan_index, ascan_index, ascan_index + raw_chunk.shape(0) - 1); }

            // perform shape checking
            fixed_cpu_view_t<const element_t, 3> chunk;
            try {
                chunk = raw_chunk.morph_right(3);
            }
            catch (const incompatible_shape&) {
                raise(_log);
            }
            if (!equal(tail<2>(chunk.shape()), _config.ascan_shape())) {
                raise(_log, "A-scan shape mismatch: [{}] vs [{}]", shape_to_string(tail<2>(chunk.shape())), shape_to_string(_config.ascan_shape()));
            }

            // perform bounds checking
            if (bscan_index >= _config.bscans_per_volume()) {
                raise(_log, "attempt to seek past B-scans per volume: {} >= {}", bscan_index, _config.bscans_per_volume());
            }
            if (ascan_index >= _config.ascans_per_bscan()) {
                raise(_log, "attempt to seek past A-scans per B-scan: {} >= {}", ascan_index, _config.ascans_per_bscan());
            }
            if (ascan_index + chunk.shape(0) > _config.ascans_per_bscan()) {
                raise(_log, "attempt to write past end of volume: {} >= {}", ascan_index + chunk.shape(0), _config.ascans_per_bscan());
            }

            if (!chunk.is_contiguous()) {
                raise(_log, "A-scans must be contiguous in memory");
            }

            // seek to correct position
            _out.seekp(_volume_position + bscan_index * _bscan_size_in_bytes() + ascan_index * _ascan_size_in_bytes());

            // write out the A-scans
            detail::write_raw(_out, chunk.data(), chunk.count());
            _farthest_write_position = std::max<size_t>(_farthest_write_position, _out.tellp());
        }

        template<typename V>
        void write_multi_bscan(size_t index, const cpu_viewable<V>& raw_chunk_) {
            static_assert(std::is_same_v<std::decay_t<typename V::element_t>, element_t>, "incompatible types");
            auto& raw_chunk = raw_chunk_.derived_cast();

            if (_log) { _log->trace("writing B-scans {}-{}", index, index + raw_chunk.shape(0) - 1); }

            // perform shape checking
            fixed_cpu_view_t<const element_t, 4> chunk;
            try {
                chunk = raw_chunk.morph_right(4);
            }
            catch (const incompatible_shape&) {
                raise(_log);
            }
            if (!equal(tail<3>(chunk.shape()), _config.bscan_shape())) {
                raise(_log, "B-scan shape mismatch: {} != {}", shape_to_string(tail<3>(chunk.shape())), shape_to_string(_config.bscan_shape()));
            }

            // perform bounds checking
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
        void write_volume(const cpu_viewable<V>& volume) {
            write_multi_bscan(0, volume);
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
            if (!ready()) {
                return;
            }

            // check if need to finish the final volume
            if ((_farthest_write_position - _header_position) % _volume_size_in_bytes() != 0) {
                if (_log) { _log->debug("completing unfinished volume"); }
                seek((_farthest_write_position - _header_position) / _volume_size_in_bytes(), 0);
                advance_volume();
            }

            // check if need to finish the header
            if (_config.header == simple_stack_header_t::numpy ||
                _config.header == simple_stack_header_t::matlab ||
                _config.header == simple_stack_header_t::nrrd ||
                _config.header == simple_stack_header_t::nifti ) {

                // build actual shape
                std::vector<size_t> shape;
                shape.push_back((_farthest_write_position - _header_position) / _volume_size_in_bytes());
                std::copy(_config.shape.begin(), _config.shape.end(), std::back_inserter(shape));

                // seek back to start
                _out.seekp(0);

                // write final header
                size_t written;
                if (_config.header == simple_stack_header_t::numpy) {
                    if (_log) { _log->trace("finalizing NumPy header"); }
                    written = detail::numpy::write_header<element_t>(_out, shape);
                } else if (_config.header == simple_stack_header_t::matlab) {
                    if (_log) { _log->trace("finalizing MATLAB header"); }
                    written = detail::matlab::write_header<element_t>(_out, shape);
                    detail::matlab::write_final_padding(_out);
                } else if (_config.header == simple_stack_header_t::nrrd) {
                    if (_log) { _log->trace("finalizing NRRD header"); }
                    written = detail::nrrd::write_header<element_t>(_out, shape);
                } else if (_config.header == simple_stack_header_t::nifti) {
                    if (_log) { _log->trace("finalizing NIfTI header"); }
                    written = detail::nifti::write_header<element_t>(_out, shape);
                } else {
                    raise(_log, "unsupported header format: {} (internal logic error)", cast(_config.header));
                }

                // check that header did not overrun data
                if (written > _header_position) {
                    raise(_log, "header reservation too small: {} > {}", written, _header_position);
                }
            }

            if (_log) { _log->debug("closing file \"{}\"", _config.path); }
            _out.close();
        }

        virtual bool ready() const {
            return _out.is_open();
        }

    protected:

        size_t _volume_size_in_bytes() {
            return _config.bscans_per_volume() * _bscan_size_in_bytes();
        }

        size_t _bscan_size_in_bytes() {
            return _config.ascans_per_bscan() * _ascan_size_in_bytes();
        }

        size_t _ascan_size_in_bytes() {
            return _config.samples_per_ascan() * _sample_size_in_bytes();
        }

        size_t _sample_size_in_bytes() {
            return _config.channels_per_sample() * sizeof(element_t);
        }

        std::shared_ptr<spdlog::logger> _log;

        config_t _config;

        std::ofstream _out;
        size_t _header_position, _volume_position, _farthest_write_position;

    };

}
