#pragma once

#include <H5Cpp.h>

#include <xtensor/containers/xtensor.hpp>

#include <vortex/core.hpp>
#include <vortex/util/cast.hpp>

#include <vortex/storage/detail/header.hpp>

namespace vortex::storage {

    namespace detail {
        template<typename T> struct datatype_helper {};

        template<> struct datatype_helper<int8_t> { static auto value() { return H5::PredType::STD_I8LE; } };
        template<> struct datatype_helper<int16_t> { static auto value() { return H5::PredType::STD_I16LE; } };
        template<> struct datatype_helper<int32_t> { static auto value() { return H5::PredType::STD_I32LE; } };
        template<> struct datatype_helper<int64_t> { static auto value() { return H5::PredType::STD_I64LE; } };
        template<> struct datatype_helper<uint8_t> { static auto value() { return H5::PredType::STD_U8LE; } };
        template<> struct datatype_helper<uint16_t> { static auto value() { return H5::PredType::STD_U16LE; } };
        template<> struct datatype_helper<uint32_t> { static auto value() { return H5::PredType::STD_U32LE; } };
        template<> struct datatype_helper<uint64_t> { static auto value() { return H5::PredType::STD_U64LE; } };
        template<> struct datatype_helper<float> { static auto value() { return H5::PredType::IEEE_F32LE; } };
        template<> struct datatype_helper<double> { static auto value() { return H5::PredType::IEEE_F64LE; } };

        template<typename T>
        auto datatype() {
            return detail::datatype_helper<T>::value();
        }
    }

    enum class hdf5_stack_header_t {
        none,
        matlab
    };

    struct hdf5_stack_config_t {
        std::string path;

        std::array<size_t, 4> shape = { {0, 0, 0, 1} };

        auto& channels_per_sample () { return shape[3]; }
        const auto& channels_per_sample() const { return shape[3]; }
        auto& samples_per_ascan() { return shape[2]; }
        const auto& samples_per_ascan() const { return shape[2]; }
        auto& ascans_per_bscan() { return shape[1]; }
        const auto& ascans_per_bscan() const { return shape[1]; }
        auto& bscans_per_volume() { return shape[0]; }
        const auto& bscans_per_volume() const { return shape[0]; }

        auto volume_shape() const {
            return shape;
        }
        std::array<size_t, 3> bscan_shape() const {
            return { ascans_per_bscan(), samples_per_ascan(), channels_per_sample() };
        }
        std::array<size_t, 2> ascan_shape() const {
            return { samples_per_ascan(), channels_per_sample() };
        }

        hdf5_stack_header_t header = hdf5_stack_header_t::none;
        int compression_level = 0;
    };

    template<typename element_t_, typename config_t_>
    class hdf5_stack_t {
    public:

        using config_t = config_t_;
        using element_t = element_t_;

        constexpr static auto matlab_userblock_size = 512;

        hdf5_stack_t(std::shared_ptr<spdlog::logger> log = nullptr)
            : _log(std::move(log)) { }

        virtual ~hdf5_stack_t() {
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

            {
                H5::FileCreatPropList prop;

                // anticipate large volume sequences
                prop.setSizes(sizeof(size_t), sizeof(size_t));

                if (_config.header == hdf5_stack_header_t::matlab) {
                    // reserve the userblock now for generation on close
                    prop.setUserblock(matlab_userblock_size);
                }

                // open new file
                if (_log) { _log->debug("opening file \"{}\"", _config.path); }
                _file = H5::H5File(_config.path, H5F_ACC_TRUNC, prop);

                if (_config.header == hdf5_stack_header_t::none) {
                    // no header
                } else if (_config.header == hdf5_stack_header_t::matlab) {
                    // close from HDF5
                    _file.close();

                    {
                        // write the MATLAB preamble
                        std::fstream f(_config.path, std::ios::binary | std::ios::in | std::ios::out);
                        detail::matlab::write_preamble(f, 0x0200);
                    }

                    // reopen the file for HDF5
                    _file = H5::H5File(_config.path, H5F_ACC_RDWR, prop);
                } else {
                    raise(_log, "unsupported header format: {}", cast(_config.header));
                }
            }

            H5::DataSpace dataspace;
            {
                // compute dimensions
                std::vector<hsize_t> max_dims;
                max_dims.push_back(H5F_UNLIMITED);
                std::copy(_config.shape.begin(), _config.shape.end(), std::back_inserter(max_dims));
                std::vector<hsize_t> initial_dims(max_dims.size(), 0);

                // create an expandable dataspace
                dataspace = H5::DataSpace(downcast<int>(initial_dims.size()), initial_dims.data(), max_dims.data());
            }

            {
                H5::DSetCreatPropList prop;

                // set chunk size to a single A-scan
                // NOTE: performance can become very poor if the chunk size is a B-scan and partial A-scans are written
                std::vector<hsize_t> chunk_shape{ 1, 1, 1, _config.samples_per_ascan(), _config.channels_per_sample() };
                prop.setChunk(downcast<int>(chunk_shape.size()), chunk_shape.data());

                // filling parameters
                auto fill = element_t();
                prop.setFillValue(detail::datatype<element_t>(), &fill);
                prop.setFillTime(H5D_FILL_TIME_ALLOC);

                // compressions
                prop.setDeflate(_config.compression_level);

                // create the dataset
                _dataset = _file.createDataSet("data", detail::datatype<element_t>(), dataspace, prop);

                if (_config.header == hdf5_stack_header_t::matlab) {
                    // add the class type for MATLAB
                    H5::StrType type(0, H5T_VARIABLE);
                    auto attr = _dataset.createAttribute("MATLAB_class", type, H5::DataSpace(H5S_SCALAR));
                    attr.write(type, std::string(detail::matlab::array<element_t>::name));
                }
            }

            _volume_index = _volume_allocate = 0;
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
            } catch (const incompatible_shape&) {
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

            // resize dataset if needed
            _resize();

            // destination
            auto dst = _dataset.getSpace();
            {
                std::vector<hsize_t> dims{ 1, 1, chunk.shape(0), chunk.shape(1), chunk.shape(2) };
                std::vector<hsize_t> start{ _volume_index, bscan_index, ascan_index, 0, 0 };
                dst.selectHyperslab(H5S_SELECT_SET, dims.data(), start.data());

                // source
                auto src = H5::DataSpace(downcast<int>(dims.size()), dims.data());

                // write out the B-scans
                _dataset.write(chunk.data(), detail::datatype<element_t>(), src, dst);
            }
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
            } catch (const incompatible_shape&) {
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

            // resize dataset if needed
            _resize();

            // destination
            auto dst = _dataset.getSpace();
            {
                std::vector<hsize_t> dims{ 1, chunk.shape(0), chunk.shape(1), chunk.shape(2), chunk.shape(3) };
                std::vector<hsize_t> start{ _volume_index, index, 0, 0, 0 };
                dst.selectHyperslab(H5S_SELECT_SET, dims.data(), start.data());

                // source
                auto src = H5::DataSpace(downcast<int>(dims.size()), dims.data());

                // write out the B-scans
                _dataset.write(chunk.data(), detail::datatype<element_t>(), src, dst);
            }
        }

        template<typename V>
        void write_volume(const cpu_viewable<V>& volume) {
            write_multi_bscan(0, volume);
        }

        virtual void advance_volume(bool allocate = true) {
            _volume_index++;
            if (allocate) {
                _resize();
            }
        }

        virtual void close() {
            if (ready()) {
                if (_log) { _log->debug("closing file \"{}\"", _config.path); }
                _dataset.close();
                _file.close();
                _file = {};
            }
        }

        virtual bool ready() const {
            return _file.getId() != H5I_INVALID_HID;
        }

    protected:

        void _resize() {
            if (_volume_index >= _volume_allocate) {
                if (_log) { _log->trace("allocating {} volume(s) on disk", _volume_index - _volume_allocate + 1); }

                _volume_allocate = _volume_index + 1;

                std::vector<hsize_t> dims;
                dims.push_back(_volume_allocate);
                std::copy(_config.shape.begin(), _config.shape.end(), std::back_inserter(dims));
                _dataset.extend(dims.data());
            }
        }

        size_t _volume_index, _volume_allocate;
        std::shared_ptr<spdlog::logger> _log;

        config_t _config;

        H5::H5File _file;
        H5::DataSet _dataset;
    };

}
