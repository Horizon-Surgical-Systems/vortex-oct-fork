#pragma once

#include <iostream>

#include <vortex/storage/detail/raw.hpp>
#include <vortex/util/exception.hpp>

namespace vortex::storage::detail {

    namespace numpy {
        template<typename T> struct dtype {};
        template<> struct dtype<int8_t> { constexpr static auto name = "int8"; };
        template<> struct dtype<int16_t> { constexpr static auto name = "int16"; };
        template<> struct dtype<int32_t> { constexpr static auto name = "int32"; };
        template<> struct dtype<int64_t> { constexpr static auto name = "int64"; };
        template<> struct dtype<uint8_t> { constexpr static auto name = "uint8"; };
        template<> struct dtype<uint16_t> { constexpr static auto name = "uint16"; };
        template<> struct dtype<uint32_t> { constexpr static auto name = "uint32"; };
        template<> struct dtype<uint64_t> { constexpr static auto name = "uint64"; };
        template<> struct dtype<float> { constexpr static auto name = "float32"; };
        template<> struct dtype<double> { constexpr static auto name = "float64"; };
        template<> struct dtype<std::complex<float>> { constexpr static auto name = "complex64"; };
        template<> struct dtype<std::complex<double>> { constexpr static auto name = "complex128"; };

        // magic for numpy file format version 2.0
        const auto magic = std::string("\x93NUMPY\x02\x00", 8);

        const auto alignment = 64;

        template<typename element_t, typename shape_t>
        auto write_header(std::ostream& out, const shape_t& shape, bool fortran_order = false, size_t padded_length = 256) {
            auto start = out.tellp();

            // write magic
            write_raw(out, magic);

            // build header
            auto header = fmt::format(
                "{{ "
                "'descr': '{}', "
                "'fortran_order': {}, "
                "'shape': ({}) "
                "}} ", // reserve a space to place the newline later on
                numpy::dtype<element_t>::name,
                fortran_order ? "True" : "False",
                shape_to_string(shape, ", ") + (std::size(shape) == 1 ? "," : "")
            );
            if (header.length() < padded_length) {
                header.resize(padded_length, ' ');
            }

            // add padding
            auto written = magic.length() + sizeof(uint32_t);
            auto count = downcast<size_t>(alignment * std::ceil(double(written + header.length()) / alignment));
            header.resize(count - written, ' ');
            header.back() = '\n';

            // write header
            write_raw(out, downcast<uint32_t>(header.size()));
            write_raw(out, header);

            return out.tellp() - start;
        }
    }

    namespace matlab {

        const uint32_t miINT8 = 1;
        const uint32_t miUINT8 = 2;
        const uint32_t miINT16 = 3;
        const uint32_t miUINT16 = 4;
        const uint32_t miINT32 = 5;
        const uint32_t miUINT32 = 6;
        const uint32_t miSINGLE = 7;
        const uint32_t miDOUBLE = 9;
        const uint32_t miINT64 = 12;
        const uint32_t miUINT64 = 13;
        const uint32_t miMATRIX = 14;
        const uint32_t miCOMPRESSED = 15;
        const uint32_t miUTF8 = 16;
        const uint32_t miUTF16 = 17;
        const uint32_t miUTF32 = 18;

        const uint8_t mxDOUBLE_CLASS = 6;
        const uint8_t mxSINGLE_CLASS = 7;
        const uint8_t mxINT8_CLASS = 8;
        const uint8_t mxUINT8_CLASS = 9;
        const uint8_t mxINT16_CLASS = 10;
        const uint8_t mxUINT16_CLASS = 11;
        const uint8_t mxINT32_CLASS = 12;
        const uint8_t mxUINT32_CLASS = 13;
        const uint8_t mxINT64_CLASS = 14;
        const uint8_t mxUINT64_CLASS = 15;

        const uint8_t complex_flag = 0b00001000;
        const uint8_t global_flag =  0b00000100;
        const uint8_t logical_flag = 0b00000010;

        template<typename T> struct array {};
        template<> struct array<int8_t> { constexpr static auto value = mxINT8_CLASS; constexpr static auto name = "int8"; constexpr static auto flags = 0; };
        template<> struct array<int16_t> { constexpr static auto value = mxINT16_CLASS; constexpr static auto name = "int16"; constexpr static auto flags = 0; };
        template<> struct array<int32_t> { constexpr static auto value = mxINT32_CLASS; constexpr static auto name = "int32"; constexpr static auto flags = 0; };
        template<> struct array<int64_t> { constexpr static auto value = mxINT64_CLASS; constexpr static auto name = "int64"; constexpr static auto flags = 0; };
        template<> struct array<uint8_t> { constexpr static auto value = mxUINT8_CLASS; constexpr static auto name = "uint8"; constexpr static auto flags = 0; };
        template<> struct array<uint16_t> { constexpr static auto value = mxUINT16_CLASS; constexpr static auto name = "uint16"; constexpr static auto flags = 0; };
        template<> struct array<uint32_t> { constexpr static auto value = mxUINT32_CLASS; constexpr static auto name = "uint32"; constexpr static auto flags = 0; };
        template<> struct array<uint64_t> { constexpr static auto value = mxUINT64_CLASS; constexpr static auto name = "uint64"; constexpr static auto flags = 0; };
        template<> struct array<float> { constexpr static auto value = mxSINGLE_CLASS; constexpr static auto name = "float32"; constexpr static auto flags = 0; };
        template<> struct array<double> { constexpr static auto value = mxDOUBLE_CLASS; constexpr static auto name = "float64"; constexpr static auto flags = 0; };
        template<typename T> struct array<std::complex<T>> { constexpr static auto value = array<T>::value; constexpr static auto flags = complex_flag; };

        const auto default_text = "MATLAB";
        const auto text_length = 116;

        const uint16_t endian_indicator = 0x4d49;

        const auto alignment = 8;

        struct tag {
            tag(std::ostream& out, uint32_t type, bool defer_padding = false)
                : _out(out), _defer_padding(defer_padding) {

                // write tag type
                write_raw(out, type);

                // write placeholder length
                _length_field = _out.tellp();
                write_raw<uint32_t>(_out, -1);
                _start = _out.tellp();
            }

            ~tag() {
                // calculate tag length
                auto length = _out.tellp() - _start;

                // write padding
                auto count = downcast<size_t>(alignment * std::ceil(double(length) / alignment) - length);
                if (count > 0) {
                    if (_defer_padding) {
                        extend(count);
                    } else {
                        write_raw(_out, std::string("\0", count));
                    }
                }
                auto resume = _out.tellp();

                // write length field
                _out.seekp(_length_field);
                write_raw(_out, downcast<uint32_t>(length + _extra));

                // return to original
                _out.seekp(resume);
            }

            void extend(size_t count) {
                _extra += count;
            }

        private:
            std::ostream& _out;
            bool _defer_padding;

            size_t _extra = 0;
            std::streampos _length_field, _start;
        };

        inline void write_preamble(std::ostream& out, uint16_t version, std::string text = default_text) {
            // enforce text length
            if (text.length() > text_length) {
                throw traced<std::invalid_argument>(fmt::format("text exceeded maximum length: {} > {}", text.length(), text_length));
            }
            text.resize(text_length, ' ');

            // write header
            write_raw(out, text);
            write_raw(out, std::string(8, '\x00')); // no subsystem-specific header
            write_raw(out, version);
            write_raw(out, endian_indicator);
        }

        template<typename element_t, typename shape_t>
        auto write_header(std::ostream& out, shape_t shape, std::string name = "data", std::string text = default_text, bool global = false, bool logical = false) {
            auto start = out.tellp();

            write_preamble(out, 0x0100, text);

            {
                // write data element
                tag main(out, miMATRIX, true);

                {
                    // write array flags subelement
                    tag t(out, miUINT32);

                    // build flags
                    auto flags = array<element_t>::flags | (global ? global_flag : 0) | (logical ? logical_flag : 0);

                    // write array class and flags
                    write_raw<uint32_t>(out, array<element_t>::value | (flags << 8));
                }

                // minimum number of dimensions is two
                if (shape.size() < 2) {
                    shape.push_back(1);
                }

                {
                    // write array shape subelement
                    tag t(out, miINT32);

                    // MATLAB uses column-major ordering
                    for (auto it = shape.crbegin(); it != shape.crend(); ++it) {
                        write_raw(out, downcast<uint32_t>(*it));
                    }
                }

                {
                    // write array name subelement
                    tag t(out, miINT8);

                    write_raw(out, name);
                }

                {
                    // write tag header for data
                    tag t(out, array<element_t>::value);

                    // extend
                    auto count = sizeof(element_t) * std::accumulate(std::begin(shape), std::end(shape), size_t(1), std::multiplies());
                    t.extend(count);
                    main.extend(count);
                }
            }

            return out.tellp() - start;
        }

        inline void write_final_padding(std::ofstream& out) {
            out.seekp(0, std::ios::beg);
            auto start = out.tellp();
            out.seekp(0, std::ios::end);
            auto end = out.tellp();

            auto count = downcast<size_t>(alignment * std::ceil(double(end - start) / alignment) - end);
            if (count > 0) {
                write_raw(out, std::string(count, '\x00'));
            }
        }
    }

    namespace nrrd {
        template<typename T> struct dtype {};
        template<> struct dtype<int8_t> { constexpr static auto name = "int8"; };
        template<> struct dtype<int16_t> { constexpr static auto name = "int16"; };
        template<> struct dtype<int32_t> { constexpr static auto name = "int32"; };
        template<> struct dtype<int64_t> { constexpr static auto name = "int64"; };
        template<> struct dtype<uint8_t> { constexpr static auto name = "uint8"; };
        template<> struct dtype<uint16_t> { constexpr static auto name = "uint16"; };
        template<> struct dtype<uint32_t> { constexpr static auto name = "uint32"; };
        template<> struct dtype<uint64_t> { constexpr static auto name = "uint64"; };
        template<> struct dtype<float> { constexpr static auto name = "float"; };
        template<> struct dtype<double> { constexpr static auto name = "double"; };

        // magic for nrrd file format version 5
        const auto magic = "NRRD0005";

        template<typename element_t, typename shape_t>
        auto write_header(std::ostream& out, const shape_t& shape, size_t padded_length = 256) {
            auto start = out.tellp();

            // reverse shape
            std::vector<size_t> rev_shape(std::crbegin(shape), std::crend(shape));

            // build header
            auto header = fmt::format(
                "{}\n"
                "type: {}\n"
                "dimension: {}\n"
                "sizes: {}\n"
#if defined(VORTEX_ENDIAN_LITTLE)
                "endian: little\n"
#elif defined(VORTEX_ENDIAN_BIG)
                "endian: big\n"
#endif
                "encoding: raw\n"
                "##", // reserve comments to make a blank line later on
                magic,
                nrrd::dtype<element_t>::name,
                std::size(rev_shape),
                shape_to_string(rev_shape, " ")
            );
            if (header.length() < padded_length) {
                header.resize(padded_length, '#');
            }

            // create the blank line
            std::fill(header.end() - 2, header.end(), '\n');

            // write header
            write_raw(out, header);

            return out.tellp() - start;
        }

    }

    namespace nifti {
        template<typename T> struct dtype {};
        template<> struct dtype<int8_t> { constexpr static int16_t code = 256; };
        template<> struct dtype<int16_t> { constexpr static int16_t code = 4; };
        template<> struct dtype<int32_t> { constexpr static int16_t code = 8; };
        template<> struct dtype<int64_t> { constexpr static int16_t code = 1024; };
        template<> struct dtype<uint8_t> { constexpr static int16_t code = 2; };
        template<> struct dtype<uint16_t> { constexpr static int16_t code = 512; };
        template<> struct dtype<uint32_t> { constexpr static int16_t code = 768; };
        template<> struct dtype<uint64_t> { constexpr static int16_t code = 1280; };
        template<> struct dtype<float> { constexpr static int16_t code = 16; };
        template<> struct dtype<double> { constexpr static int16_t code = 64; };
        template<> struct dtype<std::complex<float>> { constexpr static int16_t code = 32; };

        // magic for NIfTI file format version 2
        const auto magic = std::string("n+2\x00\x0d\x0a\x1a\x0a", 8);
        const auto max_dims = 7;
        const auto header_size = 540;

        template<typename element_t, typename shape_t>
        auto write_header(std::ostream& out, const shape_t& shape, size_t padded_length = 256) {
            auto start = out.tellp();

            // fixed fields
            write_raw<int32_t>(out, header_size);
            write_raw(out, magic);

            // data type
            write_raw(out, nifti::dtype<element_t>::code);
            write_raw<int16_t>(out, sizeof(element_t) * 8);

            // shape
            auto ndims = std::size(shape);
            if (ndims > max_dims) {
                throw std::runtime_error(fmt::format("shape dimensions exceed maximum supported: {} > {}", ndims, max_dims));
            }
            write_raw<int64_t>(out, ndims);
            for (size_t i = 0; i < max_dims; i++) {
                write_raw<int64_t>(out, i < ndims ? shape[i] : 0);
            }

            // empty intents
            for (size_t i = 0; i < 3; i++) {
                write_raw<double>(out, 0);
            }

            // fixed uniform pixel dimensions
            for (size_t i = 0; i < 8; i++) {
                write_raw<double>(out, 1);
            }

            // voxel offset with 4 padding bytes to align to 16 byte boundary
            write_raw<int64_t>(out, header_size + 4);

            // data scaling, calibration, ...
            write_raw<double>(out, 1);
            for (size_t i = 0; i < 5; i++) {
                write_raw<double>(out, 0);
            }

            // empty slice indices
            write_raw<int64_t>(out, 0);
            write_raw<int64_t>(out, 0);

            // empty description and auxiliary file names
            write_raw(out, std::string(80 + 24, '\0'));

            // quaternion / affine field codes
            write_raw<int>(out, 1);
            write_raw<int>(out, 1);

            // identity quaternion and translation
            for (size_t i = 0; i < 6; i++) {
                write_raw<double>(out, 0);
            }

            // identity affine matrix
            for (size_t r = 0; r < 3; r++) {
                for (size_t c = 0; c < 4; c++) {
                    write_raw<double>(out, r == c ? 1 : 0);
                }
            }

            // slice timing
            write_raw<int>(out, 0); // unknown

            // spatial and time units
            write_raw<int>(out, 0); // unknown

            // intent
            write_raw<int>(out, 0); // none
            write_raw(out, std::string(16, '\0'));

            // encoding directions
            write_raw<char>(out, 0);

            // padding bytes
            write_raw(out, std::string(15 + 4, '\0'));

            return out.tellp() - start;
        }

    }

}
