/** \rst

    shared configuration options for digital storage oscilloscope components

    DSO output consists of blocks of records of samples of channels.  The output
    is thus 3D: #records x #samples x #channels.  The data is in row-major format,
    meaning the record index varies the slowest and the channel index varies the
    fastest.

 \endrst */

#pragma once

#include <array>

#include <vortex/util/cast.hpp>

namespace vortex::acquire {

    struct dso_config_t {

        virtual ~dso_config_t() {}

        virtual std::array<size_t, 3> shape() const { return { records_per_block(), samples_per_record(), channels_per_sample() }; };
        virtual std::array<ptrdiff_t, 3> stride() const { return { downcast<ptrdiff_t>(samples_per_record() * channels_per_sample()), downcast<ptrdiff_t>(channels_per_sample()), 1 }; };

        virtual size_t channels_per_sample() const { return 1; }

        size_t& samples_per_record() { return _samples_per_record; }
        const size_t& samples_per_record() const { return _samples_per_record; }
        size_t& records_per_block() { return _records_per_block; }
        const size_t& records_per_block() const { return _records_per_block; }

    protected:

        size_t _samples_per_record = 1024;
        size_t _records_per_block = 1000;

    };

}
