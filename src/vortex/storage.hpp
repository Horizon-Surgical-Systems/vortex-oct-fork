#pragma once

#include <vortex/storage/marker.hpp>
#include <vortex/storage/dump.hpp>
#include <vortex/storage/simple.hpp>
#include <vortex/storage/broct.hpp>

namespace vortex {

    using stream_dump_t = storage::stream_dump_t<storage::stream_dump_config_t>;

    using marker_log_t = storage::marker_log_t<storage::marker_log_config_t>;

    template<typename element_t>
    using simple_stream_t = storage::simple_stream_t<element_t, storage::simple_stream_config_t>;
    
    template<typename element_t>
    using simple_stack_t = storage::simple_stack_t<element_t, storage::simple_stack_config_t>;

    using broct_storage_t = storage::broct_storage_t<storage::broct_storage_config_t>;

}

#if defined(VORTEX_ENABLE_HDF5)

#include <vortex/storage/hdf5.hpp>

namespace vortex {

    template<typename element_t>
    using hdf5_stack_t = storage::hdf5_stack_t<element_t, storage::hdf5_stack_config_t>;

}

#endif