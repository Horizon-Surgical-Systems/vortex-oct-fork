#pragma once

#include <string>
#include <vector>

#include <AlazarApi.h>

#include <vortex/driver/alazar/board.hpp>

namespace vortex::alazar::detail {

    struct impedance_input_range_t {
        size_t impedance_ohms;
        std::vector<size_t> input_range_millivolts;
    };

    struct alignment_info_t {
        U32 min_record_size, pretrigger_alignment, resolution;
    };


#define _LOOKUP_DECLARE(name, in_t, out_t) \
    out_t name(in_t val); \
    out_t name(in_t val, out_t fallback);

    _LOOKUP_DECLARE(lookup_board_kind, U32, const std::string&);
    _LOOKUP_DECLARE(lookup_sampling_rate, size_t, U32);
    _LOOKUP_DECLARE(lookup_supported_sampling_rate, const std::string&, const std::vector<size_t>&);
    _LOOKUP_DECLARE(lookup_impedance_ohms, size_t, U32);
    _LOOKUP_DECLARE(lookup_input_range_millivolts, size_t, U32);
    _LOOKUP_DECLARE(lookup_supported_impedance_ohms_input_range_millivolts, const std::string&, const std::vector<impedance_input_range_t>&);
    //_LOOKUP_DECLARE(lookup_packing_mode, (bits)", size_t, U32);
    _LOOKUP_DECLARE(lookup_trigger_range_volts, size_t, U32);
    _LOOKUP_DECLARE(lookup_alignment, std::string, const alignment_info_t&);
    _LOOKUP_DECLARE(lookup_features, std::string, const board_t::info_t::features_t&);
    _LOOKUP_DECLARE(lookup_error_message, RETURN_CODE, const char*);

#undef _LOOKUP_DECLARE

}
