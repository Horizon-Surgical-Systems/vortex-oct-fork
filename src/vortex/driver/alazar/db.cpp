#include <vortex/driver/alazar/db.hpp>

#include <unordered_map>

#include <AlazarCmd.h>

#include <fmt/format.h>

#include <vortex/driver/alazar/core.hpp>
#include <vortex/util/exception.hpp>

using namespace vortex::alazar;
using namespace vortex::alazar::detail;

static const std::unordered_map<U32, std::string> _board_kind_map = {
    { 0,  "NONE" },
    { 1,  "ATS850" },
    { 2,  "ATS310" },
    { 3,  "ATS330" },
    { 4,  "ATS855" },
    { 5,  "ATS315" },
    { 6,  "ATS335" },
    { 7,  "ATS460" },
    { 8,  "ATS860" },
    { 9,  "ATS660" },
    { 10, "ATS665" },
    { 11, "ATS9462" },
    { 12, "ATS9434" },
    { 13, "ATS9870" },
    { 14, "ATS9350" },
    { 15, "ATS9325" },
    { 16, "ATS9440" },
    { 17, "ATS9410" },
    { 18, "ATS9351" },
    { 19, "ATS9310" },
    { 20, "ATS9461" },
    { 21, "ATS9850" },
    { 22, "ATS9625" },
    { 23, "ATG6500" },
    { 24, "ATS9626" },
    { 25, "ATS9360" },
    { 26, "AXI9870" },
    { 27, "ATS9370" },
    { 28, "ATU7825" },
    { 29, "ATS9373" },
    { 30, "ATS9416" },
    { 31, "ATS9637" },
    { 32, "ATS9120" },
    { 33, "ATS9371" },
    { 34, "ATS9130" },
    { 35, "ATS9352" },
    { 36, "ATS9453" },
    { 37, "ATS9146" },
    { 38, "ATS9000" },
    { 39, "ATST371" },
    { 40, "ATS9437" },
    { 41, "ATS9618" },
    { 42, "ATS9358" },
    { 44, "ATS9353" },
    { 45, "ATS9872" },
    { 46, "ATS9470" },
    { 47, "ATS9628" },
    { 48, "ATS9874" },
    { 49, "ATS9473" },
    { 50, "ATS9280" },
    { 51, "ATS4001" },
    { 52, "ATS9182" },
    { 53, "ATS9364" },
    { 54, "ATS9442" },
    { 55, "ATS9376" },
    { 56, "ATS9380" },
    { 57, "ATS9428" },
};

static const std::unordered_map<size_t, U32> _sampling_rate_map = {
    { 1000,        SAMPLE_RATE_1KSPS },
    { 2000,        SAMPLE_RATE_2KSPS },
    { 5000,        SAMPLE_RATE_5KSPS },
    { 10000,       SAMPLE_RATE_10KSPS },
    { 20000,       SAMPLE_RATE_20KSPS },
    { 50000,       SAMPLE_RATE_50KSPS },
    { 100000,      SAMPLE_RATE_100KSPS },
    { 200000,      SAMPLE_RATE_200KSPS },
    { 500000,      SAMPLE_RATE_500KSPS },
    { 1000000,     SAMPLE_RATE_1MSPS },
    { 2000000,     SAMPLE_RATE_2MSPS },
    { 5000000,     SAMPLE_RATE_5MSPS },
    { 10000000,    SAMPLE_RATE_10MSPS },
    { 20000000,    SAMPLE_RATE_20MSPS },
    { 25000000,    SAMPLE_RATE_25MSPS },
    { 50000000,    SAMPLE_RATE_50MSPS },
    { 100000000,   SAMPLE_RATE_100MSPS },
    { 125000000,   SAMPLE_RATE_125MSPS },
    { 160000000,   SAMPLE_RATE_160MSPS },
    { 180000000,   SAMPLE_RATE_180MSPS },
    { 200000000,   SAMPLE_RATE_200MSPS },
    { 250000000,   SAMPLE_RATE_250MSPS },
#if ATSAPI_VERSION >= 70500 || ATSGPU_VERSION >= 40001
    { 350000000,   SAMPLE_RATE_350MSPS },
    { 370000000,   SAMPLE_RATE_370MSPS },
#endif
    { 400000000,   SAMPLE_RATE_400MSPS },
    { 500000000,   SAMPLE_RATE_500MSPS },
    { 800000000,   SAMPLE_RATE_800MSPS },
    { 1000000000,  SAMPLE_RATE_1000MSPS },
    { 1200000000,  SAMPLE_RATE_1200MSPS },
    { 1500000000,  SAMPLE_RATE_1500MSPS },
    { 1600000000,  SAMPLE_RATE_1600MSPS },
    { 1800000000,  SAMPLE_RATE_1800MSPS },
    { 2000000000,  SAMPLE_RATE_2000MSPS },
    { 2400000000,  SAMPLE_RATE_2400MSPS },
    { 3000000000,  SAMPLE_RATE_3000MSPS },
    { 3600000000,  SAMPLE_RATE_3600MSPS },
    { 4000000000,  SAMPLE_RATE_4000MSPS },
#if ATSAPI_VERSION >= 70500
    { 5000000000,  SAMPLE_RATE_5000MSPS },
    { 10000000000, SAMPLE_RATE_10000MSPS },
#endif
};

static const std::unordered_map<std::string, std::vector<size_t>> _supported_sampling_rate_map = {
    { "ATS310",  { 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000 } },
    { "ATS9120", { 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000 } },
    { "ATS330",  { 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000 } },
    { "ATS9130", { 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000 } },
    { "ATS460",  { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000 } },
    { "ATS660",  { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000 } },
    { "ATS9146", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000 } },
    { "ATS850",  { 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000 } },
    { "ATS860",  { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9182", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 25000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9350", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000, 500000000 } },
    { "ATS9351", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000, 500000000 } },
    { "ATS9360", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 800000000, 1000000000, 1200000000, 1500000000, 1800000000 } },
    { "ATS9371", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 800000000, 1000000000 } },
    { "ATS9364", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 800000000, 1000000000 } },
    { "ATS9373", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000,  100000000,  200000000,  500000000,  800000000, 1000000000, 1200000000, 1500000000, 2000000000, 2400000000, 3000000000, 3600000000, 4000000000 } },
    { "ATS9416", { 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000 } },
    { "ATS9440", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000 } },
    { "ATS9462", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 160000000, 180000000 } },
    { "ATS9625", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9626", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9628", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9428", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000 } },
    { "ATS9870", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 250000000, 500000000, 1000000000 } },
    { "AXI9870", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 250000000, 500000000, 1000000000 } },
    { "ATS9872", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 250000000, 500000000, 1000000000 } },
    { "ATS9874", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 250000000, 500000000, 1000000000 } },
    { "ATS9352", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000, 500000000 } },
    { "ATS9353", { 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 125000000, 250000000, 500000000 } },
};

static const std::unordered_map<size_t, U32> _impedance_ohms_map = {
    { 1000000,  IMPEDANCE_1M_OHM},
    { 50,       IMPEDANCE_50_OHM},
    { 75,       IMPEDANCE_75_OHM},
    { 300,      IMPEDANCE_300_OHM},
};
static const std::unordered_map<size_t, U32> _input_range_millivolts_map = {
    { 20,     INPUT_RANGE_PM_20_MV },
    { 40,     INPUT_RANGE_PM_40_MV },
    { 50,     INPUT_RANGE_PM_50_MV },
    { 80,     INPUT_RANGE_PM_80_MV },
    { 100,    INPUT_RANGE_PM_100_MV },
    { 200,    INPUT_RANGE_PM_200_MV },
    { 400,    INPUT_RANGE_PM_400_MV },
    { 500,    INPUT_RANGE_PM_500_MV },
    { 800,    INPUT_RANGE_PM_800_MV },
    { 1000,   INPUT_RANGE_PM_1_V },
    { 2000,   INPUT_RANGE_PM_2_V },
    { 4000,   INPUT_RANGE_PM_4_V },
    { 5000,   INPUT_RANGE_PM_5_V },
    { 8000,   INPUT_RANGE_PM_8_V },
    { 10000,  INPUT_RANGE_PM_10_V },
    { 20000,  INPUT_RANGE_PM_20_V },
    { 40000,  INPUT_RANGE_PM_40_V },
    { 16000,  INPUT_RANGE_PM_16_V },
    { 1250,   INPUT_RANGE_PM_1_V_25 },
    { 2500,   INPUT_RANGE_PM_2_V_5 },
    { 125,    INPUT_RANGE_PM_125_MV },
    { 250,    INPUT_RANGE_PM_250_MV },
};

static const std::unordered_map<std::string, std::vector<impedance_input_range_t>> _supported_impedance_ohms_input_range_millivolts_map = {
    { "ATS310",  { { 50, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },
    { "ATS330",  { { 50, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },
    { "ATS9120", { { 50, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },
    { "ATS9130", { { 50, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },

    { "ATS460",  { { 50, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },
    { "ATS9146", { { 50, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },
    { "ATS9182", { { 50, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },

    { "ATS660",  { { 50, { 200, 400, 800, 2000, 4000 } },
                   { 1000000, { 200, 400, 800, 2000, 4000, 8000, 16000 } } } },
    { "ATS9462", { { 50, { 200, 400, 800, 2000, 4000 } },
                   { 1000000, { 200, 400, 800, 2000, 4000, 8000, 16000 } } } },

    { "ATS850",  { { 50, { 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },

    { "ATS860",  { { 50, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000 } },
                   { 1000000, { 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000, 10000 } } } },

    { "ATS9325", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "ATS9350", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "ATS9850", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "ATS9870", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "ATS9872", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "AXI9870", { { 50, { 40, 100, 200, 400, 1000, 2000, 4000 } } } },
    { "ATS9352", { { 50, { 100, 200, 400, 1000, 2000, 4000 } } } },

    { "ATS9351", { { 50, { 400 } } } },
    { "ATS9353", { { 50, { 400 } } } },
    { "ATS9360", { { 50, { 400 } } } },
    { "ATS9370", { { 50, { 400 } } } },
    { "ATS9371", { { 50, { 400 } } } },
    { "ATS9373", { { 50, { 400 } } } },
    { "ATS9364", { { 50, { 400 } } } },

    { "ATS9625", { { 50, { 1250 } } } },
    { "ATS9626", { { 50, { 1250 } } } },
    { "ATS9628", { { 50, { 1250 } } } },
    { "ATS9428", { { 50, { 1250 } } } },

    { "ATS9440", { { 50, { 100, 200, 400, 1000, 2000, 4000 } } } },

    { "ATS9416", { { 50, { 1000 } } } },

    { "ATS9874", { { 50, { 500 } } } },
};

static const std::unordered_map<size_t, U32> _packing_mode_map = {
    {16,  PACK_DEFAULT},
    {8,   PACK_8_BITS_PER_SAMPLE},
    {12,  PACK_12_BITS_PER_SAMPLE},
};

static const std::unordered_map<size_t, U32> _trigger_range_volts_map = {
    {5000, ETR_5V},
    {1000, ETR_1V},
    {2500, ETR_2V5},
    {0,    ETR_TTL},
};

static const std::unordered_map<std::string, alignment_info_t> _alignment_map = {
    { "ATS310",  { 256,   4,  16 }},
    { "ATS330",  { 256,   4,  16 }},
    { "ATS460",  { 128,  16,  16 }},
    { "ATS660",  { 128,  16,  16 }},
    { "ATS850",  { 256,   4,  16 }},
    { "ATS860",  { 256,  32,  32 }},
    { "ATS9350", { 256,  32,  32 }},
    { "ATS9351", { 256,  32,  32 }},
    { "ATS9120", { 256,  32,  32 }},
    { "ATS9130", { 256,  32,  32 }},
    { "ATS9146", { 256,  32,  32 }},
    { "ATS9182", { 512,  64,  64 }},
    { "ATS9360", { 256, 128, 128 }},
    { "ATS9370", { 256, 128, 128 }},
    { "ATS9371", { 256, 128, 128 }},
    { "ATS9373", { 256, 128, 128 }},
    { "ATS9416", { 256, 128, 128 }},
    { "ATS9440", { 256,  32,  32 }},
    { "ATS9462", { 256,  32,  32 }},
    { "ATS9625", { 256,  32,  32 }},
    { "ATS9626", { 256,  32,  32 }},
    { "ATS9870", { 256,  64,  64 }},
    { "AXI9870", { 256,  64,  64 }},
    { "ATS9352", { 256,  32,  32 }},
    { "ATS9353", { 256,  32,  32 }},
    { "ATS9872", { 256,  64,  64 }},
    { "ATS9874", { 256,  64,  64 }},
    { "ATS9628", { 256,  32,  32 }},
    { "ATS9428", { 256, 128, 128 }},
    { "ATS9364", { 256, 128, 128 }},
};

static const std::unordered_map<std::string, board_t::info_t::features_t> _features_map = {
    { "ATS460",  { true,  {}          , false } },
    { "ATS660",  { true,  {}          , false } },
    { "ATS860",  { true,  {}          , false } },
    { "ATS9350", { true,  {}          , false } },
    { "ATS9351", { true,  {}          , false } },
    { "ATS9352", { true,  {}          , false } },
    { "ATS9353", { true,  {}          , false } },
    { "ATS9360", { false, { 50000000 }, false, true } },
    { "ATS9364", { false, { 50000000 }, true,  true } },
    { "ATS9371", { false, { 50000000 }, true,  true } },
    { "ATS9373", { false, { 50000000 }, true,  true } },
    { "ATS9440", { true,  {}          , false, true } },
    { "ATS9462", { true,  {}          , false } },
    { "ATS9625", { true,  {}          , false } },
    { "ATS9626", { true,  {}          , false } },
    { "ATS9870", { true,  {}          , false } },
    { "ATS9872", { true,  {}          , false } },
    { "ATS9874", { true,  {}          , false } },
};

static const std::unordered_map<RETURN_CODE, const char*> _error_message_map = {
    { ApiSuccess, "The operation completed without error" },
    { ApiFailed, "The operation failed." },
    { ApiAccessDenied, "Access denied." },
    { ApiDmaChannelUnavailable, "Channel selection is unavailable." },
    { ApiDmaChannelInvalid, "Channel selection in invalid." },
    { ApiDmaChannelTypeError, "Channel selection is invalid." },
    { ApiDmaInProgress, "A data transfer is in progress. This error code indicates that the current action cannot be performed while an acquisition is in progress. It also returned by AlazarPostAsyncBuffer() if this function is called with an invalid DMA buffer." },
    { ApiDmaDone, "DMA transfer is finished." },
    { ApiDmaPaused, "DMA transfer was paused." },
    { ApiDmaNotPaused, "DMA transfer is not paused." },
    { ApiDmaCommandInvalid, "A DMA command is invalid." },
    { ApiNullParam, "One of the parameters of the function is NULL and should not be." },
    { ApiUnsupportedFunction, "This function is not supported by the API. Consult the manual for more information." },
    { ApiInvalidPciSpace, "Invalid PCI space." },
    { ApiInvalidIopSpace, "Invalid IOP space." },
    { ApiInvalidSize, "Invalid size passed as argument to the function." },
    { ApiInvalidAddress, "Invalid address." },
    { ApiInvalidAccessType, "Invalid access type requested." },
    { ApiInvalidIndex, "Invalid index." },
    { ApiInvalidRegister, "Invalid register." },
    { ApiConfigAccessFailed, "Access for configuration failed." },
    { ApiInvalidDeviceInfo, "Invalid device information." },
    { ApiNoActiveDriver, "No active driver for the board. Please ensure that a driver is installed." },
    { ApiInsufficientResources, "There were not enough system resources to complete this operation. The most common reason of this return code is using too many DMA buffers, or using DMA buffers that are too big. Please try reducing the number of buffers posted to the board at any time, and/or try reducing the DMA buffer sizes." },
    { ApiNotInitialized, "The API has not been properly initialized for this function call. Please review one of the code samples from the ATS-SDK to confirm that API calls are made in the right order." },
    { ApiInvalidPowerState, "Power state requested is not valid." },
    { ApiPowerDown, "The operation cannot be completed because the device is powered down. For example, this error code is output if the computer enters hibernation while an acquisition is running." },
    { ApiNotSupportThisChannel, "The API call is not valid with this channel selection." },
    { ApiNoAction, "The function has requested no action to be taken." },
    { ApiHSNotSupported, "HotSwap is not supported." },
    { ApiVpdNotEnabled, "Vital product data not enabled." },
    { ApiInvalidOffset, "Offset argument is not valid." },
    { ApiPciTimeout, "Timeout on the PCI bus." },
    { ApiInvalidHandle, "Invalid handle passed as argument." },
    { ApiBufferNotReady, "The buffer passed as argument is not ready to be called with this API. This error code is most often seen is the order of buffers posted to the board is not respected when querying them." },
    { ApiInvalidData, "Generic invalid parameter error. Check the function's documentation for more information about valid argument values." },
    { ApiDoNothing, "" },
    { ApiDmaSglBuildFailed, "Unable to lock buffer and build SGL list." },
    { ApiPMNotSupported, "Power management is not supported." },
    { ApiInvalidDriverVersion, "Invalid driver version." },
    { ApiWaitTimeout, "The operation did not finish during the timeout interval. Try the operation again, or abort the acquisition." },
    { ApiWaitCanceled, "The operation was cancelled." },
    { ApiBufferTooSmall, "The buffer used is too small. Try increasing the buffer size." },
    { ApiBufferOverflow, "The board overflowed its internal (on-board) memory. Try reducing the sample rate, reducing the number of enabled channels. Also ensure that DMA buffer size is between 1 MB and 8 MB." },
    { ApiInvalidBuffer, "The buffer passed as argument is not valid." },
    { ApiInvalidRecordsPerBuffer, "The number of records per buffer passed as argument is invalid."},
    { ApiDmaPending, "An asynchronous I/O operation was successfully started on the board. It will be completed when sufficient trigger events are supplied to the board to fill the buffer." },
    { ApiLockAndProbePagesFailed, "The buffer is too large for the driver or operating system to prepare for scatter-gather DMA transfer. Try reducing the size of each buffer, or reducing the number of buffers queued by the application." },
    { ApiTransferComplete, "This buffer is the last in the current acquisition." },
    { ApiPllNotLocked, "The on-board PLL circuit could not lock. If the acquisition used an internal sample clock, this might be a symptom of a hardware problem; contact AlazarTech. If the acquisition used an external 10 MHz PLL signal, please make sure that the signal is fed in properly." },
    { ApiNotSupportedInDualChannelMode, "The requested acquisition is not possible with two channels. This can be due to the sample rate being too fast for DES boards, or to the number of samples per record being too large. Try reducing the number of samples per channel, or switching to single channel mode." },
    { ApiNotSupportedInQuadChannelMode, "The requested acquisition is not possible with four channels. This can be due to the sample rate being too fast for DES boards, or to the number of samples per record being too large. Try reducing the number of samples per channel, or switching to single channel mode." },
    { ApiFileIoError, "A file read or write error occurred." },
    { ApiInvalidClockFrequency, "The requested ADC clock frequency is not supported." },
    { ApiInvalidSkipTable, "Invalid skip table passed as argument." },
    { ApiInvalidDspModule, "This DSP module is not valid for the current operation." },
    { ApiDESOnlySupportedInSingleChannelMode, "Dual-edge sampling mode is only supported in signal-channel mode. Try disabling dual-edge sampling (lowering the sample rate if using internal clock), or selecting only one channel." },
    { ApiInconsistentChannel, "Successive API calls of the same acquisition have received inconsistent acquisition channel masks." },
    { ApiDspFiniteRecordsPerAcquisition, "DSP acquisition was run with a finite number of records per acquisition. Set this value to infinite." },
    { ApiNotEnoughNptFooters, "Not enough NPT footers in the buffer for extraction." },
    { ApiInvalidNptFooter, "Invalid NPT footer found." },
    { ApiOCTIgnoreBadClockNotSupported, "OCT ignore bad clock is not supported." },
    { ApiError1, "The requested number of records in a single-port acquisition exceeds the maximum supported by the digitizer. Use dual-ported AutoDMA to acquire more records per acquisition." },
    { ApiError2, "The requested number of records in a single-port acquisition exceeds the maximum supported by the digitizer." },
    { ApiOCTNoTriggerDetected, "No trigger is detected as part of the OCT ignore bad clock feature." },
    { ApiOCTTriggerTooFast, "Trigger detected is too fast for the OCT ignore bad clock feature." },
    { ApiNetworkError, "There was a network-related issue. Make sure that the network connection and settings are correct." },
#if ATSAPI_VERSION >= 70500 || ATSGPU_VERSION >= 40001
    { ApiFftSizeTooLarge, "On-FPGA FFT cannot support FFT that large. Try reducing the FFT size, or querying the maximum FFT size with AlazarDSPGetInfo()." },
    { ApiGPUError, "GPU returned an error. See log for more information." },
#endif
#if ATSAPI_VERSION >= 70500
    { ApiAcquisitionModeOnlySupportedInFifoStreaming, "This board only supports this acquisition mode in FIFO only streaming mode. Please set the ADMA_FIFO_ONLY_STREAMING flag in AlazarBeforeAsyncRead()." },
    { ApiInterleaveNotSupportedInTraditionalMode, "This board does not support sample interleaving in traditional acquisition mode. Please refer to the SDK guide for more information." },
    { ApiRecordHeadersNotSupported, "This board does not support record headers. Please refer to the SDK guide for more information." },
    { ApiRecordFootersNotSupported, "This board does not support record footers. Please refer to the SDK guide for more information." },
    { ApiFastBufferLockCountExceeded, "The number of different DMA buffers posted exceeds the limit set with AlazarConfigureFastBufferLock(). Either disable fast buffer locking, or confirm that the value passed to AlazarConfigureFastBufferLock() is respected." },
    { ApiInvalidStateDoRetry, "The operation could not complete because the system is in an invalid state. You may safely retry the call that returned this error." },
    { ApiInvalidInputRange, "The operation could not complete because the system is in an invalid state. You may safely retry the call that returned this error." },
#endif
};

#define _LOOKUP_DEFINE(name, map, display, in_t, out_t) \
    out_t vortex::alazar::detail::name(in_t val) { \
        auto it = map.find(val); \
        if(it == map.end()) { \
            throw traced<std::invalid_argument>(fmt::format("\"{}\" is not a supported value for {}", val, display)); \
        } \
        return it->second; \
    } \
    out_t vortex::alazar::detail::name(in_t val, out_t fallback) { \
        auto it = map.find(val); \
        if (it == map.end()) { \
            return fallback; \
        } \
        return it->second; \
    }

_LOOKUP_DEFINE(lookup_board_kind, _board_kind_map, "board kind", U32, const std::string&);
_LOOKUP_DEFINE(lookup_sampling_rate, _sampling_rate_map, "sampling rate", size_t, U32);
_LOOKUP_DEFINE(lookup_supported_sampling_rate, _supported_sampling_rate_map, "supported sampling rates", const std::string&, const std::vector<size_t>&);
_LOOKUP_DEFINE(lookup_impedance_ohms, _impedance_ohms_map, "impedance (Ohms)", size_t, U32);
_LOOKUP_DEFINE(lookup_input_range_millivolts, _input_range_millivolts_map, "input range (mV)", size_t, U32);
_LOOKUP_DEFINE(lookup_supported_impedance_ohms_input_range_millivolts, _supported_impedance_ohms_input_range_millivolts_map, "supported impedance (Ohms)/input range (mV) combination", const std::string&, const std::vector<impedance_input_range_t>&);
//_LOOKUP_DEFINE(lookup_packing_mode, _packing_mode_map, "packing mode (bits)", size_t, U32);
_LOOKUP_DEFINE(lookup_trigger_range_volts, _trigger_range_volts_map, "trigger range (V)", size_t, U32);
_LOOKUP_DEFINE(lookup_alignment, _alignment_map, "alignment requirements", std::string, const alignment_info_t&);
_LOOKUP_DEFINE(lookup_features, _features_map, "features requirements", std::string, const board_t::info_t::features_t&);
_LOOKUP_DEFINE(lookup_error_message, _error_message_map, "error message", RETURN_CODE, const char*);
