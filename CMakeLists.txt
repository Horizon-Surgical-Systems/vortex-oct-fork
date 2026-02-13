cmake_minimum_required(VERSION 3.17.0)

# configure options with awareness of VCPKG_MANIFEST_FEATURES
# NOTE: options may be overridden by directly setting cache variables
include(cmake/vcpkg_utils.cmake)

vcpkg_option(WITH_ASIO       asio       on  "Build features dependent on asio")
vcpkg_option(WITH_BACKWARD   backward   on  "Build features dependent on Backward")
vcpkg_option(WITH_CUDA       cuda       on  "Build features dependent on CUDA")
vcpkg_option(WITH_REFLEXXES  reflexxes  on  "Build features dependent on Reflexxes")
vcpkg_option(WITH_FFTW       fftw       on  "Build features dependent on FFTW")
vcpkg_option(WITH_HDF5       hdf5       on  "Build features dependent on HDF5")
vcpkg_option(WITH_PYTHON     python     off "Build Python bindings")

# hardware options
vcpkg_option(WITH_ALAZAR     alazar     off "Build support for Alazar ATS-SDK")
vcpkg_option(WITH_ALAZAR_GPU alazar-gpu off "Build support for Alazar ATS-GPU")
vcpkg_option(WITH_TELEDYNE   teledyne   off "Build support for Teledyne ADQ")
vcpkg_option(WITH_DAQMX      daqmx      off "Build support for NI DAQmx")
vcpkg_option(WITH_ALAZAR_DAC alazar-dac off "Build support for Alazar DAC module")
vcpkg_option(WITH_IMAQ       imaq       off "Build support for NI IMAQdx")

# Python wheel options
option(BUILD_PYTHON_WHEEL "Build Python wheel for installation" on)
option(INSTALL_PYTHON_WHEEL "Automatically install Python wheel after build" off)
option(ENABLE_DEPENDENCY_PACKAGING "Include binary dependencies in Python wheel" on)
option(ENABLE_OUT_OF_TREE_PACKAGING "Include binary dependencies from outside the build tree in Python wheel" on)

# debugging options
option(ENABLE_CUDA_KERNEL_SERIALIZATION "Synchronize the CUDA device after every kernel launch (turn on for debugging)" off)
option(ENABLE_EXCEPTION_GUARDS "Handle exceptions (turn off for debugging)" on)
option(ENABLE_PYBIND11_OPTIMIZATIONS "Allow pybind11 to optimize modules with LTO and by stripping symbols (turn off for debugging, automatic in debug mode)" on)

# miscellaneous options
option(ENABLE_MODULAR_BUILD "Build hardware modules in separate libraries" ${WITH_PYTHON})
option(ENABLE_DEMOS "Build demos" off)

report_vcpkg_options()

project(vortex)

set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_DEBUG_POSTFIX "d")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE STRING "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE STRING "")

if (WITH_CUDA)
    enable_language(CUDA)
endif()

# version detection
if(WIN32)
    include(cmake/windows_version.cmake)
endif()
include(cmake/vortex_version.cmake)

if (CMAKE_CXX_BYTE_ORDER STREQUAL BIG_ENDIAN)
    add_definitions(-DVORTEX_ENDIAN_BIG)
elseif (CMAKE_CXX_BYTE_ORDER STREQUAL LITTLE_ENDIAN)
    add_definitions(-DVORTEX_ENDIAN_LITTLE)
else()
    message(SEND_ERROR "Unknown endianness value: ${CMAKE_CXX_BYTE_ORDER}")
endif()

# IDE configuration
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY AUTOGEN_TARGETS_FOLDER AutoGen)
file(GLOB_RECURSE FILES *.hpp *.cpp *.yaml *.py *.cu *.cuh)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${FILES})

add_subdirectory(src)
if(ENABLE_DEMOS)
    add_subdirectory(demo)
endif()

# generate package files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    vortex-version.cmake
    VERSION "${VORTEX_VERSION_MAJOR}.${VORTEX_VERSION_MINOR}.${VORTEX_VERSION_PATCH}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    vortex-config.cmake.in vortex-config.cmake
    INSTALL_DESTINATION share/vortex
)

install(
    EXPORT vortex-targets
    FILE vortex-targets.cmake
    NAMESPACE vortex::
    DESTINATION share/vortex
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/vortex-config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/vortex-version.cmake
    DESTINATION share/vortex
)
