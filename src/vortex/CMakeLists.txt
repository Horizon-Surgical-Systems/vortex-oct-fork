configure_file(version.hpp.in ${PROJECT_BINARY_DIR}/vortex/version.hpp @ONLY)

add_library(vortex SHARED
    vortex.hpp

    ${PROJECT_BINARY_DIR}/vortex/version.hpp
    core.cpp core.hpp

    acquire.hpp
    acquire/null.hpp acquire/file.hpp acquire/dso.hpp

    engine.hpp
    engine/common.hpp
    engine/clock.hpp engine/dispersion.hpp engine/scan.hpp engine/adapter.hpp

    format.hpp format/common.hpp
    format/plan.hpp
    format/stack.hpp format/broct.hpp
    format/position.hpp
    format/radial.hpp format/spiral.hpp

    io.hpp
    io/null.hpp

    marker.hpp
    marker/marker.hpp marker/flags.hpp

    memory.hpp
    memory/view.hpp memory/tensor.hpp memory/cpu.hpp

    process.hpp
    process/base.hpp process/null.hpp process/copy.hpp

    scan.hpp
    scan/base.hpp
    scan/waypoints.hpp
    scan/warp.hpp
    scan/raster.hpp scan/radial.hpp scan/spiral.hpp scan/freeform.hpp

    storage.hpp
    storage/simple.hpp storage/dump.hpp storage/marker.hpp storage/broct.hpp
    storage/detail/raw.hpp storage/detail/header.hpp

    util/cast.hpp
    util/sync.cpp util/sync.hpp
    util/variant.hpp util/tuple.hpp
    util/platform.hpp util/platform.cpp
    util/stream.hpp
    util/ring.hpp
    util/copy.hpp
    util/thread.hpp
    util/exception.cpp util/exception.hpp
)
add_library(vortex::vortex ALIAS vortex)

# ref: https://stackoverflow.com/questions/25676277/cmake-target-include-directories-prints-an-error-when-i-try-to-add-the-source
get_filename_component(_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
target_include_directories(vortex PUBLIC "$<BUILD_INTERFACE:${_INCLUDE}>" "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")

#
# mandatory dependencies
#

find_package(xtensor CONFIG REQUIRED)
target_link_libraries(vortex PUBLIC xtensor xtensor::optimize xtensor::use_TBB xtensor::use_xsimd)
if(MSVC)
    target_compile_options(vortex PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /DNOMINMAX>)
endif()

find_package(TBB CONFIG REQUIRED)
target_link_libraries(vortex PUBLIC TBB::tbb)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(vortex PUBLIC fmt::fmt)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(vortex PUBLIC spdlog::spdlog)

#
# optional non-modular dependencies
#

if(WITH_ASIO)
    find_package(asio CONFIG REQUIRED)
    target_link_libraries(vortex PUBLIC asio::asio)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_ASIO)
    target_sources(vortex PRIVATE
        driver/machdsp.cpp driver/machdsp.hpp
        io/machdsp.hpp
    )
endif()

if(WITH_BACKWARD)
    find_package(Backward CONFIG REQUIRED)
    target_link_libraries(vortex PUBLIC Backward::Backward)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_BACKWARD)
endif()

if(WITH_FFTW)
    find_package(FFTW3 CONFIG REQUIRED)
    find_package(FFTW3f CONFIG REQUIRED)
    find_package(FFTW3l CONFIG REQUIRED)
    target_link_libraries(vortex PUBLIC FFTW3::fftw3 FFTW3::fftw3f FFTW3::fftw3l)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_FFTW)
    target_sources(vortex PRIVATE
        driver/fftw.hpp
        process/cpu.hpp
    )
endif()

if(WITH_REFLEXXES)
    find_package(Reflexxes CONFIG REQUIRED)
    target_link_libraries(vortex PUBLIC Reflexxes::TypeII)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_REFLEXXES)
    target_sources(vortex PRIVATE
        driver/motion.cpp driver/motion.hpp
        scan/segmented.hpp
        scan/pattern.hpp
    )
endif()

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(vortex PUBLIC CUDA::cudart CUDA::cufft)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_CUDA)
    target_sources(vortex PRIVATE
        driver/cuda/runtime.cpp driver/cuda/runtime.hpp
        driver/cuda/copy.hpp driver/cuda/types.hpp driver/cuda/fft.hpp driver/cuda/kernels.cuh
        format/position.cu
        format/radial.cu format/spiral.cu
        process/cuda.cu process/cuda.hpp
        memory/cuda.hpp
        util/copy.cu
    )

    set_property(TARGET vortex PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET vortex PROPERTY CUDA_ARCHITECTURES 50 52 61 75 86)
    endif()

    # ref: https://github.com/IBM/aihwkit/pull/68/files
    if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11)
        target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_CUDA_DYNAMIC_RESAMPLING)
    endif()
endif()

if(WITH_HDF5)
    find_package(hdf5 CONFIG REQUIRED)
    # compatibility with non-standard HDF5 find scripts that omit namespace
    if(TARGET hdf5::hdf5_cpp-shared)
        target_link_libraries(vortex PUBLIC hdf5::hdf5_cpp-shared)
    elseif(TARGET hdf5_cpp-shared)
        target_link_libraries(vortex PUBLIC hdf5_cpp-shared)
    # compatibility for when HDF5 is provided as static library only
    elseif(TARGET hdf5::hdf5_cpp-static)
        target_link_libraries(vortex PUBLIC hdf5::hdf5_cpp-static)
    elseif(TARGET hdf5_cpp-static)
        target_link_libraries(vortex PUBLIC hdf5_cpp-static)
    else()
        message(SEND_ERROR "Could not find HDF5 target")
    endif()
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_HDF5)
    target_sources(vortex PRIVATE
        storage/hdf5.hpp
    )
endif()

#
# optional modular dependencies
#

# keep track of modular targets for installation later on
set(MODULAR_TARGETS)
add_library(vortex-driver-headers INTERFACE)
add_library(vortex::driver-headers ALIAS vortex-driver-headers)
target_link_libraries(vortex PRIVATE vortex::driver-headers)

if(WITH_DAQMX)
    find_package(NIDAQmx REQUIRED)

    if(ENABLE_MODULAR_BUILD)
        add_library(vortex-daqmx SHARED)
        add_library(vortex::daqmx ALIAS vortex-daqmx)
        target_link_libraries(vortex-daqmx PUBLIC vortex)

        set(TARGET vortex-daqmx)
        list(APPEND MODULAR_TARGETS vortex-daqmx)

        target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_DAQMX)
        target_include_directories(vortex-driver-headers INTERFACE ${NIDAQmx_INCLUDE_DIR})
    else()
        set(TARGET vortex)
    endif()

    target_link_libraries(${TARGET} PUBLIC NI::DAQmx)
    target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_DAQMX)
    target_sources(vortex PRIVATE
        io/daqmx.hpp
    )
    target_sources(${TARGET} PRIVATE
        driver/daqmx.cpp driver/daqmx.hpp
    )

    install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindNIDAQmx.cmake" DESTINATION share/vortex/cmake)
endif()

if(WITH_IMAQ)
    find_package(NIIMAQ REQUIRED)

    if(ENABLE_MODULAR_BUILD)
        add_library(vortex-imaq SHARED)
        add_library(vortex::imaq ALIAS vortex-imaq)
        target_link_libraries(vortex-imaq PUBLIC vortex)

        set(TARGET vortex-imaq)
        list(APPEND MODULAR_TARGETS vortex-imaq)

        target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_IMAQ)
        target_include_directories(vortex-driver-headers INTERFACE ${NIIMAQ_INCLUDE_DIR})
    else()
        set(TARGET vortex)
    endif()

    target_link_libraries(${TARGET} PUBLIC NI::IMAQ)
    target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_IMAQ)
    target_sources(vortex PRIVATE
        acquire/frame_grabber/imaq.hpp
    )
    target_sources(${TARGET} PRIVATE
        driver/imaq.cpp driver/imaq.hpp
    )

    install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindNIIMAQ.cmake" DESTINATION share/vortex/cmake)
endif()

if(WITH_ALAZAR OR WITH_ALAZAR_GPU)
    if(WITH_ALAZAR_GPU AND NOT WITH_CUDA)
        message(SEND_ERROR "Alazar GPU support requires CUDA support")
    endif()

    if(WITH_ALAZAR_GPU)
        # NOTE: ATS-GPU provides ATS-SDK too
        find_package(ATSGPU REQUIRED)
    else()
        find_package(ATSSDK REQUIRED)
    endif()

    if(ENABLE_MODULAR_BUILD)
        add_library(vortex-alazar SHARED)
        add_library(vortex::alazar ALIAS vortex-alazar)
        target_link_libraries(vortex-alazar PUBLIC vortex)

        set(TARGET vortex-alazar)
        list(APPEND MODULAR_TARGETS vortex-alazar)

        target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_ALAZAR)
        target_include_directories(vortex-driver-headers INTERFACE ${Alazar_ATSApi_INCLUDE_DIR})
        if(WITH_ALAZAR_GPU)
            target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_ALAZAR_GPU)
            target_include_directories(vortex-driver-headers INTERFACE ${Alazar_ATSGPU_INCLUDE_DIR})
        endif()
        if(WITH_ALAZAR_DAC)
            target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_ALAZAR_DAC)
        endif()
    else()
        set(TARGET vortex)
    endif()

    # needed during modular build for error messages that state version limitations
    target_compile_definitions(vortex PUBLIC ALAZAR_LIBRARY_VERSION_STRING=${ALAZAR_LIBRARY_VERSION_STRING})

    target_link_libraries(${TARGET} PUBLIC Alazar::ATSApi)
    target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_ALAZAR)
    if(WITH_ALAZAR_GPU)
        target_link_libraries(${TARGET} PUBLIC Alazar::ATS-GPU)
        target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_ALAZAR_GPU)
    endif()
    if(WITH_ALAZAR_DAC)
        target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_ALAZAR_DAC)
    endif()
    target_sources(vortex PRIVATE
        acquire/dso/alazar/config.hpp
        acquire/dso/alazar/base.hpp
        acquire/dso/alazar/host.hpp
        acquire/dso/alazar/fft.hpp
    )
    target_sources(${TARGET} PRIVATE
        driver/alazar/core.cpp driver/alazar/core.hpp
        driver/alazar/board.cpp driver/alazar/board.hpp
        driver/alazar/db.cpp driver/alazar/db.hpp
    )

    if(WITH_ALAZAR_GPU)
        install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindATSGPU.cmake" DESTINATION share/vortex/cmake)
    else()
        install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindATSSDK.cmake" DESTINATION share/vortex/cmake)
    endif()
endif()

if(WITH_TELEDYNE)
    find_package(ADQApi REQUIRED)

    if(ENABLE_MODULAR_BUILD)
        add_library(vortex-teledyne SHARED)
        add_library(vortex::teledyne ALIAS vortex-teledyne)
        target_link_libraries(vortex-teledyne PUBLIC vortex)

        set(TARGET vortex-teledyne)
        list(APPEND MODULAR_TARGETS vortex-teledyne)

        target_compile_definitions(vortex-driver-headers INTERFACE VORTEX_ENABLE_TELEDYNE)
        target_include_directories(vortex-driver-headers INTERFACE ${ADQAPI_INCLUDE_DIR})
    else()
        set(TARGET vortex)
    endif()

    target_link_libraries(${TARGET} PUBLIC ADQApi::ADQApi)
    target_compile_definitions(${TARGET} PUBLIC VORTEX_ENABLE_TELEDYNE)
    target_sources(vortex PRIVATE
        acquire/dso/teledyne.hpp
        memory/teledyne.hpp
    )
    target_sources(${TARGET} PRIVATE
        driver/teledyne.cpp driver/teledyne.hpp
    )

    install(FILES "${CMAKE_SOURCE_DIR}/cmake/FindADQApi.cmake" DESTINATION share/vortex/cmake)
endif()

#
# optional sources
#

if(WITH_CUDA AND WITH_ALAZAR_GPU)
    target_sources(vortex PRIVATE
        acquire/dso/alazar/cuda.hpp
        memory/alazar.hpp
    )
endif()

if(WITH_CUDA)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_ENGINE)
    target_sources(vortex PRIVATE
        engine/engine.hpp engine/detail/session.hpp
        endpoint/common.hpp endpoint/cpu.hpp endpoint/cuda.hpp endpoint/storage.hpp
    )
endif()

if(WITH_CUDA AND WITH_DAQMX AND WITH_ALAZAR)
    target_sources(vortex PRIVATE
        simple/engine.hpp
    )
endif()

#
# compile definitions
#

if(ENABLE_CUDA_KERNEL_SERIALIZATION)
    target_compile_definitions(vortex PUBLIC VORTEX_SERIALIZE_CUDA_KERNELS)
endif()

if(ENABLE_EXCEPTION_GUARDS)
    target_compile_definitions(vortex PUBLIC VORTEX_EXCEPTION_GUARDS)
endif()

if(ENABLE_MODULAR_BUILD)
    target_compile_definitions(vortex PUBLIC VORTEX_ENABLE_MODULAR_BUILD)
endif()

#
# installation
#

install(
    TARGETS vortex ${MODULAR_TARGETS}
    EXPORT vortex-targets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp")
install(FILES ${PROJECT_BINARY_DIR}/vortex/version.hpp DESTINATION include/vortex)
get_target_property(_TYPE vortex TYPE)
if(WIN32 AND _TYPE STREQUAL "SHARED_LIBRARY")
    install(FILES "$<TARGET_PDB_FILE:${PROJECT_NAME}>" DESTINATION bin)
endif()
