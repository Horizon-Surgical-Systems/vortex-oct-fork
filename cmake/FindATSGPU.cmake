# Find Alazar

if(NOT TARGET Alazar::ATSApi)

    set(SEARCH
        "${AlazarTech_DIR}"
    )

    set(INCLUDE_SUFFIX)
    set(LIBRARY_SUFFIX)

    if(MSVC)
        # add default installation locations for Windows
        list(APPEND SEARCH "C:/AlazarTech/ATS-GPU/*/base")
        set(INCLUDE_SUFFIX "include")

        if(CMAKE_CL_64)
            set(LIBRARY_SUFFIX "library/x64")
        else()
            set(LIBRARY_SUFFIX "library/Win32")
        endif()
    endif()

    #
    # ATSApi
    #

    # includes
    find_path(Alazar_ATSApi_INCLUDE_DIR NAMES "AlazarApi.h" PATHS ${SEARCH} PATH_SUFFIXES ${INCLUDE_SUFFIX})

    # imported library or shared object
    find_library(Alazar_ATSApi_LIB_RELEASE NAMES "ATSApi" PATHS ${SEARCH} PATH_SUFFIXES ${LIBRARY_SUFFIX})
    set(Alazar_ATSApi_LIB_DEBUG ${Alazar_ATSApi_LIB_RELEASE})

    if(Alazar_ATSApi_LIB_RELEASE AND Alazar_ATSApi_INCLUDE_DIR)
        add_library(Alazar::ATSApi SHARED IMPORTED GLOBAL)

        set_property(TARGET Alazar::ATSApi PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Alazar_ATSApi_INCLUDE_DIR})
        if(WIN32)
            set_property(TARGET Alazar::ATSApi PROPERTY IMPORTED_IMPLIB_DEBUG ${Alazar_ATSApi_LIB_DEBUG})
            set_property(TARGET Alazar::ATSApi PROPERTY IMPORTED_IMPLIB ${Alazar_ATSApi_LIB_RELEASE})
        endif(WIN32)
        set_property(TARGET Alazar::ATSApi PROPERTY IMPORTED_LOCATION ${Alazar_ATSApi_LIB_RELEASE})

        #notify user
        message(STATUS "Found ATSApi: ${Alazar_ATSApi_LIB_RELEASE} and ${Alazar_ATSApi_INCLUDE_DIR}")
        set(ATSSDK_FOUND TRUE)
    else()
        message(SEND_ERROR "Failed to find Alazar ATSApi")
    endif()

    #
    # ATS-GPU
    #

    # includes
    find_path(Alazar_ATSGPU_INCLUDE_DIR NAMES "ATS_GPU.h" PATHS ${SEARCH} PATH_SUFFIXES ${INCLUDE_SUFFIX})

    # imported library or shared object
    find_library(Alazar_ATSGPU_LIB_RELEASE NAMES "ATS_GPU" PATHS ${SEARCH} PATH_SUFFIXES ${LIBRARY_SUFFIX})
    set(Alazar_ATSGPU_LIB_DEBUG ${Alazar_ATSGPU_LIB_RELEASE})

    if(Alazar_ATSGPU_LIB_RELEASE AND Alazar_ATSGPU_INCLUDE_DIR)
        add_library(Alazar::ATS-GPU SHARED IMPORTED GLOBAL)

        set_property(TARGET Alazar::ATS-GPU PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Alazar_ATSGPU_INCLUDE_DIR})
        if(WIN32)
            set_property(TARGET Alazar::ATS-GPU PROPERTY IMPORTED_IMPLIB_DEBUG ${Alazar_ATSGPU_LIB_DEBUG})
            set_property(TARGET Alazar::ATS-GPU PROPERTY IMPORTED_IMPLIB ${Alazar_ATSGPU_LIB_RELEASE})
        endif(WIN32)
        set_property(TARGET Alazar::ATS-GPU PROPERTY IMPORTED_LOCATION ${Alazar_ATSGPU_LIB_RELEASE})

        if(NOT Alazar_VERSION)
            if(MSVC)
                # find from include path
                string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Alazar_VERSION ${Alazar_ATSGPU_INCLUDE_DIR})
            elseif(UNIX)
                # find from package version
                execute_process(COMMAND dpkg -l OUTPUT_VARIABLE DPKG_LISTING ERROR_QUIET)
                string(REGEX MATCH "libats +([0-9]+\\.[0-9]+\\.[0-9]+)" IGNORE "${DPKG_LISTING}")
                set(Alazar_VERSION ${CMAKE_MATCH_1})
                unset(DPKG_LISTING)
            endif()
        endif()
        if(NOT Alazar_VERSION)
            set(Alazar_VERSION "0.0.0")
            message(WARNING "Unable to detect Alazar ATS-GPU version so using ${Alazar_VERSION} -> Define Alazar_VERSION=XX.YY.ZZ for CMake")
        endif()

        string(REGEX MATCHALL "([0-9]+)\\.([0-9]+)\\.([0-9]+)" IGNORE ${Alazar_VERSION})
        math(EXPR ATSGPU_VERSION "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
        set_property(TARGET Alazar::ATS-GPU PROPERTY INTERFACE_COMPILE_DEFINITIONS
            ATSGPU_VERSION=${ATSGPU_VERSION}
            ALAZAR_LIBRARY_VERSION_STRING="ATS-GPU v${Alazar_VERSION}"
        )
        unset(IGNORE)

        #notify user
        message(STATUS "Found ATS-GPU: ${Alazar_ATSGPU_LIB_RELEASE} and ${Alazar_ATSGPU_INCLUDE_DIR} (found version \"${Alazar_VERSION}\")")
        if(ATSSDK_FOUND)
            set(ATSGPU_FOUND TRUE)
        endif()
    else()
        message(SEND_ERROR "Failed to find Alazar ATS-GPU")
    endif()

    #
    # ATS-CUDA
    #

    # includes
    find_path(Alazar_ATS-CUDA_INCLUDE_DIR NAMES "ATS_CUDA.h" PATHS ${SEARCH} PATH_SUFFIXES ${INCLUDE_SUFFIX})

    # imported library or shared object
    find_library(Alazar_ATS-CUDA_LIB_RELEASE NAMES "ATS_CUDA" PATHS ${SEARCH} PATH_SUFFIXES ${LIBRARY_SUFFIX})
    set(Alazar_ATS-CUDA_LIB_DEBUG ${Alazar_ATS-CUDA_LIB_RELEASE})

    if(Alazar_ATS-CUDA_LIB_RELEASE AND Alazar_ATS-CUDA_INCLUDE_DIR)
        add_library(Alazar::ATS-CUDA SHARED IMPORTED GLOBAL)

        set_property(TARGET Alazar::ATS-CUDA PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Alazar_ATS-CUDA_INCLUDE_DIR})
        if(WIN32)
            set_property(TARGET Alazar::ATS-CUDA PROPERTY IMPORTED_IMPLIB_DEBUG ${Alazar_ATS-CUDA_LIB_DEBUG})
            set_property(TARGET Alazar::ATS-CUDA PROPERTY IMPORTED_IMPLIB ${Alazar_ATS-CUDA_LIB_RELEASE})
        endif(WIN32)
        set_property(TARGET Alazar::ATS-CUDA PROPERTY IMPORTED_LOCATION ${Alazar_ATS-CUDA_LIB_RELEASE})

        #notify user
        message(STATUS "Found ATS-CUDA: ${Alazar_ATS-CUDA_LIB_RELEASE} and ${Alazar_ATS-CUDA_INCLUDE_DIR}")
        if(ATSGPU_FOUND)
            set(ATSCUDA_FOUND TRUE)
        endif()
    else()
        message(SEND_ERROR "Failed to find Alazar ATS-CUDA")
    endif()

    unset(SEARCH)
    unset(INCLUDE_SUFFIX)
    unset(LIBRARY_SUFFIX)

endif()
