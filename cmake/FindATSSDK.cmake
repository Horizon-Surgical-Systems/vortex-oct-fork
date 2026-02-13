# Find Alazar

if(NOT TARGET Alazar::ATSApi)

    set(SEARCH
        "${AlazarTech_DIR}"
    )

    if(MSVC)
        # add default installation locations for Windows
        list(APPEND SEARCH "C:/AlazarTech/ATS-SDK/*/Samples_C")
        set(INCLUDE_SUFFIX "Include")

        if(CMAKE_CL_64)
            set(LIBRARY_SUFFIX "Library/x64")
        else()
            set(LIBRARY_SUFFIX "Library/Win32")
        endif()
    endif()

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

        # detect version if needed
        if(NOT Alazar_VERSION)
            if(MSVC)
                # find from include path
                string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Alazar_VERSION ${Alazar_ATSApi_INCLUDE_DIR})
            elseif(UNIX)
                # find from package version
                execute_process(COMMAND dpkg -l OUTPUT_VARIABLE DPKG_LISTING ERROR_QUIET)
                string(REGEX MATCH "(libats|drivers-ats[0-9]+-dkms) +([0-9]+\\.[0-9]+\\.[0-9]+)" IGNORE "${DPKG_LISTING}")
                set(Alazar_VERSION ${CMAKE_MATCH_2})
                unset(DPKG_LISTING)
            endif()
        endif()
        if(NOT Alazar_VERSION)
            set(Alazar_VERSION "0.0.0")
            message(WARNING "Unable to detect Alazar ATS-SDK version so using ${Alazar_VERSION} -> Define Alazar_VERSION=XX.YY.ZZ for CMake")
        endif()

        string(REGEX MATCHALL "([0-9]+)\\.([0-9]+)\\.([0-9]+)" IGNORE ${Alazar_VERSION})
        math(EXPR ATSAPI_VERSION "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
        set(ALAZAR_LIBRARY_VERSION_STRING="ATS-SDK v${Alazar_VERSION}")

        set_property(TARGET Alazar::ATSApi PROPERTY INTERFACE_COMPILE_DEFINITIONS
            ATSAPI_VERSION=${ATSAPI_VERSION}
            ALAZAR_LIBRARY_VERSION_STRING=${ALAZAR_LIBRARY_VERSION_STRING}
        )
        unset(IGNORE)

        # notify user
        message(STATUS "Found ATSApi: ${Alazar_ATSApi_LIB_RELEASE} and ${Alazar_ATSApi_INCLUDE_DIR} (found version \"${Alazar_VERSION}\")")
        set(ATSSDK_FOUND TRUE)
    else()
        message(SEND_ERROR "Failed to find Alazar ATSApi")
    endif()

    unset(SEARCH)
    unset(INCLUDE_SUFFIX)
    unset(LIBRARY_SUFFIX)

endif()
