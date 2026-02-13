# Find NIQADmx

if(NOT TARGET NIQADmx)

    set(SEARCH
        "${NIDAQmx_DIR}"
    )

    if(MSVC)
        set(INCLUDE_SUFFIX "include")

        # add default installation locations for Windows
        if(CMAKE_CL_64)
            list(APPEND SEARCH "C:/Program Files (x86)/National Instruments/Shared/ExternalCompilerSupport/C")
            set(LIBRARY_SUFFIX "lib64/msvc")
        else()
            list(APPEND SEARCH "C:/Program Files/National Instruments/Shared/ExternalCompilerSupport/C")
            set(LIBRARY_SUFFIX "lib32/msvc")
        endif()
    endif()

    # includes
    find_path(NIDAQmx_INCLUDE_DIR NAMES "NIDAQmx.h" PATHS ${SEARCH} PATH_SUFFIXES ${INCLUDE_SUFFIX})

    # imported library or shared object
    find_library(NIDAQmx_LIB_RELEASE NAMES "nidaqmx" PATHS ${SEARCH} PATH_SUFFIXES ${LIBRARY_SUFFIX})
    set(NIDAQmx_LIB_DEBUG ${NIDAQmx_LIB_RELEASE})

    if(NIDAQmx_LIB_RELEASE AND NIDAQmx_INCLUDE_DIR)
        # create target
        add_library(NIDAQmx SHARED IMPORTED GLOBAL)
        add_library(NI::DAQmx ALIAS NIDAQmx)

        # populate locations
        set_property(TARGET NIDAQmx PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${NIDAQmx_INCLUDE_DIR}")
        if(WIN32)
            set_property(TARGET NIDAQmx PROPERTY IMPORTED_IMPLIB_DEBUG "${NIDAQmx_LIB_DEBUG}")
            set_property(TARGET NIDAQmx PROPERTY IMPORTED_IMPLIB "${NIDAQmx_LIB_RELEASE}")
        endif(WIN32)
        set_property(TARGET NIDAQmx PROPERTY IMPORTED_LOCATION "${NIDAQmx_LIB_RELEASE}")

        #notify user
        message(STATUS "Found NI-DAQmx: ${NIDAQmx_LIB_RELEASE} and ${NIDAQmx_INCLUDE_DIR}")
        set(NIDAQmx_FOUND TRUE)
    else()
        message(SEND_ERROR "Failed to find NI-DAQmx")
    endif()

    unset(SEARH)
    unset(INCLUDE_SUFFIX)
    unset(LIBRARY_SUFFIX)

endif()
