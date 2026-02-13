# Find NIQADmx

if(NOT TARGET NIIMAQ)

    set(SEARCH
        "${NIIMAQ_DIR}"
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
    find_path(NIIMAQ_INCLUDE_DIR NAMES "niimaq.h" PATHS ${SEARCH} PATH_SUFFIXES ${INCLUDE_SUFFIX})

    # imported library or shared object
    find_library(NIIMAQ_LIB_RELEASE NAMES "imaq" PATHS ${SEARCH} PATH_SUFFIXES ${LIBRARY_SUFFIX})
    set(NIIMAQ_LIB_DEBUG ${NIIMAQ_LIB_RELEASE})

    if(NIIMAQ_LIB_RELEASE AND NIIMAQ_INCLUDE_DIR)
        # create target
        add_library(NIIMAQ SHARED IMPORTED GLOBAL)
        add_library(NI::IMAQ ALIAS NIIMAQ)

        # populate locations
        set_property(TARGET NIIMAQ PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${NIIMAQ_INCLUDE_DIR}")
        if(WIN32)
            set_property(TARGET NIIMAQ PROPERTY IMPORTED_IMPLIB_DEBUG "${NIIMAQ_LIB_DEBUG}")
            set_property(TARGET NIIMAQ PROPERTY IMPORTED_IMPLIB "${NIIMAQ_LIB_RELEASE}")
        endif(WIN32)
        set_property(TARGET NIIMAQ PROPERTY IMPORTED_LOCATION "${NIIMAQ_LIB_RELEASE}")

        #notify user
        message(STATUS "Found NI-IMAQ: ${NIIMAQ_LIB_RELEASE} and ${NIIMAQ_INCLUDE_DIR}")
        set(NIIMAQ_FOUND TRUE)
    else()
        message(SEND_ERROR "Failed to find NI-IMAQ")
    endif()

    unset(SEARH)
    unset(INCLUDE_SUFFIX)
    unset(LIBRARY_SUFFIX)

endif()
