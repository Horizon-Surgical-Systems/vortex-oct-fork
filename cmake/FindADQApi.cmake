# Find Teledyne ADQ SDK

if(NOT TARGET ADQApi::ADQApi)

    set(SEARCH
        "${Teledyne_DIR}"
    )

    if(MSVC)
        # add default installation locations for Windows
        if(CMAKE_CL_64)
            list(APPEND SEARCH "C:/Program Files/SP Devices/ADQAPI_x64")
        else()
            list(APPEND SEARCH "C:/Program Files/SP Devices/ADQAPI")
        endif()
    endif()

    find_path(ADQAPI_INCLUDE_DIR NAMES ADQAPI.h PATHS ${PC_ADQAPI_INCLUDE_DIRS} ${SEARCH})
    find_library(ADQAPI_LIBRARY NAMES adq ADQAPI PATHS ${PC_ADQAPI_LIBRARY_DIRS} ${SEARCH})

    if(ADQAPI_LIBRARY AND ADQAPI_INCLUDE_DIR)
        add_library(ADQApi::ADQApi UNKNOWN IMPORTED)
        set_target_properties(ADQApi::ADQApi PROPERTIES
            IMPORTED_LOCATION "${ADQAPI_LIBRARY}"
            INTERFACE_COMPILE_OPTIONS "${PC_ADQAPI_CFLAGS_OTHER}"
            INTERFACE_INCLUDE_DIRECTORIES "${ADQAPI_INCLUDE_DIR}"
        )
        if(UNIX)
            target_compile_definitions(ADQApi::ADQApi INTERFACE LINUX)
        endif()

        # notify user
        message(STATUS "Found ADQApi: ${ADQAPI_LIBRARY} and ${ADQAPI_INCLUDE_DIR}")
        set(ADQApi_FOUND TRUE)
    else()
        message(SEND_ERROR "Failed to find Teledyne ADQApi")
    endif()

    unset(SEARCH)

endif()
