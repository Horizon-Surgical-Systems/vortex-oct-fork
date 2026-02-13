set(_INITIAL_VCPKG_MANIFEST_FEATURES ${VCPKG_MANIFEST_FEATURES})

macro(vcpkg_option VARIABLE FEATURE DEFAULT HELP)
    set(VALUE ${DEFAULT})

    if(DEFINED ${VARIABLE} AND _INITIAL_VCPKG_MANIFEST_FEATURES)
        # user controlling dependencies explicitly or building as vcpkg port -> do not interfere
        list(APPEND _MANUAL_FEATURES ${FEATURE})
        list(APPEND _MANUAL_OPTIONS ${VARIABLE})

    elseif(_INITIAL_VCPKG_MANIFEST_FEATURES)
        # likely top-level build using vcpkg for dependencies with explicit features -> set option to match
        # NOTE: even if vcpkg is not used, the user may set VCPKG_MANIFEST_FEATURES to control build options

        if(${FEATURE} IN_LIST VCPKG_MANIFEST_FEATURES)
            set(VALUE ON)
            list(APPEND _ENABLED_OPTIONS ${VARIABLE})
        else()
            set(VALUE OFF)
            list(APPEND _DISABLED_OPTIONS ${VARIABLE})
        endif()

    elseif(NOT CMAKE_TOOLCHAIN_FILE)
        # no vcpkg toolchain loaded or features set with VCPKG_MANIFEST_FEATURES -> no need to set manifest features

    else()
        # likely top-level build using vcpkg for dependencies with option defaults -> set feature for default options

        if(${VARIABLE} OR (${DEFAULT} AND NOT DEFINED ${VARIABLE}))
            list(APPEND VCPKG_MANIFEST_FEATURES ${FEATURE})
            list(APPEND _ENABLED_FEATURES ${FEATURE})
        endif()

    endif()

    # actually set the option
    set(${VARIABLE} ${VALUE} CACHE BOOL ${HELP})

endmacro()

macro(report_vcpkg_options)

    if(_ENABLED_OPTIONS OR _DISABLED_OPTIONS)
        list(SORT _ENABLED_OPTIONS)
        list(SORT _DISABLED_OPTIONS)
        list(SORT _MANUAL_OPTIONS)
        list(JOIN _DISABLED_OPTIONS ", " _DISABLED_OPTIONS)
        list(JOIN _ENABLED_OPTIONS ", " _ENABLED_OPTIONS)
        list(JOIN _MANUAL_OPTIONS ", " _MANUAL_OPTIONS)

        message(STATUS "Updated build options based on vcpkg features")
        message(STATUS "  Enabled:  ${_ENABLED_OPTIONS}")
        message(STATUS "  Disabled: ${_DISABLED_OPTIONS}")
        if(_MANUAL_OPTIONS)
            message(STATUS "  Manual:   ${_MANUAL_OPTIONS}")
        endif()

    elseif(_ENABLED_FEATURES)
        list(SORT _ENABLED_FEATURES)
        list(SORT _MANUAL_FEATURES)
        list(JOIN _ENABLED_FEATURES ", " _ENABLED_FEATURES)
        list(JOIN _MANUAL_FEATURES ", " _MANUAL_FEATURES)

        message(STATUS "Updated vcpkg features based on build options")
        message(STATUS "  Features: ${_ENABLED_FEATURES}")
        if(_MANUAL_FEATURES)
            message(STATUS "  Manual:   ${_MANUAL_FEATURES}")
        endif()

    endif()
endmacro()
