if(NOT VORTEX_VERSION_STRING)
    find_package(Git QUIET)
    if(Git_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git/index")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe --tags --always HEAD
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE IGNORE
            OUTPUT_VARIABLE OUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        set_property(GLOBAL APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/.git/index")

        string(REGEX MATCH "v?([0-9]+\\.[0-9]+\\.[0-9]+)-?([0-9]*)-?([0-9a-z]*)" "\\1+\\2" IGNORE ${OUT})
        set(VORTEX_VERSION_STRING "${CMAKE_MATCH_1}")
        if(CMAKE_MATCH_2)
            string(APPEND VORTEX_VERSION_STRING "+${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
        endif()
    endif()
endif()
if(NOT VORTEX_VERSION_STRING)
    set(VORTEX_VERSION_STRING "0.0.0")
    message(WARNING "Unable to detect vortex version so using ${VORTEX_VERSION} -> Define VORTEX_VERSION=XX.YY.ZZ for CMake")
endif()
message(STATUS "Detected vortex v${VORTEX_VERSION_STRING}")

string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" IGNORE ${VORTEX_VERSION_STRING})
set(VORTEX_VERSION_MAJOR ${CMAKE_MATCH_1})
set(VORTEX_VERSION_MINOR ${CMAKE_MATCH_2})
set(VORTEX_VERSION_PATCH ${CMAKE_MATCH_3})
math(EXPR VORTEX_VERSION "${CMAKE_MATCH_1} * 10000 + ${CMAKE_MATCH_2} * 100 + ${CMAKE_MATCH_3}")
