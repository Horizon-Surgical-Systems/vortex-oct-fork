project (vortex-python)

# NOTE: find Python3 first to ensure that pybind11 finds the same Python referenced by search hints
find_package(Python3 COMPONENTS Interpreter NumPy)
find_package(pybind11 CONFIG REQUIRED)

find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-python CONFIG REQUIRED)

set(DOCSTRING_SOURCES
    module/acquire.rst
    module/engine.rst
    module/process.rst
    module/scan.rst
)
list(TRANSFORM DOCSTRING_SOURCES PREPEND "${CMAKE_SOURCE_DIR}/doc/")

# generate docstrings from the documentation
set(DOCSTRING_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/doc/docstring.py)
set(DOCSTRING_OUTPUT ${PROJECT_BINARY_DIR}/docstring.cpp)
add_custom_command(
    OUTPUT  ${DOCSTRING_OUTPUT}
    COMMAND ${Python3_EXECUTABLE} ${DOCSTRING_SCRIPT} ${CMAKE_SOURCE_DIR}/doc/module ${DOCSTRING_OUTPUT}
    DEPENDS ${DOCSTRING_SCRIPT} ${DOCSTRING_SOURCES}
    COMMENT "Generating docstrings"
)

set(VORTEX_PYTHON_SOURCES
    bindings.cpp

    bind/acquire.cpp bind/acquire.hpp
    bind/common.hpp
    bind/core.cpp
    bind/engine.cpp bind/engine.hpp
    bind/exception.hpp
    bind/format.cpp
    bind/io.cpp
    bind/log.cpp bind/log.hpp
    bind/memory.cpp bind/memory.hpp
    bind/marker.cpp bind/marker.hpp
    bind/process.cpp bind/process.hpp
    bind/scan.cpp bind/scan.hpp
    bind/storage.cpp

    module/loader.cpp module/loader.hpp

    ${PROJECT_BINARY_DIR}/docstring.cpp doc/docstring.hpp
)
set(VORTEX_PYTHON_ALAZAR_SOURCES
    module/alazar.cpp module/helper.hpp
)
set(VORTEX_PYTHON_DAQMX_SOURCES
    module/daqmx.cpp module/helper.hpp
)
set(VORTEX_PYTHON_IMAQ_SOURCES
    module/imaq.cpp module/helper.hpp
)
set(VORTEX_PYTHON_TELEDYNE_SOURCES
    module/teledyne.cpp module/helper.hpp
)
set(VORTEX_PYTHON_SIMPLE_SOURCES
    module/simple.cpp module/helper.hpp
)

set(MODULE_TARGETS)
if(ENABLE_MODULAR_BUILD)
    # compile each driver module into its own library
    if(WITH_ALAZAR)
        add_library(vortex-python-alazar SHARED ${VORTEX_PYTHON_ALAZAR_SOURCES})
        target_link_libraries(vortex-python-alazar PUBLIC vortex::alazar)
        list(APPEND MODULE_TARGETS vortex-python-alazar)
    endif()
    if(WITH_DAQMX)
        add_library(vortex-python-daqmx SHARED ${VORTEX_PYTHON_DAQMX_SOURCES})
        target_link_libraries(vortex-python-daqmx PUBLIC vortex::daqmx)
        list(APPEND MODULE_TARGETS vortex-python-daqmx)
    endif()
    if(WITH_IMAQ)
        add_library(vortex-python-imaq SHARED ${VORTEX_PYTHON_IMAQ_SOURCES})
        target_link_libraries(vortex-python-imaq PUBLIC vortex::imaq)
        list(APPEND MODULE_TARGETS vortex-python-imaq)
    endif()
    if(WITH_TELEDYNE)
        add_library(vortex-python-teledyne SHARED ${VORTEX_PYTHON_TELEDYNE_SOURCES})
        target_link_libraries(vortex-python-teledyne PUBLIC vortex::teledyne)
        list(APPEND MODULE_TARGETS vortex-python-teledyne)
    endif()
    if(WITH_ALAZAR AND WITH_CUDA AND WITH_DAQMX)
        add_library(vortex-python-simple SHARED ${VORTEX_PYTHON_SIMPLE_SOURCES})
        target_link_libraries(vortex-python-simple PUBLIC vortex::alazar vortex::daqmx)
        list(APPEND MODULE_TARGETS vortex-python-simple)
    endif()

    set(MODULE_PATHS)
    foreach(TARGET ${MODULE_TARGETS})
        list(APPEND MODULE_PATHS "$<TARGET_FILE:${TARGET}>")
        target_link_libraries(${TARGET} PUBLIC pybind11::module Python3::NumPy xtensor-python)
    endforeach()

    # save out paths for Python-driven build
    set(MODULE_PATHS_FILE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/modules.txt")
    # NOTE: CMake refuses to output a semi-colon delimited list
    list(JOIN MODULE_PATHS "|" MODULE_PATHS_JOINED)
    add_custom_command(
        DEPENDS ${CMAKE_SOURCE_DIR}/cmake/write_list.cmake
        OUTPUT ${MODULE_PATHS_FILE}
        COMMAND ${CMAKE_COMMAND}
            -DTARGET="${MODULE_PATHS_FILE}"
            -DCONTENTS="${MODULE_PATHS_JOINED}"
            -P ${CMAKE_SOURCE_DIR}/cmake/write_list.cmake
        COMMENT "Writing driver module index"
    )
    list(APPEND VORTEX_PYTHON_SOURCES ${MODULE_PATHS_FILE})
else()
    # compile all driver modules into the main extension
    if(WITH_ALAZAR)
        list(APPEND VORTEX_PYTHON_SOURCES ${VORTEX_PYTHON_ALAZAR_SOURCES})
    endif()
    if(WITH_DAQMX)
        list(APPEND VORTEX_PYTHON_SOURCES ${VORTEX_PYTHON_DAQMX_SOURCES})
    endif()
    if(WITH_IMAQ)
        list(APPEND VORTEX_PYTHON_SOURCES ${VORTEX_PYTHON_IMAQ_SOURCES})
    endif()
    if(WITH_TELEDYNE)
        list(APPEND VORTEX_PYTHON_SOURCES ${VORTEX_PYTHON_TELEDYNE_SOURCES})
    endif()
    if(WITH_ALAZAR AND WITH_CUDA AND WITH_DAQMX)
        list(APPEND VORTEX_PYTHON_SOURCES ${VORTEX_PYTHON_SIMPLE_SOURCES})
    endif()
endif()

if(ENABLE_PYBIND11_OPTIMIZATIONS)
    # keep the defaults
    set(PYBIND11_OPTIONS "")
else()
    # disable optimmizations
    set(PYBIND11_OPTIONS "NO_EXTRAS")
endif()
pybind11_add_module(vortex-python MODULE ${VORTEX_PYTHON_SOURCES} ${PYBIND11_OPTIONS})
add_library(vortex::python ALIAS vortex-python)
if(ENABLE_PYBIND11_OPTIMIZATIONS)
    target_compile_definitions(vortex PUBLIC VORTEX_PYBIND11_OPTIMIZATIONS)
endif()

# trickery to name the output "vortex${PYTHON_MODULE_EXTENSION}" without
# causing LNK1149 due to same input and output base files names
# essentially, move part of the Python-specific suffix into OUTPUT_NAME
get_filename_component(_EXT "${PYTHON_MODULE_EXTENSION}" LAST_EXT)
get_filename_component(_SFX "${PYTHON_MODULE_EXTENSION}" NAME_WLE)
set_target_properties(vortex-python PROPERTIES
    SUFFIX "${_EXT}"
    OUTPUT_NAME "vortex${_SFX}"
    DEBUG_POSTFIX ""
    RELEASE_POSTFIX ""
    MINSIZEREL_POSTFIX ""
    RELWITHDEBINFO_POSTFIX ""
)

target_include_directories(vortex-python PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../")
foreach(TARGET ${MODULE_TARGETS})
    string(TOUPPER ${TARGET} VAR)
    string(REPLACE "-" "_" VAR ${VAR})
    # need to know file name of driver modules in platform-independent manner
    target_compile_definitions(vortex-python PRIVATE ${VAR}_MODULE="$<TARGET_FILE_NAME:${TARGET}>")
endforeach()

target_link_libraries(vortex-python PUBLIC vortex::vortex Python3::NumPy xtensor-python)
target_link_libraries(vortex-python PRIVATE vortex::driver-headers)
if(ENABLE_MODULAR_BUILD AND UNIX)
    target_link_libraries(vortex-python PUBLIC dl)
endif()

if(BUILD_PYTHON_WHEEL)
    # build package name, depending on CUDA usage
    set(PACKAGE_NAME "vortex-oct")
    if(WITH_CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(PACKAGE_NAME "${PACKAGE_NAME}-cuda${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")
    endif()

    # detect output wheel file name
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "
from distutils.core import Extension;
from setuptools.dist import Distribution;
ext = Extension('tmp', ['tmp.c']);
dist = Distribution(attrs={'name': '${PACKAGE_NAME}', 'version': '${VORTEX_VERSION_STRING}', 'ext_modules': [ext]});
bdist_wheel_cmd = dist.get_command_obj('bdist_wheel');
bdist_wheel_cmd.ensure_finalized();
distname = bdist_wheel_cmd.wheel_dist_name;
tag = '-'.join(bdist_wheel_cmd.get_tag());
wheel_name = f'{distname}-{tag}.whl';
print(wheel_name);"
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE WHEEL_NAME
        ERROR_VARIABLE ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT ${RESULT} EQUAL 0)
        message(FATAL_ERROR "wheel name detection failed\n${ERROR}")
    endif()
    set(WHEEL_DIR ${CMAKE_BINARY_DIR}/dist)
    set(WHEEL_PATH ${WHEEL_DIR}/${WHEEL_NAME})

    if(ENABLE_OUT_OF_TREE_PACKAGING)
        set(DEPLOY_PATH_RESTRICTION "")
    else()
        set(DEPLOY_PATH_RESTRICTION "^${CMAKE_SOURCE_DIR}")
    endif()

    # add target and associated commands for building the wheel
    add_custom_target(vortex-python-wheel ALL
        DEPENDS ${WHEEL_PATH}
        SOURCES setup.py.in ${CMAKE_SOURCE_DIR}/cmake/prepare_wheel.cmake
    )
    add_dependencies(vortex-python-wheel vortex-python ${MODULE_TARGETS})
    add_custom_command(
        DEPENDS
            $<TARGET_FILE:vortex-python>
            ${MODULE_PATHS}
            ${CMAKE_SOURCE_DIR}/cmake/prepare_wheel.cmake
            ${CMAKE_SOURCE_DIR}/cmake/deploy_dependencies.cmake
            ${CMAKE_SOURCE_DIR}/setup.py setup.py.in
        OUTPUT ${WHEEL_PATH}
        COMMAND ${CMAKE_COMMAND}
            -DROOT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            -DROOT_BINARY_DIR="${CMAKE_BINARY_DIR}"
            -DPACKAGE_NAME="${PACKAGE_NAME}"
            -DEXTENSION_PATH="$<TARGET_FILE:vortex-python>"
            -DEXTENSION_VERSION="${VORTEX_VERSION_STRING}"
            -DMODULE_PATHS="${MODULE_PATHS_JOINED}"
            -DSETUP_TEMPLATE="${CMAKE_CURRENT_LIST_DIR}/setup.py.in"
            -DOUTPUT_DIR="${WHEEL_DIR}"
            -DWORK_DIR="${PROJECT_BINARY_DIR}"
            -DPYTHON_EXE="${Python3_EXECUTABLE}"
            -DDEPLOY_DEPENDENCIES="${ENABLE_DEPENDENCY_PACKAGING}"
            -DDEPLOY_PATH_RESTRICTION="${DEPLOY_PATH_RESTRICTION}"
            -DDEPLOY_STUBS="ON"
            -P "${CMAKE_SOURCE_DIR}/cmake/prepare_wheel.cmake"
        COMMENT "Building Python wheel for vortex${PYTHON_MODULE_EXTENSION}"
    )

    if(INSTALL_PYTHON_WHEEL)
        # add target for installing the wheel during build
        set(TIMESTAMP_PATH "${PROJECT_BINARY_DIR}/install_wheel.timestamp")
        add_custom_target(vortex-python-install ALL
            DEPENDS ${TIMESTAMP_PATH}
        )
        add_dependencies(vortex-python-install vortex-python-wheel)
        add_custom_command(
            DEPENDS ${WHEEL_PATH}
            OUTPUT ${TIMESTAMP_PATH}
            COMMAND "${Python3_EXECUTABLE}" -m pip install --no-dependencies --force-reinstall "${WHEEL_PATH}"
            COMMAND ${CMAKE_COMMAND} -E touch ${TIMESTAMP_PATH}
            COMMENT "Installing ${WHEEL_NAME}"
        )
    endif()

    # install the wheel during the normal install process
    install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install --force-reinstall ${WHEEL_PATH})")
endif()
