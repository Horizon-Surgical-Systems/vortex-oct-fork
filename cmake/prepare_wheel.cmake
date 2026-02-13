# EXTENSION_PATH - path to vortex extension
# OUTPUT_DIR
# WORK_DIR
# SETUP_TEMPLATE - path to setup.py template
# PYTHON_EXE - path to Python executable
# DEPLOY_DEPENDENCIES - include binary dependencies in Python wheel
# DEPLOY_PATH_RESTRICTION - only include binary dependencies under the given path
# ROOT_SOURCE_DIR, ROOT_BINARY_DIR - root of the source tree and binary outputs

if(NOT DEFINED OUTPUT_DIR)
    get_filename_component(OUTPUT_DIR ${EXTENSION_PATH} DIRECTORY)
endif()
get_filename_component(EXTENSION_NAME ${EXTENSION_PATH} NAME_WE)

set(WHEEL_DIR ${WORK_DIR}/wheel)
set(PACKAGE_ROOT ${WHEEL_DIR}/${EXTENSION_NAME})

# setup fresh wheel directory
file(REMOVE_RECURSE ${WHEEL_DIR})
file(MAKE_DIRECTORY ${PACKAGE_ROOT})

# deploy the basic files
configure_file(${SETUP_TEMPLATE} ${WHEEL_DIR}/setup.py @ONLY)
file(COPY ${ROOT_SOURCE_DIR}/LICENSE DESTINATION ${WHEEL_DIR})
file(COPY ${ROOT_SOURCE_DIR}/pyproject.toml DESTINATION ${WHEEL_DIR})

# build the wheel
file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_NATIVE_COMMAND)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env CMAKE_COMMAND=${CMAKE_NATIVE_COMMAND} ${PYTHON_EXE} -m pip wheel . -w ${WHEEL_DIR}
    WORKING_DIRECTORY ${WHEEL_DIR}
    OUTPUT_QUIET
    COMMAND_ERROR_IS_FATAL ANY
)

# move the wheel to the output directory
file(GLOB WHEEL_SRC "${WHEEL_DIR}/${EXTENSION_NAME}*.whl")
get_filename_component(WHEEL_NAME ${WHEEL_SRC} NAME)

set(WHEEL_DST "${OUTPUT_DIR}/${WHEEL_NAME}")
file(MAKE_DIRECTORY ${OUTPUT_DIR})
file(RENAME ${WHEEL_SRC} ${WHEEL_DST})
message("${EXTENSION_PATH} -> ${WHEEL_DST}")
