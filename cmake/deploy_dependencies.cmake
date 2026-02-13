# PACKAGE_ROOT - path to place dependencies
# PACKAGE_TARGET_PATHS - path to vortex extension
# BUILD_TARGET_PATHS - path to vortex extension
# VORTEX_DIR - directory containing vortex library
# DEPLOY_PATH_RESTRICTION - only include binary dependencies under the given path

if("${DEPLOY_PATH_RESTRICTION}" STREQUAL "")
    # match all paths
    set(DEPLOY_PATH_RESTRICTION ".*")
endif()

foreach(BUILD_TARGET_PATH ${BUILD_TARGET_PATHS})
    get_filename_component(BUILD_TARGET_DIR ${BUILD_TARGET_PATH} DIRECTORY)
    list(APPEND BUILD_TARGET_DIRS ${BUILD_TARGET_DIR})
endforeach()

set(SEARCH_PATH ${BUILD_TARGET_DIRS} ${VORTEX_DIR} $ENV{PATH} $ENV{LD_LIBRARY_PATH})
file(GET_RUNTIME_DEPENDENCIES
    RESOLVED_DEPENDENCIES_VAR FOUND
    UNRESOLVED_DEPENDENCIES_VAR MISSING
    CONFLICTING_DEPENDENCIES_PREFIX CONFLICT
    MODULES ${BUILD_TARGET_PATHS}
    DIRECTORIES ${SEARCH_PATH}
    PRE_EXCLUDE_REGEXES "^api-ms-win-" "^python" "^vcruntime"
    POST_INCLUDE_REGEXES "^/usr/local/"
    POST_EXCLUDE_REGEXES "^[A-Za-z]:[\\/](WINDOWS|Windows)[\\/]" "^/usr/" "^/lib/" "^/lib64/"
)

# report any missing dependencies
foreach(DEP ${MISSING})
    message(STATUS "Missing:  ${DEP}")
endforeach()

# handle conflicts
foreach(DEP ${CONFLICT_FILENAMES})
    # compute the reference checksum
    list(POP_FRONT CONFLICT_${DEP} FIRST)
    file(SHA256 ${FIRST} REF_CHECKSUM)

    # check that all other checksums match
    set(SUCCESS TRUE)
    foreach(PATH ${CONFLICT_${DEP}})
        file(SHA256 ${PATH} THIS_CHECKSUM)
        if(NOT ${REF_CHECKSUM} STREQUAL ${THIS_CHECKSUM})
            set(SUCCESS FALSE)
            break()
        endif()
    endforeach()

    if(SUCCESS)
        # add to found list
        list(APPEND FOUND ${FIRST})
    else()
        # report problems
        message(ERROR "Conflict: ${DEP} -> ${FIRST};${CONFLICT_${DEP}}")
    endif()
endforeach()

# clean up the list
list(SORT FOUND COMPARE FILE_BASENAME)

set(LOCAL_LIBS ${PACKAGE_TARGET_PATHS})

# populate local dependencies
foreach(DEP ${FOUND})
    file(TO_CMAKE_PATH ${DEP} DEP)
    if(DEP MATCHES ${DEPLOY_PATH_RESTRICTION})
        # resolve symbolic links so that actual binary is copied
        file(REAL_PATH ${DEP} SRC)
        if(NOT DEP STREQUAL SRC)
            message(STATUS "Deploy:   ${DEP} (${SRC})")
        else()
            message(STATUS "Deploy:   ${SRC}")
        endif()
        file(COPY ${SRC} DESTINATION ${PACKAGE_ROOT})

        list(APPEND LOCAL_LIBS ${DEP})

    else()
        message(STATUS "Skip:     ${DEP}")
    endif()
endforeach()

if(UNIX)

    # rewrite dependency paths for local copies
    foreach(LIB ${LOCAL_LIBS})
        file(REAL_PATH ${LIB} SRC)
        get_filename_component(LIB_NAME ${SRC} NAME)

        # query dependencies
        execute_process(
            COMMAND patchelf --print-needed ${LIB} OUTPUT_VARIABLE DEPS
            COMMAND_ERROR_IS_FATAL ANY
        )

        set(FIX_RUNPATH FALSE)

        # check to see if dependencies include a locally copied library
        foreach(SUBLIB ${LOCAL_LIBS})
            get_filename_component(SUBLIB_NAME ${SUBLIB} NAME)
            string(FIND "${DEPS}" ${SUBLIB_NAME} IDX)

            if(NOT(IDX EQUAL -1))
                file(REAL_PATH ${SUBLIB} SRC)
                get_filename_component(SRC_NAME ${SRC} NAME)

                if((NOT SUBLIB STREQUAL SRC_NAME))
                    # change library name to local library
                    execute_process(
                        COMMAND patchelf --replace-needed ${SUBLIB_NAME} ${SRC_NAME} ${PACKAGE_ROOT}/${LIB_NAME}
                        COMMAND_ERROR_IS_FATAL ANY
                    )
                endif()

                # mark for using local runpath later
                set(FIX_RUNPATH TRUE)
            endif()
        endforeach()

        if(FIX_RUNPATH)
            # set local runpath since local dependencies were deployed
            execute_process(
                COMMAND patchelf --set-rpath "$ORIGIN" ${PACKAGE_ROOT}/${LIB_NAME}
                COMMAND_ERROR_IS_FATAL ANY
            )
        endif()

    endforeach()

endif()
