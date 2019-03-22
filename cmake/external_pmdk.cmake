# Enable ExternalProject Cmake module
include(ExternalProject)

#--------------------------------------------------------------------------------
# Download PMDK
#--------------------------------------------------------------------------------

set(PMDK_GIT_REPO_URL https://github.com/pmem/pmdk)
set(PMDK_GIT_LABEL 1.5)

set(PMDK_INSTALL_DIR ${EXTERNAL_PROJECTS_ROOT}/pmdk)
set(PMDK_LIB_DIR ${PMDK_INSTALL_DIR}/lib)
set(PMDK_INCLUDE_DIR ${PMDK_INSTALL_DIR}/include)

set(PMEM_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}pmem${CMAKE_SHARED_LIBRARY_SUFFIX})
set(PMEMOBJ_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}pmemobj${CMAKE_SHARED_LIBRARY_SUFFIX})

find_program(MAKE_EXE NAMES gmake nmake make)

ExternalProject_Add(
    ext_pmdk
    PREFIX pmdk
    GIT_REPOSITORY ${PMDK_GIT_REPO_URL}
    GIT_TAG ${PMDK_GIT_LABEL}
    # Disable install step
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE_EXE} CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} -j all
    # See the comments in `external_mkldnn.cmake` for what the heck this is doing
    PATCH_COMMAND patch -p1 --forward --reject-file=- -i ${CMAKE_SOURCE_DIR}/cmake/pmdk.patch || exit 0
    INSTALL_COMMAND ${MAKE_EXE} install prefix=${PMDK_INSTALL_DIR}

    BUILD_IN_SOURCE TRUE

    # Setup directories
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/pmdk/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/pmdk/stamp"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/pmdk/src"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/pmdk"
    EXCLUDE_FROM_ALL TRUE
)

#------------------------------------------------------------------------------

# ExternalProject_Get_Property(ext_pmdk SOURCE_DIR)
# ExternalProject_Get_Property(ext_pmdk INSTALL_DIR)
add_library(libpmem SHARED IMPORTED)
add_dependencies(libpmem ext_pmdk)

set_target_properties(libpmem
    PROPERTIES
        IMPORTED_LOCATION ${PMDK_LIB_DIR}/${PMEM_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${PMDK_INCLUDE_DIR}
        INSTALL_RPATH ${CMAKE_INSTALL_RPATH}
    )

add_library(libpmemobj SHARED IMPORTED)
add_dependencies(libpmemobj PRIVATE libpmem ext_pmdk)
set_target_properties(libpmemobj
    PROPERTIES
        IMPORTED_LOCATION ${PMDK_LIB_DIR}/${PMEMOBJ_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${PMDK_INCLUDE_DIR}
    )

install(
    FILES
        # This is really hacky - find a better way to do this!
        ${PMDK_LIB_DIR}/${PMEMOBJ_LIB}
        ${PMDK_LIB_DIR}/${PMEMOBJ_LIB}.1
        ${PMDK_LIB_DIR}/${PMEMOBJ_LIB}.1.0.0
        ${PMDK_LIB_DIR}/${PMEM_LIB}
        ${PMDK_LIB_DIR}/${PMEM_LIB}.1
        ${PMDK_LIB_DIR}/${PMEM_LIB}.1.0.0
    DESTINATION
        ${NGRAPH_INSTALL_LIB}
)

install(
    DIRECTORY
        ${PMDK_INCLUDE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/pmdk"
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

