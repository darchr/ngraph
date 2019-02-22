# Enable ExternalProject Cmake module
include(ExternalProject)

#--------------------------------------------------------------------------------
# Download PMDK
#--------------------------------------------------------------------------------

set(PMDK_GIT_REPO_URL https://github.com/pmem/pmdk)
set(PMDK_GIT_LABEL 1.5)

set(PMDK_INSTALL_DIR ${EXTERNAL_PROJECTS_ROOT}/pmdk)

find_program(MAKE_EXE NAMES gmake nmake make)

ExternalProject_Add(
    ext_pmdk
    PREFIX pmdk
    GIT_REPOSITORY ${PMDK_GIT_REPO_URL}
    GIT_TAG ${PMDK_GIT_LABEL}
    # Disable install step
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE_EXE} -j all
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

ExternalProject_Get_Property(ext_pmdk SOURCE_DIR)
ExternalProject_Get_Property(ext_pmdk INSTALL_DIR)

add_library(libpmemobj INTERFACE)
target_include_directories(libpmemobj SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_link_libraries(libpmemobj INTERFACE ${INSTALL_DIR}/lib/libpmemobj.so)
link_directories(${INSTALL_DIR}/lib)
add_dependencies(libpmemobj ext_pmdk)
