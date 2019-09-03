# - Find PMDK
# Find the PMDK library and includes
#
# PMDK_INCLUDE_DIRS - where to find numa.h, etc.
# PMDK_LIBRARIES - List of libraries when using NUMA.
# PMDK_FOUND - True if NUMA found.

set(PMDK_ROOT_DIR "/usr/local")

find_path(PMDK_INCLUDE_DIRS
NAMES libpmem.h
HINTS "${PMDK_ROOT_DIR}/include")

find_library(PMDK_LIBRARIES
NAMES pmem
HINTS "${PMDK_ROOT_DIR}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMDK DEFAULT_MSG PMDK_LIBRARIES PMDK_INCLUDE_DIRS)

mark_as_advanced(
    PMDK_LIBRARIES
    PMDK_INCLUDE_DIRS)

if(PMDK_FOUND AND NOT (TARGET PMDK::PMDK))
add_library (PMDK::PMDK UNKNOWN IMPORTED)
set_target_properties(PMDK::PMDK
 PROPERTIES
   IMPORTED_LOCATION ${PMDK_LIBRARIES}
   INTERFACE_INCLUDE_DIRECTORIES ${PMDK_INCLUDE_DIRS})
endif()
