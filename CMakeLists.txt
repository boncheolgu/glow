
cmake_minimum_required(VERSION 3.7)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" "${CMAKE_CURRENT_SOURCE_DIR}/external/flatbuffers/CMake")

project(Glow C CXX)
enable_testing()

option(GLOW_WITH_CPU "Build the LLVM-based JIT CPU backend" ON)
option(GLOW_WITH_OPENCL "Build the OpenCL backend" ON)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Export a JSON file with the compilation commands that external tools can use
# to analyze the source code of the project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GlowDefaults)
include(GlowTestSupport)
include(SanitizerSupport)
include(CoverageSupport)
include(DoxygenSupport)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/CMakeGraphVizOptions.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/CMakeGraphVizOptions.cmake COPYONLY)

set(GLOW_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GLOW_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bin)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include_directories(${GLOW_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB_RECURSE header_files include/*.h tools/*.h src/*.h)
add_custom_target(CollectHeaders SOURCES ${header_files})

find_package(PNG)

if(GLOW_WITH_CPU)
  add_definitions(-DGLOW_WITH_CPU=1)
endif ()

if (GLOW_WITH_OPENCL)
  add_definitions(-DGLOW_WITH_OPENCL=1)
  find_package(OpenCL REQUIRED)
endif ()

# Prefer LLVM 6.
find_package(LLVM 6 CONFIG)
# If LLVM 6 not found, try build with whatever is available.
if (NOT LLVM_FOUND)
  find_package(LLVM CONFIG)
endif()
if (NOT LLVM_FOUND)
  message(FATAL_ERROR "Could not find LLVM. Build LLVM manually and configure the project with -DCMAKE_PREFIX_PATH=/path/to/llvm/install")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

if(NOT EXISTS "${GLOW_SOURCE_DIR}/tests/googletest/googletest")
  message(FATAL_ERROR "No git submodules. Run: git submodule update --init --recursive")
endif()

if(NOT EXISTS "${GLOW_SOURCE_DIR}/external/flatbuffers/include")
  message(FATAL_ERROR "No git submodules. Run: git submodule update --init --recursive")
endif()

add_subdirectory(tests/googletest/)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
add_subdirectory(external/flatbuffers/)
set(FLATBUFFERS_INCLUDE_DIR "${GLOW_SOURCE_DIR}/external/flatbuffers/include")

add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(tests)

add_custom_target(dependency_graph
                  "${CMAKE_COMMAND}" "--graphviz=dependency_graph" .
                  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

# Fetch the dependencies for all the tests.
get_property(GLOW_TEST_DEPENDS GLOBAL PROPERTY GLOW_TEST_DEPENDS)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}
                    DEPENDS ${GLOW_TEST_DEPENDS} USES_TERMINAL)
