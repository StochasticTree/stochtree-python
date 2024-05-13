# Build options
option(USE_DEBUG "Set to ON for Debug mode" ON)
option(USE_SANITIZER "Use santizer flags" OFF)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to CMake 3.20
cmake_minimum_required(VERSION 3.20)

# Define the project
project(pystochtree LANGUAGES C CXX)

# Debug flags
if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()
if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -w -fPIC"
  )
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
endif()

# stochtree-cpp submodule directory
set(STOCHTREECPP_DIR ${PROJECT_SOURCE_DIR}/src/stochtree-cpp)

# stochtree-cpp header file directory
set(StochTreeCpp_HEADER_DIR ${STOCHTREECPP_DIR}/include)

# boost.math header file directory
set(BOOSTMATH_HEADER_DIR ${STOCHTREECPP_DIR}/dependencies/boost_math/include)

# Eigen header file directory
set(EIGEN_HEADER_DIR ${STOCHTREECPP_DIR}/dependencies/eigen)

# Aggregate the source files underpinning the implementation in the C++ library
file(
  GLOB 
  SOURCES 
  ${STOCHTREECPP_DIR}/src/container.cpp
  ${STOCHTREECPP_DIR}/src/cutpoint_candidates.cpp
  ${STOCHTREECPP_DIR}/src/data.cpp
  ${STOCHTREECPP_DIR}/src/io.cpp
  ${STOCHTREECPP_DIR}/src/json11.cpp
  ${STOCHTREECPP_DIR}/src/leaf_model.cpp
  ${STOCHTREECPP_DIR}/src/partition_tracker.cpp
  ${STOCHTREECPP_DIR}/src/random_effects.cpp
  ${STOCHTREECPP_DIR}/src/tree.cpp
)

# Define the C++ source code as a target
add_library(stochtree_objs OBJECT ${SOURCES})

# Include the headers in the source library
target_include_directories(stochtree_objs PRIVATE ${StochTreeCpp_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR})

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

# Add pybind11 module
add_subdirectory(pybind11)
pybind11_add_module(stochtree_cpp src/stochtree.cpp)

# Link to C++ source and headers
target_include_directories(stochtree_cpp PRIVATE ${StochTreeCpp_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR})
target_link_libraries(stochtree_cpp PRIVATE stochtree_objs)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(stochtree_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
