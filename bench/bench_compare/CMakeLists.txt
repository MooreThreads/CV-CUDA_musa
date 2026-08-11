cmake_minimum_required(VERSION 3.16)
project(opencv_cvcuda_compare LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BACKEND "MUSA" CACHE STRING "GPU backend: MUSA or CUDA")
set_property(CACHE BACKEND PROPERTY STRINGS MUSA CUDA)
string(TOUPPER "${BACKEND}" BACKEND)
if(NOT BACKEND STREQUAL "MUSA" AND NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "BACKEND must be MUSA or CUDA, got '${BACKEND}'")
endif()

set(CVCUDA_ROOT "" CACHE PATH "CV-CUDA source root")
set(CVCUDA_BUILD "" CACHE PATH "CV-CUDA build root")

if(NOT CVCUDA_ROOT OR NOT EXISTS "${CVCUDA_ROOT}")
    message(FATAL_ERROR "CVCUDA_ROOT must point to an existing CV-CUDA source tree")
endif()
if(NOT CVCUDA_BUILD OR NOT EXISTS "${CVCUDA_BUILD}")
    message(FATAL_ERROR "CVCUDA_BUILD must point to an existing CV-CUDA build tree")
endif()

set(CVCUDA_LIB "${CVCUDA_BUILD}/lib/libcvcuda.so.0.16.0")
set(NVCV_TYPES_LIB "${CVCUDA_BUILD}/lib/libnvcv_types.so.0.16.0")
if(NOT EXISTS "${CVCUDA_LIB}")
    message(FATAL_ERROR "CV-CUDA library not found: ${CVCUDA_LIB}")
endif()
if(NOT EXISTS "${NVCV_TYPES_LIB}")
    message(FATAL_ERROR "NVCV types library not found: ${NVCV_TYPES_LIB}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_executable(opencv_cvcuda_benchmark_compare opencv_cvcuda_benchmark_compare.cpp)
target_include_directories(opencv_cvcuda_benchmark_compare PRIVATE
    ${CVCUDA_ROOT}/bench
    ${CVCUDA_BUILD}/src/cvcuda/include
    ${CVCUDA_ROOT}/src/nvcv/src/include
    ${CVCUDA_BUILD}/src/nvcv/src/include
)
target_link_directories(opencv_cvcuda_benchmark_compare PRIVATE
    ${CVCUDA_BUILD}/lib
)
target_link_libraries(opencv_cvcuda_benchmark_compare PRIVATE
    ${OpenCV_LIBS}
    ${CVCUDA_LIB}
    ${NVCV_TYPES_LIB}
)

set(BACKEND_RPATH "${CVCUDA_BUILD}/lib")
if(BACKEND STREQUAL "MUSA")
    set(MUSA_ROOT "/usr/local/musa" CACHE PATH "MUSA toolkit root")
    find_path(MUSA_INCLUDE_DIR musa_runtime.h
        HINTS ${MUSA_ROOT}
        PATH_SUFFIXES include
    )
    find_library(MUSA_RUNTIME_LIBRARY
        NAMES musart libmusart.so
        HINTS ${MUSA_ROOT}
        PATH_SUFFIXES lib lib64
    )
    if(NOT MUSA_INCLUDE_DIR)
        message(FATAL_ERROR "MUSA include directory not found. Set MUSA_ROOT or MUSA_INCLUDE_DIR.")
    endif()
    if(NOT MUSA_RUNTIME_LIBRARY)
        message(FATAL_ERROR "MUSA runtime library musart not found. Set MUSA_ROOT or MUSA_RUNTIME_LIBRARY.")
    endif()

    target_compile_definitions(opencv_cvcuda_benchmark_compare PRIVATE NVCV_USE_MUSA=1)
    target_include_directories(opencv_cvcuda_benchmark_compare PRIVATE
        ${CVCUDA_ROOT}/tests/common_musa
        ${CVCUDA_ROOT}/src/cvcuda/include_musa
        ${CVCUDA_ROOT}/src/cvcuda/include
        ${MUSA_INCLUDE_DIR}
    )
    target_link_libraries(opencv_cvcuda_benchmark_compare PRIVATE ${MUSA_RUNTIME_LIBRARY})
    get_filename_component(MUSA_RUNTIME_LIBRARY_DIR "${MUSA_RUNTIME_LIBRARY}" DIRECTORY)
    list(APPEND BACKEND_RPATH "${MUSA_RUNTIME_LIBRARY_DIR}")
elseif(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(opencv_cvcuda_benchmark_compare PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_directories(opencv_cvcuda_benchmark_compare PRIVATE
        ${CUDAToolkit_LIBRARY_DIR}
    )
    target_link_libraries(opencv_cvcuda_benchmark_compare PRIVATE CUDA::cudart)
    list(APPEND BACKEND_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endif()

set_target_properties(opencv_cvcuda_benchmark_compare PROPERTIES
    BUILD_RPATH "${BACKEND_RPATH}"
)

message(STATUS "opencv-cvcuda-compare backend: ${BACKEND}")
message(STATUS "CV-CUDA source root: ${CVCUDA_ROOT}")
message(STATUS "CV-CUDA build root: ${CVCUDA_BUILD}")
