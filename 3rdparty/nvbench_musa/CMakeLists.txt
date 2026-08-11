set(NVBENCH_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../nvbench" CACHE PATH "NVBench source root" FORCE)
include("${NVBENCH_SOURCE_ROOT}/CMakeLists_musa.txt")

add_library(cvcuda_nvbench_main STATIC ${nvbench_main_OBJECTS})
target_link_libraries(cvcuda_nvbench_main PUBLIC nvbench)
add_library(cvcuda::nvbench::main ALIAS cvcuda_nvbench_main)
