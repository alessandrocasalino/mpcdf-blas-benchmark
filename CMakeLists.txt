cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 NEW) # Set policy to handle CUDA_ARCHITECTURES

project(OctopusDevel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Files to compile
set(SRC_FILES
    ${SRC_DIR}/blas_bench.cpp
)

# Add the executable
add_executable(blas_bench ${SRC_FILES})

# Include the header directory
target_include_directories(blas_bench PRIVATE ${SRC_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(blas_bench PRIVATE -g -O0)
    message(STATUS "Building in Debug mode.")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(blas_bench PRIVATE -O3)
    message(STATUS "Building in Release mode.")
endif()

# Check for ROCm/HIP
find_package(HIP)

if(HIP_FOUND)
    message(STATUS "HIP found. Building with ROCm support.")
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    add_definitions(-D__HIP_PLATFORM_HCC__)
    target_compile_options(blas_bench PRIVATE -DHIP_ENABLED --offload-arch=gfx942)

    find_package(rocblas REQUIRED)
    target_link_libraries(blas_bench PRIVATE roc::rocblas hip::device)
else()
    message(STATUS "HIP not found. Building with CUDA support.")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CUDA)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(blas_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-G --compiler-options -std=c++20>")
    else()
        target_compile_options(blas_bench PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -std=c++20>")
    endif()
    set_target_properties(blas_bench PROPERTIES
             CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(blas_bench PRIVATE cuda CUDA::cublas CUDA::cuda_driver)
    set_property(TARGET blas_bench PROPERTY CUDA_ARCHITECTURES "80")
endif()
