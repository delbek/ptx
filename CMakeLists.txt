cmake_minimum_required(VERSION 3.20.2)
project(ptx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Cuda
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    enable_language(CUDA)
    add_definitions(-DGPU_AVAILABLE)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xcompiler ${OpenMP_CXX_FLAGS}")
    find_package(CUDAToolkit REQUIRED)
endif()

# MPI
find_package(MPI QUIET)
if(MPI_FOUND)
    add_definitions(-DMPI_AVAILABLE)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

include_directories(.)

add_executable(ptx
        main.cu
        compiler.cu
        gpuHelpers.cuh
        compiler.cuh
        kernels.cuh
)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(ptx PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_link_libraries(ptx OpenMP::OpenMP_CXX)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(ptx
            cuda
            CUDA::cudart
            ${CUDA_LIBRARIES}
            nvrtc
    )
endif()

if(MPI_FOUND)
    target_link_libraries(ptx
            MPI::MPI_CXX
    )
endif()

target_compile_options(ptx PRIVATE -O3)
target_compile_options(ptx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --generate-line-info
        >)
