#include <cuda_runtime.h>
#include <iostream>

#define checkCudaDriver(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* msg; \
        cuGetErrorString(err, &msg); \
        std::cerr << "CUDA Driver error: " << msg << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCudaRuntime(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
