#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "gpuHelpers.cuh"

CUmodule module;
CUcontext ctx;

std::string loadPTX(std::string filename)
{
    std::ifstream file(filename);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

CUfunction compilePTX(std::string filename, std::string kernelName)
{
    std::string ptxSource = loadPTX(filename);

    checkCudaDriver(cuInit(0));
    CUdevice device;
    checkCudaDriver(cuDeviceGet(&device, 0));
    checkCudaDriver(cuCtxCreate(&ctx, 0, device));

    char errorLog[8192] = {0};
    CUjit_option options[] = 
    {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_OPTIMIZATION_LEVEL
    };
    void* optionVals[] = 
    {
        errorLog,
        reinterpret_cast<void*>(sizeof(errorLog)),
        reinterpret_cast<void*>(3)
    };

    checkCudaDriver(cuModuleLoadDataEx(&module, ptxSource.c_str(), 3, options, optionVals));

    CUfunction kernel;
    checkCudaDriver(cuModuleGetFunction(&kernel, module, kernelName.c_str()));

    std::cout << "Kernel compiled" << std::endl;
    return kernel;
}

void destroyContext()
{
    checkCudaDriver(cuModuleUnload(module));
    checkCudaDriver(cuCtxDestroy(ctx));
}
