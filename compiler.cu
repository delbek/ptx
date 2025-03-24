#include "compiler.cuh"

std::string loadPTX(std::string filename) 
{
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

CUfunction compilePTX(std::string filename, std::string kernelName)
{
    std::string ptxSource = loadPTX(filename);
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, ptxSource.c_str(), filename.c_str(), 0, nullptr, nullptr);
    nvrtcCompileProgram(prog, 0, nullptr);

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());

    checkCudaDriver(cuInit(0));
    CUdevice dev;
    checkCudaDriver(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    checkCudaDriver(cuCtxCreate(&ctx, 0, dev));

    CUmodule module;
    checkCudaDriver(cuModuleLoadData(&module, ptx.data()));
    CUfunction kernel;
    checkCudaDriver(cuModuleGetFunction(&kernel, module, kernelName.c_str()));

    checkCudaDriver(cuModuleUnload(module));
    checkCudaDriver(cuCtxDestroy(ctx));
    nvrtcDestroyProgram(&prog);

    return kernel;
}
