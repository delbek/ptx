#include "compiler.cuh"

#define PTX_PATH "cuda.ptx"

int main()
{
    CUfunction kernel = compilePTX(PTX_PATH, "vecAdd");

    float* d_a;
    float* d_b;
    float* d_c;
    unsigned N = 1024;
    checkCudaRuntime(cudaMalloc(&d_a, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_b, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_c, sizeof(float) * N));

    void* args[] = { &d_a, &d_b, &d_c, &N };
    checkCudaDriver(cuLaunchKernel(kernel,
        1, 1, 1,
        1024, 1, 1,
        0,
        0,
        args,
        0
    ));

    float* h_c = (float*)malloc(sizeof(float) * N);
    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(d_a));
    checkCudaRuntime(cudaFree(d_b));
    checkCudaRuntime(cudaFree(d_c));
    delete[] h_c;
    
    destroyContext();
    return 0;
}
