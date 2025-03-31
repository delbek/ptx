#include "compiler.cuh"
#include "kernels.cuh"
#include "omp.h"

#define KERNELS_DENIZ "/arf/home/delbek/ptx/kernels_deniz.ptx"
#define KERNELS_NVCC "/arf/home/delbek/ptx/kernels_nvcc.ptx"

void initializeData(float* data, int n) 
{
    for (int i = 0; i < n; ++i) 
    {
        data[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    CUfunction kernel = compilePTX(KERNELS_DENIZ, "_Z32vecAddUnrolledBy4ILPMaximizationPfS_S_i");

    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;
    int N = 176947200;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    srand(time(0));
    initializeData(h_a, N);
    initializeData(h_b, N);

    checkCudaRuntime(cudaMalloc(&d_a, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_b, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_c, sizeof(float) * N));

    checkCudaRuntime(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));

    checkCudaRuntime(cudaDeviceSynchronize());

    double start, end;
    int blockSize;
    int gridSize;

    auto sharedMemFunc = [](int block_size) -> size_t {
        return 0;
    };

    /*
    checkCudaRuntime(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize,
                             kernel, sharedMemFunc, 0));
    */
    gridSize = 108;
    blockSize = 1024;

    printf("Launching kernel with Grid Size = %d, Block Size = %d\n", gridSize, blockSize);

    start = omp_get_wtime();
    void* args[] = { &d_a, &d_b, &d_c, &N };
    checkCudaDriver(cuLaunchKernel(kernel,
        gridSize, 1, 1,      
        blockSize, 1, 1,     
        0,                   
        0,                   
        args,                
        0                    
    ));
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("vecAddUnrolledBy4ILPMaximization time: %f\n", end - start);

    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(d_a));
    checkCudaRuntime(cudaFree(d_b));
    checkCudaRuntime(cudaFree(d_c));

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    destroyContext();

    return 0;
}
