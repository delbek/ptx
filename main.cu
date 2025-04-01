#include "compiler.cuh"
#include "kernels.cuh"
#include "omp.h"

#define KERNELS_DENIZ "/arf/home/delbek/ptx/kernels_deniz.ptx"
#define KERNELS_NVCC "/arf/home/delbek/ptx/kernels_nvcc.ptx"

void initializeDataForAtomicFilter(int* data, int n, float filterRatio)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] = (float)rand() / RAND_MAX < filterRatio ? 1 : -1;
    }
}

void compareAtomicKernels()
{
    int *d_a, *d_b;
    unsigned *d_counter;
    int *h_a;
    unsigned *h_counter;

    unsigned N = 104857600;
    h_a = new int[N];
    h_counter = new unsigned(0);

    srand(time(0));
    initializeDataForAtomicFilter(h_a, N, 0.5f);

    checkCudaRuntime(cudaMalloc(&d_a, sizeof(int) * N));
    checkCudaRuntime(cudaMalloc(&d_b, sizeof(int) * N));
    checkCudaRuntime(cudaMalloc(&d_counter, sizeof(unsigned)));

    int gridSize = 108;
    int blockSize = 1024;
    printf("Launching kernels with Grid Size = %d, Block Size = %d\n", gridSize, blockSize);

    double start, end;

    // Naive Atomic Filter
    CUfunction kernelNaive = compilePTX(KERNELS_NVCC, "_Z12atomicFilterPiS_Pjj");

    checkCudaRuntime(cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_counter, h_counter, sizeof(unsigned), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaDeviceSynchronize());

    start = omp_get_wtime();
    void* args1[] = { &d_a, &d_b, &d_counter, &N };
    checkCudaDriver(cuLaunchKernel(kernelNaive,
        gridSize, 1, 1,      
        blockSize, 1, 1,     
        0,                   
        0,                   
        args1,                
        0                    
    ));
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Naive Atomic Filter time: %f\n", end - start);
    destroyContext();
    // ------------------------------------------------------------

    // Aggregated Atomic Filter
    CUfunction kernelAggregated = compilePTX(KERNELS_NVCC, "_Z26warpAggregatedAtomicFilterPiS_Pjj");

    checkCudaRuntime(cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_counter, h_counter, sizeof(unsigned), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaDeviceSynchronize());

    start = omp_get_wtime();
    void* args2[] = { &d_a, &d_b, &d_counter, &N };
    checkCudaDriver(cuLaunchKernel(kernelAggregated,
        gridSize, 1, 1,      
        blockSize, 1, 1,     
        0,                   
        0,                   
        args2,                
        0                    
    ));
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Aggregated Atomic Filter time: %f\n", end - start);
    destroyContext();
    // ------------------------------------------------------------

    checkCudaRuntime(cudaFree(d_a));
    checkCudaRuntime(cudaFree(d_b));
    checkCudaRuntime(cudaFree(d_counter));

    delete[] h_a;
    delete[] h_counter;
}

int main()
{
    compareAtomicKernels();
    return 0;
}
