#include "compiler.cuh"
#include "kernels.cuh"
#include "omp.h"

#define PTX_PATH "/arf/home/delbek/ptx/kernel.ptx"

int main()
{
    float *d_a, *d_b, *d_c;
    float* h_c;
    float correctSum;
    float sum;
    const int N = 1000000;
    h_c = new float[N];
    const int STAGE_COUNT = 4;

    checkCudaRuntime(cudaMalloc(&d_a, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_b, sizeof(float) * N));
    checkCudaRuntime(cudaMalloc(&d_c, sizeof(float) * N));

    checkCudaRuntime(cudaMemset(d_a, rand() % N, sizeof(float) * N));
    checkCudaRuntime(cudaMemset(d_b, rand() % N, sizeof(float) * N));
    checkCudaRuntime(cudaDeviceSynchronize());

    double start, end;
    int blockSize;
    int gridSize;

    auto noSharedMemSizeFunc = [](int blockSize) -> size_t {
        return 0;
    };

    // --- Benchmark vecAddNaive (no shared memory) ---
    start = omp_get_wtime();
    checkCudaRuntime(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize, 
                             vecAddNaive, noSharedMemSizeFunc));
    vecAddNaive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Naive: %f seconds (grid=%d, block=%d)\n", end - start, gridSize, blockSize);
    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
    correctSum = 0;
    for (int i = 0; i < N; ++i)
    {
        correctSum += h_c[i];
    }
    std::cout << "correctSum: " << correctSum << std::endl;

    // --- Benchmark vecAddUnrolledBy4 (no shared memory) ---
    start = omp_get_wtime();
    checkCudaRuntime(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize, 
                             vecAddUnrolledBy4, noSharedMemSizeFunc));
    vecAddUnrolledBy4<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Unrolled by 4: %f seconds (grid=%d, block=%d)\n", end - start, gridSize, blockSize);
    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
    sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += h_c[i];
    }
    std::cout << "sum: " << sum << std::endl;
    if (sum != correctSum)
    {
        printf("Error in vecAddUnrolledBy4\n");
    }
    
    // --- Benchmark vecAddUnrolledBy4ILPMaximization (no shared memory) ---
    start = omp_get_wtime();
    checkCudaRuntime(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize, 
                                vecAddUnrolledBy4ILPMaximization, noSharedMemSizeFunc));
    vecAddUnrolledBy4ILPMaximization<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Unrolled by 4 ILP Maximization: %f seconds (grid=%d, block=%d)\n", end - start, gridSize, blockSize);
    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
    sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += h_c[i];
    }
    std::cout << "sum: " << sum << std::endl;
    if (sum != correctSum)
    {
        printf("Error in vecAddUnrolledBy4ILPMaximization\n");
    }

    // --- Benchmark vecAddUnrolledBy4Pipelined (uses shared memory) ---
    auto sharedMemSizeFunc = [](int bSize) -> size_t {
        return STAGE_COUNT * bSize * 8 * sizeof(float);
    };
    start = omp_get_wtime();
    checkCudaRuntime(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&gridSize, &blockSize, 
                             vecAddUnrolledBy4Pipelined<STAGE_COUNT>, sharedMemSizeFunc));
    size_t sharedSize = sharedMemSizeFunc(blockSize);
    vecAddUnrolledBy4Pipelined<STAGE_COUNT><<<gridSize, blockSize, sharedSize>>>(d_a, d_b, d_c, N);
    checkCudaRuntime(cudaDeviceSynchronize());
    end = omp_get_wtime();
    printf("Unrolled by 4 %d-stage Pipelined: %f seconds (grid=%d, block=%d, shared mem=%zu bytes)\n", 
           STAGE_COUNT, end - start, gridSize, blockSize, sharedSize);
    checkCudaRuntime(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
    sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += h_c[i];
    }
    std::cout << "sum: " << sum << std::endl;
    if (sum != correctSum)
    {
        printf("Error in vecAddUnrolledBy4Pipelined\n");
    }

    checkCudaRuntime(cudaFree(d_a));
    checkCudaRuntime(cudaFree(d_b));
    checkCudaRuntime(cudaFree(d_c));
    delete[] h_c;
    return 0;
}
