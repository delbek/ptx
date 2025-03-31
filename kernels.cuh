#include "gpuHelpers.cuh"

__global__ void vecAddNaive(float* a, float* b, float* c, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vecAddUnrolledBy4(float* a, float* b, float* c, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = idx * 4; i <= (N - 4); i += (gridDim.x * blockDim.x * 4))
    {
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
        c[i + 2] = a[i + 2] + b[i + 2];
        c[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void vecAddUnrolledBy4ILPMaximization(float* a, float* b, float* c, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = idx * 4; i <= (N - 4); i += (gridDim.x * blockDim.x * 4))
    {
        float a0 = a[i];
        float a1 = a[i + 1];
        float a2 = a[i + 2];
        float a3 = a[i + 3];

        float b0 = b[i];
        float b1 = b[i + 1];
        float b2 = b[i + 2];
        float b3 = b[i + 3];

        c[i] = a0 + b0;
        c[i + 1] = a1 + b1;
        c[i + 2] = a2 + b2;
        c[i + 3] = a3 + b3;
    }
}

__global__ void vecAddUnrolledBy4Vectorized(float* a, float* b, float* c, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = idx * 4; i <= (N - 4); i += (gridDim.x * blockDim.x * 4))
    {
        float4 a0 = *reinterpret_cast<float4*>(a + i);
        float4 b0 = *reinterpret_cast<float4*>(b + i);
        float c0 = a0.x + b0.x;
        float c1 = a0.y + b0.y;
        float c2 = a0.z + b0.z;
        float c3 = a0.w + b0.w;
        *reinterpret_cast<float4*>(c + i) = make_float4(c0, c1, c2, c3);
    }
}

template <int STAGE_COUNT>
__global__ void vecAddUnrolledBy4Pipelined(float* a, float* b, float* c, int N)
{
    static_assert(STAGE_COUNT > 1, "STAGE_COUNT must be greater than 1");
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int noThreads = gridDim.x * blockDim.x;
    extern __shared__ float shared[]; // size: STAGE_COUNT * 8 * blockDim.x
    float* aPtr = reinterpret_cast<float*>(shared); // size: STAGE_COUNT * 4 * blockDim.x
    float* bPtr = reinterpret_cast<float*>(shared + STAGE_COUNT * 4 * blockDim.x);  // size: STAGE_COUNT * 4 * blockDim.x
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (unsigned i = 0; i < STAGE_COUNT; ++i)
    {
        pipe.producer_acquire();
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
        {
            if (idx + j * noThreads < N)
            {
                cuda::memcpy_async(&aPtr[i * 4 * blockDim.x + j * blockDim.x + threadIdx.x], &a[idx + j * noThreads], sizeof(float), pipe);
                cuda::memcpy_async(&bPtr[i * 4 * blockDim.x + j * blockDim.x + threadIdx.x], &b[idx + j * noThreads], sizeof(float), pipe);
            }
        }
        pipe.producer_commit();
    }

    int stage = 0;
    for (; idx < N; idx += noThreads * 4)
    {
        cuda::pipeline_consumer_wait_prior<STAGE_COUNT - 1>(pipe);
        float valA1 = aPtr[stage * 4 * blockDim.x + 0 * blockDim.x + threadIdx.x];
        float valB1 = bPtr[stage * 4 * blockDim.x + 0 * blockDim.x + threadIdx.x];
        float valA2 = aPtr[stage * 4 * blockDim.x + 1 * blockDim.x + threadIdx.x];
        float valB2 = bPtr[stage * 4 * blockDim.x + 1 * blockDim.x + threadIdx.x];
        float valA3 = aPtr[stage * 4 * blockDim.x + 2 * blockDim.x + threadIdx.x];
        float valB3 = bPtr[stage * 4 * blockDim.x + 2 * blockDim.x + threadIdx.x];
        float valA4 = aPtr[stage * 4 * blockDim.x + 3 * blockDim.x + threadIdx.x];
        float valB4 = bPtr[stage * 4 * blockDim.x + 3 * blockDim.x + threadIdx.x];
        pipe.consumer_release();

        if (idx < N)
        {
            c[idx] = valA1 + valB1;
        }
        if (idx + 1 * noThreads < N)
        {
            c[idx + 1 * noThreads] = valA2 + valB2;
        }
        if (idx + 2 * noThreads < N)
        {
            c[idx + 2 * noThreads] = valA3 + valB3;
        }
        if (idx + 3 * noThreads < N)
        {
            c[idx + 3 * noThreads] = valA4 + valB4;
        }

        pipe.producer_acquire();
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
        {
            if (idx + j * noThreads < N)
            {
                cuda::memcpy_async(&aPtr[stage * 4 * blockDim.x + j * blockDim.x + threadIdx.x], &a[idx + j * noThreads], sizeof(float), pipe);
                cuda::memcpy_async(&bPtr[stage * 4 * blockDim.x + j * blockDim.x + threadIdx.x], &b[idx + j * noThreads], sizeof(float), pipe);
            }
        }
        pipe.producer_commit();
        stage = (stage + 1) % STAGE_COUNT;
    }
}

__global__ void euclideanDistance(float* xCoords, float* yCoords, float* dists, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned i = idx; i < N; i += (gridDim.x * blockDim.x))
    {
        float x = xCoords[i];
        float y = yCoords[i];
        dists[i] = sqrt(x * x + y * y);
    }
}

__global__ void euclideanDistanceApprox(float* xCoords, float* yCoords, float* dists, int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = idx; i < N; i += (gridDim.x * blockDim.x))
    {
        float x = xCoords[i];
        float y = yCoords[i];
        asm volatile
        (
            "sqrt.approx.f32 %0, %1;"
            : "=f"(dists[i])
            : "f"(x * x + y * y)
        );
    }
}
