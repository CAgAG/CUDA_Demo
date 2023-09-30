#include <stdio.h>
#include <cooperative_groups.h>  // 协作组
using namespace cooperative_groups;

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 100;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const unsigned FULL_MASK = 0xffffffff;

void timing(const real *d_x, const int method);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    cudaMalloc(&d_x, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);

    printf("\nusing syncwarp:\n");
    timing(d_x, 0);
    printf("\nusing shfl:\n");
    timing(d_x, 1);
    printf("\nusing cooperative group:\n");
    timing(d_x, 2);

    free(h_x);
    cudaFree(d_x);
    return 0;
}

void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
        // 核函数中定义的线程块在执行时将被分配到还没有完全占满的 SM 中。
        // 一个线程块不会被分配到不同的 SM 中，而总是在一个 SM 中，但一个 SM 可以有一个或多个线程块。
        // 一个 SM 以 32 个线程为单位产生、管理、调度、执行线程。这样的 32 个线程称为一个线程束。
        // 一个 SM 可以处理一个或多个线程块。一个线程块又可分为若干个线程束。
        // 例如，一个 128 线程的线程块将被分为 4 个线程束，其中每个线程束包含 32 个具有连续线程号的线程。这样的划分对目前所有的 GPU 架构都是成立的。
        // 当所涉及的线程都在一个线程束内时，可以将线程块同步函数 __syncthreads 换成一个更加廉价的线程束同步函数 __syncwarp。
    }

    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}

void __global__ reduce_shfl(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}

real reduce(const real *d_x, const int method)
{
    const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int smem = sizeof(real) * BLOCK_SIZE;

    real h_y[1] = {0};
    real *d_y;
    cudaMalloc(&d_y, sizeof(real));
    cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice);

    switch (method)
    {
        case 0:
            reduce_syncwarp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 1:
            reduce_shfl<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        case 2:
            reduce_cp<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y, N);
            break;
        default:
            printf("Wrong method.\n");
            exit(1);
    }

    cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_y);

    return h_y[0];
}

void timing(const real *d_x, const int method)
{
    real sum = 0;
    
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

        sum = reduce(d_x, method); 

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Time = %g ms.\n", elapsed_time);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("sum = %f.\n", sum);
}


