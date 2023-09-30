// 规约是一类并行算法，对传入的N个数据，使用一个二元的符合结合律的操作符，生成1个结果。这类操作包括取最小、取最大、求和、平方和、逻辑与/或、向量点积。
// 规约也是其他高级算法中重要的基础算法。

#include <stdio.h>
#include <math.h>

#define N 100000000
#define BLOCK_SIZE 256
#define GRID_SIZE 32

__managed__ int source[N];
__managed__ int gpu_result[1] = {0};

// 问题:
// source[N]:  1 + 2 + 3 + 4 + ............... + N   
// cpu: for loop 
// gpu: 1 + 2 + 3 + 4 + ............... + N
// 解决: 
// 每个线程处理 2 个数据
// thread id: t0: source[0] + source[N/2]
// 例如: 设 N = 8
// thread id step 0:  tid0: source[0] + source[4] -> source[0]
//                    tid1: source[1] + source[5] -> source[1]
//                    tid2: source[2] + source[6] -> source[2]
//                    tid4: source[4] + source[7] -> source[3]
//           set N = N/2 
//           step 1:  tid0: source[0] + source[2] -> source[0]
//                    tid1: source[1] + source[3] -> source[1]
//           set N = N/2 
//           step 2:  tid0: source[0] + source[1] -> source[0]
// 防止线程不够: 
// thread id: blockDim.x * blockIdx.x + threadIdx.x + step * blockDim.x * GridDim.x  ==> 如果有8个线程要处理32个数据: 0 8 16 24
// thread 0: source[0, 8, 16, 24] sum -> shared memory

__global__ void sum_gpu(int *in, int count, int *out)
{
    __shared__ int shared_memory[BLOCK_SIZE];
    // grid_loop
    // 防止线程不够: 
    int shared_tmp=0;
    for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += blockDim.x * gridDim.x)
    {
        shared_tmp +=in[idx];
    }
    shared_memory[threadIdx.x] = shared_tmp;
    __syncthreads();

    int tmp = 0;
    for(int total_threads = BLOCK_SIZE/2; total_threads>=1; total_threads/=2)
    {
        if(threadIdx.x < total_threads)
        {
            tmp = shared_memory[threadIdx.x] + shared_memory[threadIdx.x + total_threads]; 
        }
        __syncthreads();  // 对一个变量的读写之间建议加一个同步, 此处不加也可以
        if(threadIdx.x < total_threads)
        {
            shared_memory[threadIdx.x] = tmp;
        }
    }
    // block_sum -> share memory[0]
    if(blockIdx.x * blockDim.x < count)
    {
        if(threadIdx.x == 0)
        {
            // block 0 和 block 1 可能同时进行写入
            // out[0] += shared_memory[0];

            // 原子操作: 多个线程同时进行操作, 保证一个线程的内存空间读, 写, 修改不受其他线程的任何操作的影响
            atomicAdd(out, shared_memory[0]);
            /*
            为了获得高的内存带宽，共享内存在物理上被分为 32 个（刚好等于一个线程束中的线程数目，即内建变量 warpSize 的值）同样宽度的、能被同时访问的内存 bank。
            例如: 
                1. 共享内存数组中连续的 128 字节的内容分摊到 32 个 bank 的某一层中，每个 bank 负责 4 字节的内容。
                2. 对一个长度为 128 的单精度浮点数变量的共享内存数组来说，第 0-31 个数组元素依次对应到 32 个 bank 的第一层；
                第 32-63 个数组元素依次对应到 32 个 bank 的第二层；... 
            只要同一线程束内的多个线程不同时访问同一个 bank 中不同层的数据，该线程束对共享内存的访问就只需要一次内存事务（memory transaction）。
            当同一线程束内的多个线程试图访问同一个 bank 中不同层的数据时，就会发生 bank 冲突。
            */
        }
    }
}



int main()
{
    int cpu_result =0;

    printf("Init input source[N]\n");
    for(int i =0; i<N; i++)
    {
        source[i] = rand()%10;
    }

    cudaEvent_t start, stop_cpu, stop_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    for(int i = 0; i<20; i++)
    {
        gpu_result[0] = 0;
        sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    for(int i =0; i<N; i++)
    {
        cpu_result +=source[i];
    }

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);

    printf("CPU time: %.2f\nGPU time: %.2f\n", time_cpu, time_gpu/20);
    printf("Result: %s\nGPU_result: %d;\nCPU_result: %d;\n", (gpu_result[0] == cpu_result)?"Pass":"Error", gpu_result[0], cpu_result);
    
    return 0;
}