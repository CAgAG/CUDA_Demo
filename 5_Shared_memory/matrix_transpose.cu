// 合并访存: 一个线程束内的每个线程之间访问的地址需要是连续的, 而不合并访存则是每个线程之间访问的地址是不连续的
// 连续的线程访问连续的数据
//              matrix transpose: 转置存在问题 ==> 读取是连续的，但是写入不连续 ==> 比如 读 b06 b07 b08 (连续) 对应写一列不连续
//                                                     t56 t57 t58  ==> 线程
//               in        b00 b01 b02 | b03 b04 b05 | b06 b07 b08 
//                         b10 b11 b12 | b13 b14 b15 | b16 b17 b18
//                         b20 b21 b22 | b23 b24 b25 | b26 b27 b28
//                         ------------+-------------+------------
//                         b30 b31 b32 | b33 b34 b35 | b36 b37 b38
//                         b40 b41 b41 | b43 b44 b45 | b46 b47 b48
//                         b50 b51 b52 | b53 b54 b55 | b56 b57 b58   block 1, 2 && thread 2,1  ==> 读取t57
//                                                                   
//
//                         
//               out       b00 b10 b20 | b30 b40 b50
//                         b01 b11 b21 | b31 b41 b51
//                         b02 b12 b22 | b32 b42 b52
//                         ------------+------------
//                         b03 b13 b23 | b33 b43 b53
//                         b04 b14 b24 | b34 b44 b54
//                         b05 b15 b25 | b35 b45 b55
//                         ------------+------------
//                         b06 b16 b26 | b36 b46 b56
//                         b07 b17 b27 | b37 b47 b57
//                         b08 b18 b28 | b38 b48 b58                 block 2, 1 && thread 2,1 ==> 写入t57
// shared memory:  此处当作缓存使用, 在shared memory中进行转置
//  t57 read b57 from global memroy to shared memroy
//  t57 read b48 from shared memory
//  t57 write b48 to global memory

# include <stdio.h>
# include <math.h>

#define BLOCK_SIZE 32
#define M 3000
#define N 1000

// 在该问题中，并不是一定要使用二维的网格和线程块，因为矩阵中的数据排列本质上依然是一维的。
__managed__ int matrix[N][M];  // N行M列
__managed__ int gpu_result[M][N];
__managed__ int cpu_result[M][N];

void cpu_matrix_transpose(int in[N][M], int out[M][N])
{
    for(int y = 0; y < N; y++)
    {
        for(int x = 0; x < M; x++)
        {
            out[x][y] = in[y][x];
        }
    }
}

__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N])
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if( x < M && y < N)
    {
        out[x][y] = in[y][x];
    }
}

__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N])
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    // 对于多维数组， x 维度的线程指标 threadIdx.x 是最内层的（变化最快）
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int shared_tmp[BLOCK_SIZE+1][BLOCK_SIZE+1];  // shared_tmp[32] warp

    if(x < M && y < N)
    {
        // 连续的线程读取连续的数据
        shared_tmp[threadIdx.y][threadIdx.x] = in[y][x];
    }
    // 保证数据读取完毕
    __syncthreads();

    /* 以 b57 为例
    in : 读取 b57              ==> 对应 block 1, 2 && thread 2,1
    out: 写入的时候对应的 b48   ==> 对应 block 2, 1 && thread 2,1
    也即 block 内部的 thread 位置没变, 只变了 block 的位置
    in:              out:
       b36 b37 b38       b36 b46 b56      
       b46 b47 b48       b37 b47 b57      
       b56 b57 b58       b38 b48 b58       
    */
    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if(x1 < N && y1 < M)
    {
        // 在shared memory中进行转置
        out[y1][x1] = shared_tmp[threadIdx.x][threadIdx.y];  // 32 bank
    }

}


int main()
{
    for(int y=0; y<N; y++)
    {
        for(int x=0; x<M; x++)
        {
            matrix[y][x] = rand()%1024;
        }
    }

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    dim3 dimGrid((M + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE -1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    for(int i = 0; i < 20; i++)
    {
        gpu_shared_matrix_transpose<<<dimGrid,dimBlock>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix_transpose(matrix, cpu_result);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool errors = false;
    for(int y = 0; y<M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            if(fabs(cpu_result[y][x] - gpu_result[y][x]) > (1.0e-10))
            {
                errors = true;
            }
        }
        
    }

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f\nGPU time: %.2f\n", time_cpu, time_gpu/20.0);

    return 0;
}