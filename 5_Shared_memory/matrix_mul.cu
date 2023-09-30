/*
共享内存和寄存器类似，存在于芯片上，具有仅次于寄存器的读写速度，数量也有限。
不同于寄存器的是，共享内存对整个线程块可见，其生命周期也与整个线程块一致。
共享内存的主要作用是减少对全局内存的访问，或者改善对全局内存的访问模式。
*/

#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 16
// example: 
// a[][] * b[][] = c[][]
// 
//  a00 a01 a02 a03    b00 b01 b02 b03    c00 c01 c02 c03
//  a10 a11 a12 a13    b10 b11 b12 b13    c10 c11 c12 c13 
//  a20 a21 a22 a23    b20 b21 b22 b23    c20 c21 c22 c23
//  a30 a31 a32 a33    b30 b31 b32 b33    c30 c31 c32 c33
// 
// 实现
// a的行乘b的列 ==> 使用一维向量存储二维
// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31   # c21 ==> y=2(第几行), x=1(第几列)
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33
//
// index = y * size + x   # 宽高 size = 4 
// step 0 -> 3: 
//     a_index = y * size + step;  # 取一行
//     b_index = step * size + x;  # 取一列

/*
cudaMalloc -> global memory
data global memory -> shared memory
threads shared memory -> register  ==> faster
shared memory SM(stream multi-processor): same block and same shared memory
但是要注意的是 shared memory 不像 global memory 那么大

因此可以做以下优化:
例如对于c20 c21 c30 c31 的计算:
                               划分为2个子矩阵:
                                b00 b01  ==> sub_b_step_0
                                b10 b11
   
                                b20 b21  ==> sub_b_step_1
                                b30 b31
划分为2个子矩阵:
 sub_a_step_0:  sub_a_step_1:   sub_c:
 a20 a21        a22 a23         c20 c21
 a30 a31        a32 a33         c30 c31

最后的结果可以看作是两个子矩阵之和: sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;

伪代码:
 tmp = 0;
 for(int step =0; step < N/block_size; ++step)
      load sub_a_step to shared memory;
      load sub_b_step to shared memory;
      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
 sub_c = tmp;
*/

// 定义矩阵维度
#define M 1000
#define N 500
#define K 1000

// __managed__: 定义统一内存, 自动维护
__managed__ int a[M*N];  // M行N列
__managed__ int b[N*K];
__managed__ int c_gpu[M*K];
__managed__ int c_cpu[M*K];


__global__ void gpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];  // 共享内存, 不会重复申请, 指向同一个空间: __shared__
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp =0;
    int idx;
    for(int step=0; step <= n/BLOCK_SIZE; step++)
    {
        // 例如计算 c21
        // sub_a_step_0:  sub_a_step_1:  
        // a20 a21        a22 a23      
        // 
        // b01  ==> sub_b_step_0
        // b11
        // b21  ==> sub_b_step_1
        // b31
        int step_x = step * BLOCK_SIZE + threadIdx.x;
        int step_y = y;
        idx = step_y * n + step_x;
        if(step_x >= n || step_y >= m)
        {
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }

        step_x = x;
        step_y = step * BLOCK_SIZE + threadIdx.y;
        idx = step_y * k +step_x;
        if(step_x >= k || step_y >= n)
        {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }

        __syncthreads();  // 不同block是并行计算的, 需要先进行同步

        for(int i = 0; i < BLOCK_SIZE; i++)
        {
            tmp +=sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    if ( x < k && y < m)
    {
        c[y*k + x] = tmp; 
    }
}

void cpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    for( int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            int tmp = 0;
            for(int step =0; step < n; step++)
            {
                tmp += a[y*n + step] * b[step*k + x];
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // fill data
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand()%1024;
        }
    }
    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y*K + x] = rand()%1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE -1)/BLOCK_SIZE;  // 列
    unsigned int grid_y = (M + BLOCK_SIZE -1)/BLOCK_SIZE;  // 行

    dim3 dimGrid(grid_x, grid_y);  // M行K列  ==> c矩阵
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // gpu 计算
    gpu_matrix<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    cudaDeviceSynchronize();  // windows平台需要加
    // cpu 计算
    cpu_matrix(a, b, c_cpu, M, N, K);

    // evaluate
    bool errors = false;
    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(c_cpu[y*K + x] - c_gpu[y*K+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu: %d", c_cpu[y*K + x], c_gpu[y*K+x]);
            }
        }
    }
    printf("Result: %s\n", errors?"Error":"Pass");

    return 0;
}