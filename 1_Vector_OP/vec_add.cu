#include <stdio.h>
#include <math.h>


// x[index] + y[index] = z[index]
// 对应位置相加
/*
在设备的核函数中，我们用“单指令-多线程”的方式编写代
码，故可去掉该循环，只需将数组元素指标与线程指标一一对应即可。

核函数不允许有返回值, 但还是可以使用 return; 语句
*/
__global__ void VecAdd_GPU(const double *x, const double *y, double *z, int count)
{
    // 每一个线程的全局索引
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    // t00 t01 t02 t10 t11 t12 t20 t21 t22
    // t + block ids + thread ids
    /* example
    t21  ==> index = 7
    block ids  = 2  
    thread ids = 1
    blockDim   = 3  # block 的维度: 0 1 2

    blockDim.x * blockIdx.x ==> 表示元素所在位置之前有多少个thread
    */
    if( index < count )
    {
        z[index] = x[index] + y[index];
    }
    /* 此处加判断的原因: 
    当 N 是 blockDim.x（即 block_size）的整数倍时，不会引起问题，因为核函数中的线程数目刚好等于数组元素的个数。
    然而，当 N 不是 blockDim.x 的整数倍时，就有可能引发错误。

    对应此处代码其实可以不用判断, 因为在设置的 grid_size 的时候已经进行了取整:
    grid_size = (N + block_size -1)/block_size;  // 取整
    */
}

void VecAdd_CPU(const double *x, const double *y, double *z, int count)
{
    for(int i = 0; i<count; ++i)
    {
        z[i] = x[i] + y[i];
    }
}


int main()
{
    const int N = 1000;
    const int M = sizeof(double) * N;

    // cpu mem alloc
    double *cpu_x = (double*) malloc(M);
    double *cpu_y = (double*) malloc(M);
    double *cpu_z = (double*) malloc(M);
    // gpu结果转cpu
    double *gpu2cpu_z = (double*) malloc(M);

    // fill data
    for( int i = 0; i<N; ++i)
    {
        cpu_x[i] = 1;
        cpu_y[i] = 2;
    }

    // gpu mem alloc
    double *gpu_x, *gpu_y, *gpu_z;
    cudaMalloc((void**) &gpu_x, M);  // 该函数的功能是改变指针 gpu_x 本身的值（将一个指针赋值给 gpu_x），而不是改变 gpu_x 所指内存缓冲区中的变量值。
    cudaMalloc((void**) &gpu_y, M);
    cudaMalloc((void**) &gpu_z, M);

    // CPU ==> GPU
    cudaMemcpy(gpu_x, cpu_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, cpu_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N + block_size -1)/block_size;  // 取整

    // GPU compute
    VecAdd_GPU<<<grid_size, block_size>>>(gpu_x, gpu_y, gpu_z, N);
    // GPU ==> CPU, gpu结果转cpu
    cudaMemcpy(gpu2cpu_z, gpu_z, M, cudaMemcpyDeviceToHost);
    // CPU compute
    VecAdd_CPU(cpu_x, cpu_y, cpu_z, N);

    // evaluate
    bool error = false;
    for(int i=0; i<N; ++i)
    {
        if(fabs(cpu_z[i] - gpu2cpu_z[i]) > (1.0e-10))
        {
            error = true;
            break;
        }
    }
    printf("Result: %s\n", error?"Errors":"Pass");

    free(cpu_x);
    free(cpu_y);
    free(cpu_z);
    free(gpu2cpu_z);
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);

    return 0;
}

// 在 CUDA 数学库中，还有很多类似的数学函数，如幂函数、三角函数、指数函数、对数函数等。这些函数可以在如下网站查询： http://docs.nvidia.com/cuda/cuda-math-api。
