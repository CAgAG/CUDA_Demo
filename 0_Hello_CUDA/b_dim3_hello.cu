#include <stdio.h>

/*
在核函数内部，程序是知道执行配置参数 grid_size 和 block_size 的值的。
对于一维而言:
    gridDim.x：该变量的数值等于执行配置中变量 grid_size 的数值。
    blockDim.x：该变量的数值等于执行配置中变量 block_size 的数值。

    blockIdx.x：该变量指定一个线程在一个网格中的线程块指标，其取值范围是从 0 到 gridDim.x - 1。
    threadIdx.x：该变量指定一个线程在一个线程块中的线程指标，其取值范围是从 0 到 blockDim.x - 1。
拓展到 x,y,z 三维对应即可。
多维的网格和线程块本质上还是一维的，就像多维数组本质上也是一维数组一样。

对任何从开普勒到安培架构的 GPU 来说，网格大小在 x、 y 和 z 这 3 个方向的最大允许值分别为 2^31−1、 65535 和 65535；
线程块大小在 x、 y 和 z 这 3 个方向的最大允许值分别为 1024、 1024 和 64。
另外还要求线程块总的大小，即 blockDim.x、 blockDim.y 和 blockDim.z 的乘积不能大于 1024。也就是说，不管如何定义，一个线程块最多只能有 1024 个线程。
*/

__global__ void first_kernel()
{   
    // thread unique ID in block
    int tidx = threadIdx.x;  
    int tidy = threadIdx.y;

    // block unique ID in grid
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    // x 维度的线程指标 threadIdx.x 是最内层的（变化最快）
    printf("hello CUDA thread (block ID: (%d, %d), thread ID: (%d, %d))! \n", bidy, bidx, tidy, tidx);
}


int main()
{
    printf("hello CPU! \n");

    // 三(二)维形式的 block, 对应 blockIdx, blockIdy and threadIdx, threadIdy
    /*  example:
    t00, t01, t02
    t10, t11, t12
    t20, t21, t22 
    */
    dim3 block_size(3,3);

    // 三(二)维形式的 grid
    /* example:
    b00, b01
    b10, b11
    */
    dim3 grid_size(2,2);
    
    // print grids(2*2=4) * blocks(3*3=9)  times "hello ..."
    first_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}

/* 三维定义:
dim3 grid_size(Gx, Gy, Gz);
dim3 block_size(Bx, By, Bz);

如果第三个维度的大小是 1，可以写
dim3 grid_size(Gx, Gy);
dim3 block_size(Bx, By);
*/