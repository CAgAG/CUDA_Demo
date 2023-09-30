/*
CUDA:
包含层级: 
grid -->block --> thread

每个线程块的计算是相互独立的。不管完成计算的次序如何，每个线程块中的每个线程都进行一次计算, 所以:
total threads = number of runs = grids * blocks

一个线程块中的线程还可以细分为不同的线程束（thread warp）。一个线程束（即一束线程）是同一个线程块中相邻的 warpSize 个线程。 
warpSize 也是一个内建变量，表示线程束大小，其值对于目前所有的 GPU 架构都是 32。
所以，一个线程束就是连续的 32 个线程。
具体地说，一个线程块中第 0 到第 31 个线程属于第 0 个线程束，第 32 到第 63 个线程属于第 1 个线程束，依此类推。
因此建议将线程块大小取为 32 的整数倍。
*/

#include <stdio.h>

__global__ void first_kernel()
{   
    // thread unique ID in block
    int tidx = threadIdx.x;  

    // block unique ID in grid
    int bidx = blockIdx.x;

    printf("hello CUDA thread (block ID: %d, thread ID: %d)! \n", bidx, tidx);

}


int main()
{
    printf("hello CPU! \n");

    // 一维形式的 block 和 grid
    // eg: block --> t0, t1, t2 ...  --> 只对应 threadIdx
    int grid_nums = 3;
    int block_nums = 2;

    // print grids * blocks  times "hello ..."
    // 括号中的第一个数字可以看作线程块的个数，第二个数字可以看作每个线程块中的线程数。
    /*
    一个核函数的全部线程块构成一个网格（grid），而线程块的个数就记为网格大小（grid size）。
    每个线程块中含有同样数目的线程，该数目称为线程块大小（block size）。
    所以，核函数中总的线程数就等于网格大小乘以线程块大小，而三括号中的两个数字分别就是网格大小和线程块大小，
    即 <<<网格大小, 线程块大小>>>。
    */
    first_kernel<<<grid_nums, block_nums>>>();

    // 注意: cudaMemcpy 就会进行同步，不需要执行执行 cudaDeviceSynchronize
    /*
    调用输出函数时，输出流是先存放在缓冲区的，而缓冲区不会自动刷新。
    只有程序遇到某种同步操作时缓冲区才会刷新。
    函数 cudaDeviceSynchronize 的作用是同步主机与设备，所以能够促使缓冲区刷新。
    */
    cudaDeviceSynchronize();

    return 0;
}