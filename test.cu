// nvcc -g -G test.cu -o test

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

// compute最终会在gpu上，以3个线程启动进行执行
__global__ void compute(float* a, float* b, float* c)
{

    /* 
    * The function invokes kernel func on gridDim (gridDim.x × gridDim.y × gridDim.z) grid of blocks. 
    * Each block contains blockDim (blockDim.x × blockDim.y × blockDim.z) threads.
    * gridDim、blockDim、blockIdx、threadIdx是系统内置的变量，可以直接访问 
    * gridDim，对应网格维度，由kernel启动时指定，deviceQuery中明确了，gridDim的最大值是受限的 Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
    * blockDim，每个网格里面块的维度，由kernel启动时指定，deviceQuery中明确了，gridDim的最大值是受限的 Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    * blockIdx，是核在运行时，所处block的索引
    * threadIdx，是核在运行时，所处thread的索引
    * grid、block是虚拟的，由CUDA的任务调度器管理并分配到真实cuda core中，实际每次启动的线程数由调度器决定
    * gridDim、blockDim都是dim3类型，具有x、y、z属性值，blockIdx、threadIdx类型是uint3，具有x、y、z属性值
    * 哪个线程先运行是不确定的
    */

    int d0 = gridDim.z;
    int d1 = gridDim.y;
    int d2 = gridDim.x;
    int d3 = blockDim.z;
    int d4 = blockDim.y;
    int d5 = blockDim.x;

    // 构成了一个tensor是d0 x d1 x d2 x d3 x d4 x d5
    int p0 = blockIdx.z;
    int p1 = blockIdx.y;
    int p2 = blockIdx.x;
    int p3 = threadIdx.z;
    int p4 = threadIdx.y;
    int p5 = threadIdx.x;

    int position = (((((p0 * d1) + p1) * d2 + p2) * d3 + p3) * d4 + p4) * d5 + p5;

    //int position = ((blockIdx.y * gridDim.x) + blockIdx.x + threadIdx.y) * blockDim.x + threadIdx.x;
    //int position = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    //int position = (blockDim.x * blockIdx.x + threadIdx.x);
    c[position] = a[position] * b[position];

    printf("gridDim = %dx%dx%d, blockDim = %dx%dx%d, [blockIdx = %d,%d,%d, threadIdx = %d,%d,%d], position = %d, avalue = %f\n", 
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z,
        position, a[position]
    );
}

void query_device_info()
{
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("Device id:                                 %d\n",
        device_id);
    printf("Device name:                               %s\n",
        prop.name);
    printf("Compute capability:                        %d.%d\n",
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n",
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);
    // 在 CUDA 工具箱中，有一个名为 deviceQuery.cu 的程序，可以输出更多的信息
}

int main()
{
    query_device_info();

    const int num = 16;
    float a[num] = {1, 2, 3};
    float b[num] = {5, 7, 9};
    float c[num] = {0};
    for(int i = 0; i < num; ++i){
        a[i] = i;
        b[i] = i;
    }

    size_t size_array = sizeof(c);
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    
    // 分配GPU中的内存，大小是3个float，也就是12个字节
    cudaMalloc(&device_a, size_array);
    cudaMalloc(&device_b, size_array);
    cudaMalloc(&device_c, size_array);
    
    // 把cpu中的数组复制到GPU中
    cudaMemcpy(device_a, a, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size_array, cudaMemcpyHostToDevice);
    
    // 启动核函数compute，并以1个block和32个thread的方式进行运行
    compute<<<1, 32>>>(device_a, device_b, device_c);
    
    // 等待核函数执行完毕后，将数据复制到CPU（host）上变量c中
    cudaMemcpy(c, device_c, size_array, cudaMemcpyDeviceToHost);
    
    // 打印c中的数据
    for(int i = 8; i < 8 + 3; ++i){
        printf("value.%d = %f\n", i, c[i]);
    }
    return 0;
}