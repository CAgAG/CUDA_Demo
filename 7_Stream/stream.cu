// 之前的算法是 kernel 内的并行, cuda stream 是在 kernel 外的并行
//   ==> cuda lib: cudnn cublas tensort
// stream: 一系列的指令执行队列
// multi-stream -- asyn -- order -- asyn  多个流 异步进行 

#include <stdio.h>
#include <math.h>

#define N (1024 * 1024)
#define FULL_SIZE (N * 30)  // 数据规模

// a[] + b[] = c[]
__global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if( idx < N)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0;

        c[idx] = (as + bs)/2;
    }
}


int main()
{
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if( !prop.deviceOverlap )
    {
        printf("Your device do not support speed up from multi-streams \n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t my_streams[3];  // 定义 3 条流

    int *h_a, *h_b, *h_c;
    // 流数据
    int *d_a0, *d_b0, *d_c0;
    int *d_a1, *d_b1, *d_c1;
    int *d_a2, *d_b2, *d_c2;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStreamCreate(&my_streams[0]);
    cudaStreamCreate(&my_streams[1]);
    cudaStreamCreate(&my_streams[2]);

    cudaMalloc((void**) &d_a0, N * sizeof(int));
    cudaMalloc((void**) &d_b0, N * sizeof(int));
    cudaMalloc((void**) &d_c0, N * sizeof(int));
    cudaMalloc((void**) &d_a1, N * sizeof(int));
    cudaMalloc((void**) &d_b1, N * sizeof(int));
    cudaMalloc((void**) &d_c1, N * sizeof(int));
    cudaMalloc((void**) &d_a2, N * sizeof(int));
    cudaMalloc((void**) &d_b2, N * sizeof(int));
    cudaMalloc((void**) &d_c2, N * sizeof(int));

    // 流(stream) 要求数据一直驻留在主机, 因此 cudaHostAlloc 手动分配主机 CPU 数据
    cudaHostAlloc((void**) &h_a, FULL_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**) &h_b, FULL_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**) &h_c, FULL_SIZE * sizeof(int), cudaHostAllocDefault);

    // fill data
    for(int i = 0; i<FULL_SIZE; i++)
    {
        h_a[i] = rand() % 1024;
        h_b[i] = rand() % 1024;
    }

    cudaEventRecord(start);
    for(int i = 0; i < FULL_SIZE; i += N * 3)
    {
        cudaMemcpyAsync(d_a0, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[0]);  // 异步传输数据给到 GPU ==> stream
        cudaMemcpyAsync(d_a1, h_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[1]);
        cudaMemcpyAsync(d_a2, h_a+i+N+N, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[2]);
        cudaMemcpyAsync(d_b0, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[0]);
        cudaMemcpyAsync(d_b1, h_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[1]);
        cudaMemcpyAsync(d_b2, h_a+i+N+N, N*sizeof(int), cudaMemcpyHostToDevice, my_streams[2]);

        kernel<<<N/256, 256, 0, my_streams[0]>>>(d_a0, d_b0, d_c0);  // 在不同的 stream 中执行
        kernel<<<N/256, 256, 0, my_streams[1]>>>(d_a1, d_b1, d_c1);
        kernel<<<N/256, 256, 0, my_streams[2]>>>(d_a2, d_b2, d_c2);

        cudaMemcpyAsync(h_c+i, d_c0, N*sizeof(int), cudaMemcpyDeviceToHost, my_streams[0]);
        cudaMemcpyAsync(h_c+i+N, d_c1, N*sizeof(int), cudaMemcpyDeviceToHost, my_streams[1]);
        cudaMemcpyAsync(h_c+i+N+N, d_c2, N*sizeof(int), cudaMemcpyDeviceToHost, my_streams[2]);
    }

    cudaStreamSynchronize(my_streams[0]);
    cudaStreamSynchronize(my_streams[1]);
    cudaStreamSynchronize(my_streams[2]);

    cudaEventRecord(stop, 0);  // 此处的 0 表示: 第 0 号 stream 结束时, 停止计时
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime , start, stop);
    printf("Time: %3.2f ms\n", elapsedTime);

    // cudaFree ...

    return 0;
}
