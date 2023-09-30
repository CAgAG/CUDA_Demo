#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 32
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

void cpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    for(int y=0; y<size; ++y)
    {
        for(int x=0; x<size; ++x)
        {
            int tmp = 0;
            for(int step = 0; step < size; ++step)
            {
                tmp += a[y*size + step] * b[step * size + x];
            }
            c[y * size + x] = tmp;
        }
    }
}

__global__ void gpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int tmp = 0;
    if( x < size && y < size)
    {
        for( int step = 0; step < size; ++step)
        {
            tmp += a[y * size + step] * b[step * size + x];
        }
        c[y * size + x] = tmp;
    }
}


int main()
{
    int matrix_size = 1000;
    int memsize = sizeof(int) * matrix_size * matrix_size;

    // 使用一维向量存储二维
    int *cpu_a, *cpu_b, *cpu_c, *gpu2cpu_c;
    // cpu的初始化
    cudaMallocHost( (void**)&cpu_a, memsize);
    cudaMallocHost( (void**)&cpu_b, memsize);
    cudaMallocHost( (void**)&cpu_c, memsize);
    cudaMallocHost( (void**)&gpu2cpu_c, memsize);

    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            cpu_a[y * matrix_size + x] = rand() % 1024;
        }
    }

    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            cpu_b[y * matrix_size + x] = rand() % 1024;
        }
    }

    int *gpu_a, *gpu_b, *gpu_c;
    // gpu的初始化
    cudaMalloc((void**) &gpu_a , memsize);
    cudaMalloc((void**) &gpu_b , memsize);
    cudaMalloc((void**) &gpu_c , memsize);

    cudaMemcpy( gpu_a, cpu_a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy( gpu_b,cpu_b, memsize, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (matrix_size + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols, grid_rows);
    // 1! advice: gpu warp 32
    // 2. BLOCK_SIZE * BLOCK_SIZE <= 1024
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // gpu compute
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c, matrix_size);
    cudaMemcpy(gpu2cpu_c, gpu_c, memsize, cudaMemcpyDeviceToHost);
    // cpu compute
    cpu_matrix_mult(cpu_a, cpu_b, cpu_c, matrix_size);

    // evaluate
    bool errors = false;
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            if(fabs(gpu2cpu_c[y*matrix_size + x] - cpu_c[y*matrix_size + x]) > (1.0e-10))
            {
                //printf("%d, %d\n", y, x);
                errors = true;
            }
        }
    }
    printf("Result: %s\n", errors?"Errors":"Pass");

    cudaFreeHost(cpu_a);
    cudaFreeHost(cpu_b);
    cudaFreeHost(cpu_c);
    cudaFreeHost(gpu2cpu_c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}