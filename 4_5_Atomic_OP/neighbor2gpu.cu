// https://github.com/brucefan1983/CUDA-Programming/blob/master/src/09-atomic/neighbor2gpu.cu

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N; // number of atoms
const int NUM_REPEATS = 20; // number of timings
const int MN = 10; // maximum number of neighbors for each atom

const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(std::vector<real>& , std::vector<real>& );
void timing(int *, int *, const real *, const real *, const bool);
void print_neighbor(const int *, const int *, const bool);

int main(void)
{
    std::vector<real> v_x, v_y;
    read_xy(v_x, v_y);
    N = v_x.size();
    int mem1 = sizeof(int) * N;
    int mem2 = sizeof(int) * N * MN;
    int mem3 = sizeof(real) * N;
    int *h_NN = (int*) malloc(mem1);
    int *h_NL = (int*) malloc(mem2);
    int *d_NN, *d_NL;
    real *d_x, *d_y;
    cudaMalloc(&d_NN, mem1);
    cudaMalloc(&d_NL, mem2);
    cudaMalloc(&d_x, mem3);
    cudaMalloc(&d_y, mem3);
    cudaMemcpy(d_x, v_x.data(), mem3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, v_y.data(), mem3, cudaMemcpyHostToDevice);

    std::cout << std::endl << "using atomicAdd:" << std::endl;
    timing(d_NN, d_NL, d_x, d_y, true);
    std::cout << std::endl << "not using atomicAdd:" << std::endl;
    timing(d_NN, d_NL, d_x, d_y, false);


    cudaMemcpy(h_NN, d_NN, mem1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_NL, d_NL, mem2, cudaMemcpyDeviceToHost);

    print_neighbor(h_NN, h_NL, false);

    cudaFree(d_NN);
    cudaFree(d_NL);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_NN);
    free(h_NL);
    return 0;
}

void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while(std::getline(infile, line))
    {
        std::istringstream words(line);
        if(line.length()==0)
        {
            continue;
        }
        for(int i=0;i<2;i++)
        {
            if(words >> word)
            {
                if(i==0)
                {
                    v_x.push_back(std::stod(word));
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word));
                }
            }
            else
            {
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void __global__ find_neighbor_atomic
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y,
    const int N, const real cutoff_square
)
{
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        d_NN[n1] = 0;  // 首先将每一个原子的邻居数目初始化为零
        const real x1 = d_x[n1];
        const real y1 = d_y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            const real x12 = d_x[n2] - x1;
            const real y12 = d_y[n2] - y1;
            const real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                d_NL[n1 * MN + atomicAdd(&d_NN[n1], 1)] = n2;  // 原子操作
                d_NL[n2 * MN + atomicAdd(&d_NN[n2], 1)] = n1;

                /* 如果要把 atomicAdd(&d_NN[n1]) 分开, 应该使用下列方式:
                line 1 >>> int tmp1 = atomicAdd(&d_NN[n1], 1);
                line 2 >>> d_NL[n1 * MN + tmp1] = n2;

                如果 atomicAdd(&d_NN[n1], 1); 放在第二行, 
                并不能保证是第二行的原子函数读取的 d_NN[n1] 的值，而有可能是被别的线程的原子函数改动过的值。
                使用用临时变量 tmp1 和 tmp2 保留了 d_NN[n1] 和 d_NN[n2] 的 "旧值" 才是正确的做法。
                */
            }
        }
    }
}

void __global__ find_neighbor_no_atomic
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y,
    const int N, const real cutoff_square
)
{
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;  // 寄存器变量 count 减少了对全局内存变量 d_NN[n1] 的访问
        const real x1 = d_x[n1];
        const real y1 = d_y[n1];
        for (int n2 = 0; n2 < N; ++n2)
        {
            const real x12 = d_x[n2] - x1;
            const real y12 = d_y[n2] - y1;
            const real distance_square = x12 * x12 + y12 * y12;
            if ((distance_square < cutoff_square) && (n2 != n1))  // 循环变量 n2 的值包含了 n1, 此处需要判断
            {
                // 将 d_NL[n1 * MN + count++] 改成了 d_NL[(count++) * N + n1]。因为 n1 的变化步调与 threadIdx.x 一致。
                // 这样修改之后，对全局内存数组 d_NL 的访问将是合并的（不考虑数据不对齐带来的影响）。
                d_NL[(count++) * N + n1] = n2;
            }
        }
        d_NN[n1] = count;
    }
}  

void timing
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y, 
    const bool atomic
)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudaEventQuery(start);

        int block_size = 128;
        int grid_size = (N + block_size - 1) / block_size;

        if (atomic)
        {
            find_neighbor_atomic<<<grid_size, block_size>>>
            (d_NN, d_NL, d_x, d_y, N, cutoff_square);
        }
        else
        {
            find_neighbor_no_atomic<<<grid_size, block_size>>>
            (d_NN, d_NL, d_x, d_y, N, cutoff_square);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void print_neighbor(const int *NN, const int *NL, const bool atomic)
{
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN)
        {
            std::cout << "Error: MN is too small." << std::endl;
            exit(1);
        }
        outfile << NN[n];
        for (int k = 0; k < MN; ++k)
        {
            if(k < NN[n])
            {
                int tmp = atomic ? NL[n * MN + k] : NL[k * N + n];
                outfile << " " << tmp;
            }
            else
            {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}


