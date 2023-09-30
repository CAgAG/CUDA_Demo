// https://github.com/brucefan1983/CUDA-Programming/blob/master/src/09-atomic/neighbor1cpu.cu

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
const int MN = 10; // maximum number of neighbors for each atom: 最多邻居数
const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(std::vector<real>& x, std::vector<real>& y);
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);
void print_neighbor(const int *NN, const int *NL);

int main(void)
{
    std::vector<real> x, y;
    read_xy(x, y);
    N = x.size();  // 节点数量
    // 符合阈值 cutoff_square 的节点数量: 例如 NN[n] 是第 n 个粒子的邻居个数
    int *NN = (int*) malloc(N * sizeof(int));  
    // 一维 邻接矩阵, 但不是全部 N 个节点, 而是限制 MN 个节点: NL[n * MN + k] 是第 n 个粒子的第 k 个邻居
    int *NL = (int*) malloc(N * MN * sizeof(int));  
    
    timing(NN, NL, x, y);
    print_neighbor(NN, NL);

    free(NN);
    free(NL);
    return 0;
}

// 读取 x, y 坐标给 v_x, v_y
void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream words(line);
        if(line.length() == 0)
        {
            continue;
        }
        for (int i = 0; i < 2; i++)
        {
            if(words >> word)
            {
                if(i == 0)
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

void find_neighbor(int *NN, int *NL, const real* x, const real* y)
{
    for (int n = 0; n < N; n++)
    {
        NN[n] = 0;
    }

    // n1 和 n2 表示的既是 x,y 的索引 也是节点
    for (int n1 = 0; n1 < N; ++n1)
    {
        real x1 = x[n1];
        real y1 = y[n1];
        // 只考虑 n2 > n1 的情形，从而省去一半的计算
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            real x12 = x[n2] - x1;
            real y12 = y[n2] - y1;
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                // 超过 MN 的节点, 会被之后的节点覆盖
                NL[n1 * MN + NN[n1]++] = n2;
                NL[n2 * MN + NN[n2]++] = n1;
            }
        }
    }
}

// 给 find_neighbor 函数计时
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        while(cudaEventQuery(start)!=cudaSuccess){}
        find_neighbor(NN, NL, x.data(), y.data());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void print_neighbor(const int *NN, const int *NL)
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
                outfile << " " << NL[n * MN + k];
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

