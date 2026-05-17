#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void reduceSum(int* input, int* blockSums, int n) {

    __shared__ int sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (index < n) ? input[index] : 0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = sharedData[0];
    }
}

int reduceCPU(int* data, int n) {

    int sum = 0;

    for (int i = 0; i < n; i++) {
        sum += data[i];
    }

    return sum;
}

int main() {

    int* h_input = new int[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    // CPU süre ölçümü
    auto cpu_start = std::chrono::high_resolution_clock::now();

    int cpu_result = reduceCPU(h_input, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_end - cpu_start);

    // GPU değişkenleri
    int* d_input, * d_blockSums;

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    cudaMemcpy(
        d_input,
        h_input,
        N * sizeof(int),
        cudaMemcpyHostToDevice
    );

    // GPU süre ölçümü
    auto gpu_start = std::chrono::high_resolution_clock::now();

    reduceSum << <numBlocks, BLOCK_SIZE >> > (
        d_input,
        d_blockSums,
        N
        );

    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto gpu_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            gpu_end - gpu_start);

    int* h_blockSums = new int[numBlocks];

    cudaMemcpy(
        h_blockSums,
        d_blockSums,
        numBlocks * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    int gpu_result = 0;

    for (int i = 0; i < numBlocks; i++) {
        gpu_result += h_blockSums[i];
    }

    std::cout << "CPU sonucu: " << cpu_result << "\n";
    std::cout << "GPU sonucu: " << gpu_result << "\n";

    std::cout << "CPU suresi: "
        << cpu_duration.count()
        << " mikro saniye\n";

    std::cout << "GPU suresi: "
        << gpu_duration.count()
        << " mikro saniye\n";

    delete[] h_input;
    delete[] h_blockSums;

    cudaFree(d_input);
    cudaFree(d_blockSums);

    return 0;
}