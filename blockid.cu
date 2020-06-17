#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void print_threadIds()
{
    printf("blockIdx.x : %d blockIdx.y : %d blockIdx.z : %d gridDim.x : %d gridDim.y : %d gridDim.z : %d\n", blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z); 
}

int main()
{
    int nx=16, ny=16;

    dim3 block(8, 8);
    dim3 grid(nx/block.x, ny/block.y);
    print_threadIds <<<grid, block>>> ();
    cudaDeviceSynchronize();
    cudaDeviceReset();
}