#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void code_wo_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a=b=0;
    int warp_id = gid/32;
    if(warp_id %2==0)
    {
        a=100.0;
        b=50.0;
    }
    else
    {
        a=200.0;
        b=75.0;
    }
}

__global__ void code_w_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a=b=0;
    if(gid%2==0)
    {
        a=100.0;
        b=50.0;
    }
    else
    {
        a=200.0;
        b=75.0;
    }
}

int main()
{
    int size = 1<<22;
    dim3 block(128);
    dim3 grid((size+block.x-1)/block.x);

    code_wo_divergence <<<grid, block>>>();
    cudaDeviceSynchronize();

    code_w_divergence <<<grid, block>>> ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}