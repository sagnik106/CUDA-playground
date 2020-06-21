#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


__global__ void prt_details_wrp()
{
    int gid=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    int warpid=threadIdx.x/32;
    int flatbid = blockIdx.y*gridDim.x+blockIdx.x;
    printf("gid : %d, warpid : %d, flattened bid : %d\n",gid, warpid,flatbid);
}

int main()
{
    dim3 block(42);
    dim3 grid(2, 2);
    prt_details_wrp <<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}