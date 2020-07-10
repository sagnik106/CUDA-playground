#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__ void dynamic_para(int depth)
{
    printf("Depth : %d - tid : %d \n", depth, threadIdx.x);
    if(blockDim.x==1)
        return;
    
        if(threadIdx.x==0)
        {
            dynamic_para<<<1, blockDim.x/2>>>(depth+1);
        }
}

int main()
{
    dynamic_para<<<1, 16>>>(0);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}