#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void memtransf(int * arr)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid : %d, gid : %d, value : %d\n", threadIdx.x, gid, arr[gid]);
}

int main()
{
    int shape = 128;
    int size = shape * sizeof(int);

    int * h_arr;
    h_arr = (int *)malloc(size);

    for(int i=0; i<shape;i++)
    {
        h_arr[i]=(int)(rand() & 0xff);
    }

    int * d_arr;
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);
    memtransf<<<grid, block>>>(d_arr);

    cudaDeviceSynchronize();

    cudaFree(d_arr);
    free(h_arr);

    cudaDeviceReset();
    return 0;
}