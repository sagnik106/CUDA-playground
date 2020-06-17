#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring> //for memset

__global__ void adderboi(int * a, int * b, int * c)
{
    int gid=threadIdx.x + blockIdx.x * blockDim.x;
    c[gid] = a[gid] + b[gid];
}

int main()
{
    int shape=10;
    int size = shape*sizeof(int);

    int * h_arr1;
    int * h_arr2;
    int * h_arr3;

    h_arr1=(int *)malloc(size);
    h_arr2=(int *)malloc(size);
    h_arr3=(int *)malloc(size);

    for(int i=0; i< shape; i++)
    {
        h_arr1[i]=(int)(rand() & 0x0f);
        h_arr2[i]=(int)(rand() & 0x0f);
        h_arr3[i]=0;
    }

    int * d_arr1;
    int * d_arr2;
    int * d_arr3;

    cudaMalloc((int**)&d_arr1, size);
    cudaMalloc((int**)&d_arr2, size);
    cudaMalloc((int**)&d_arr3, size);

    cudaMemcpy(d_arr1, h_arr1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, size, cudaMemcpyHostToDevice);

    adderboi <<<1, shape>>> (d_arr1, d_arr2, d_arr3);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr3, d_arr3, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<shape;i++)
    {
        printf("%d\t+\t%d\t= %d\n", h_arr1[i], h_arr2[i], h_arr3[i]);
    }

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_arr3);
    free(h_arr1);
    free(h_arr2);
    free(h_arr3);
    cudaDeviceReset();
}