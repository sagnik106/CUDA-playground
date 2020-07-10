#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void adderboi(int * a, int * b, int * c)
{
    int gid=threadIdx.x + blockIdx.x * blockDim.x;
    c[gid] = a[gid] + b[gid];
}

int cpu_adder(int * a, int * b, int * c, int shape)
{
    for(int i=0;i<shape;i++)
    {
        if(c[i]!=a[i]+b[i])
            return 0;
    }
    return 1;
}

int main()
{
    int shape=1<<22;
    int size = shape*sizeof(int);
    int b;
    printf("Enter block size : ");
    scanf("%d",&b);
    dim3 grid(shape/b);

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

    adderboi <<<grid, b>>> (d_arr1, d_arr2, d_arr3);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr3, d_arr3, size, cudaMemcpyDeviceToHost);

    printf(cpu_adder(h_arr1, h_arr2, h_arr3, shape)?"CPU and GPU values match\n":"CPU and GPU values donot match\n");
    /*for(int i=0;i<shape;i++)
    {
        printf("%d\t+\t%d\t= %d\n", h_arr1[i], h_arr2[i], h_arr3[i]);
    }*/

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_arr3);
    free(h_arr1);
    free(h_arr2);
    free(h_arr3);
    cudaDeviceReset();
}