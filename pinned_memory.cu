#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void pined_memory()
{
    int shape = 1<<25;
    int size = shape * sizeof(float);

    float * h_a;
    cudaMallocHost((float **)&h_a, size);

    float * d_a;
    cudaMalloc((float **)&d_a, size);

    for(int i=0;i<shape;i++)
    {
        h_a[i]=6;
    }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFreeHost(h_a);

    cudaDeviceReset();
}

void paged_memory()
{
    int shape = 1<<25;
    int size = shape * sizeof(float);

    float * h_a = (float *)malloc(size);

    float * d_a;
    cudaMalloc((float **)&d_a, size);

    for(int i=0;i<shape;i++)
    {
        h_a[i]=6;
    }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    free(h_a);

    cudaDeviceReset();
}

int main()
{
    int n;
    printf("Enter 1 for paged and 2 for pinned : ");
    scanf("%d",&n);
    switch(n)
    {
        case 1: paged_memory();break;
        case 2: pined_memory();break;
    }
    return 0;
}