#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void reduction_neighbored_pairs(int * arr, int l, int offset)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid>=offset)
        return;

    if(gid<offset)
    {
        arr[gid]+=arr[gid+offset];
        arr[gid+offset]=0;
    } 
}

int cpu_summer(int * arr, int l)
{
    int s=0;
    for(int i=0;i<l;i++)
    {
        s+=arr[i];
    }
    return s;
}

int main()
{   
    srand(time(0));
    int shape=1<<27;
    int size=shape*sizeof(int);
    int block_size=128;

    dim3 block(block_size);
    dim3 grid(shape>>1/block.x);

    int * arr;
    arr=(int *)malloc(size);
        
    int * d_arr;
    cudaMalloc((void**)&d_arr, size);

    clock_t ct1,ct2,gt1,gt2,gtt1,gtt2;
    
    printf("CPU,GPU,GPU memory transfer");
    for(int counter=0;counter<200;counter++)
    {
        for(int i=0; i< shape; i++)
        {
            arr[i]=(int)(rand() & 0x0f);
        }

        ct1=clock();
        int cpu=cpu_summer(arr, shape);
        ct2=clock();

        gtt1=clock();
        cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
        gt1=clock();
        for(int offset=shape>>1;offset!=0;offset=offset>>1)
        {
            grid.x=offset>block.x?offset/block.x:1;
            reduction_neighbored_pairs<<<grid, block>>>(d_arr, shape, offset);
            cudaDeviceSynchronize();
        }
        gt2=clock();
        cudaMemcpy(arr, d_arr, sizeof(int), cudaMemcpyDeviceToHost);
        gtt2=clock();

        printf("\n%lf,%lf,%lf",(double)((ct2-ct1)/(double)CLOCKS_PER_SEC),(double)((gt2-gt1)/(double)CLOCKS_PER_SEC),(double)((-gt2+gt1+gtt2-gtt1)/(double)CLOCKS_PER_SEC));
    }
    cudaFree(d_arr);
    free(arr);
    cudaDeviceReset();

    return 0;
}