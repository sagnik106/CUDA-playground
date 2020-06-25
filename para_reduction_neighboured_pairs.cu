#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void reduction_neighbored_pairs(int * arr, int * temp, int l)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(gid>l)
        return;

    for(int offset=1;offset<=blockDim.x/2;offset*=2)
    {
        if(tid%(2*offset)==0)
        {
            arr[gid]+=arr[gid+offset];
        }
        __syncthreads();
    }
    if(tid==0)
    {
        temp[blockIdx.x]=arr[gid];
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
    int shape=1<<27;
    int size=shape*sizeof(int);
    int block_size=128;

    dim3 block(block_size);
    dim3 grid(shape/block.x);

    int * arr;
    arr=(int *)malloc(size);
    
    
    int temp_size=sizeof(int)*grid.x;
    int * tarr;
    tarr=(int *)malloc(temp_size);
    
    
    int * d_arr, * d_temp;
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_temp, temp_size);
    cudaMemset(d_temp, 0, temp_size);


    for(int i=0; i< shape; i++)
    {
        arr[i]=(int)(rand() & 0x0f);
    }

    clock_t ct1,ct2,gt1,gt2;
    ct1=clock();
    int cpu=cpu_summer(arr, shape);
    ct2=clock();

    gt1=clock();
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    reduction_neighbored_pairs<<<grid, block>>>(d_arr, d_temp, shape);
    cudaDeviceSynchronize();
    
    cudaMemcpy(tarr, d_temp, temp_size, cudaMemcpyDeviceToHost);

    int gpu=0;
    for(int i=0;i<grid.x;i++)
    {
        gpu+=tarr[i];
    }

    gt2=clock();

    printf(cpu==gpu?"CPU and GPU values Match\n":"CPU and GPU values do not match\n");
    printf("GPU time : %lf sec\n",(double)((gt2-gt1)/(double)CLOCKS_PER_SEC));
    printf("CPU time : %lf sec\n",(double)((ct2-ct1)/(double)CLOCKS_PER_SEC));

    cudaFree(d_arr);
    cudaFree(d_temp);
    free(arr);
    free(tarr);
    cudaDeviceReset();

    return 0;
}