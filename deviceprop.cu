#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using namespace std;

int main()
{
    int n;
    cudaGetDeviceCount(&n);
    cout<<"Number of CUDA enabled devices : "<<n<<endl;
    if(n!=0)
    {
        for(int i=0;i<n;i++)
        {
            cout<<"Device No. : "<<i<<endl;
            cudaDeviceProp iProp;
            cudaGetDeviceProperties(&iProp, i);
            cout<<"\tDevice Name               : "<<iProp.name<<endl;
            cout<<"\tNo. of multiprocessors    : "<<iProp.multiProcessorCount<<endl;
            cout<<"\tClock rate                : "<<iProp.clockRate<<" kHz"<<endl;
            cout<<"\tCoumpute Capability       : "<<iProp.major<<"."<<iProp.minor<<endl;
            cout<<"\tTotal Global Memory       : "<<iProp.totalGlobalMem<<" B"<<endl;
            cout<<"\tTotal Constant Memory     : "<<iProp.totalConstMem<<" B"<<endl;
            cout<<"\tShared Memory per Block   : "<<iProp.sharedMemPerBlock<<" B"<<endl;
            cout<<"\tRegisters per block       : "<<iProp.regsPerBlock<<endl;
            cout<<"\tWarp Size                 : "<<iProp.warpSize<<endl;
            cout<<"\tMaximum thread per block  : "<<iProp.maxThreadsPerBlock<<endl;
            cout<<"\tMaximum thread dimensions : ("<<iProp.maxThreadsDim[0]<<", "<<iProp.maxThreadsDim[1]<<", "<<iProp.maxThreadsDim[2]<<")"<<endl;
            cout<<"\tMaximum grid size         : ("<<iProp.maxGridSize[0]<<", "<<iProp.maxGridSize[1]<<", "<<iProp.maxGridSize[2]<<")"<<endl;
        }
    }
    return 0;
}