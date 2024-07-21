
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void hello_cuda() { printf("hello CUDA world \n"); }

__global__ void printThreadsInfo() {
  printf(
      "gridDim.x: %d, gridDim.y: %d, gridDim.z: %d, blockIdx.x: %d, "
      "blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, "
      "threadIdx.z: %d,\n",
      gridDim.x, gridDim.y, gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z,
      threadIdx.x, threadIdx.y, threadIdx.z);
}

// int main() {
//   int nx, ny;
//   nx = 16;
//   ny = 4;
//   dim3 block(8, 2);
//   dim3 grid(nx / block.x, ny / block.y);

//   hello_cuda<<<grid, block>>>();
//   cudaDeviceSynchronize();

//   cudaDeviceReset();
//   return 0;
// }

int main() {
  dim3 block(2, 2, 2);
  dim3 grid(4, 4, 4);

  printThreadsInfo<<<grid, block>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}