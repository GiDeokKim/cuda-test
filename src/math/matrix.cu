#include "matrix.cuh"

__global__ void MatrixMultiplicationKernel(int* M_d, int* N_d, int* P_d,
                                           int width) {
  // 2D Thread ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int result = 0;
  int M_dElement = 0;
  int N_dElement = 0;

  for (int k = 0; k < width; ++k) {
    M_dElement = M_d[ty * width + k];
    N_dElement = N_d[k * width + tx];
    result += M_dElement * N_dElement;
  }

  P_d[ty * width + tx] = result;
}

int MatrixMultiplication(int* M, int* N, int* P, int width) {
  int size = width * width * sizeof(int);
  int* M_d;
  int* N_d;
  int* P_d;

  // Transfer M and N to device memory
  cudaError_t mallocStatus = cudaMalloc((void**)&M_d, size);
  if (mallocStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(mallocStatus)
              << std::endl;
    return 1;
  }

  cudaError_t memcpyStatus = cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
  if (memcpyStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(memcpyStatus)
              << std::endl;
    cudaFree(M_d);
    return 1;
  }

  mallocStatus = cudaMalloc((void**)&N_d, size);
  if (mallocStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(mallocStatus)
              << std::endl;
    return 1;
  }

  memcpyStatus = cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
  if (memcpyStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(memcpyStatus)
              << std::endl;
    cudaFree(M_d);
    cudaFree(N_d);
    return 1;
  }
  // Allocate P on the device
  mallocStatus = cudaMalloc((void**)&P_d, size);
  if (mallocStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(mallocStatus)
              << std::endl;
    return 1;
  }

  // Ensure all preceding operations are complete
  cudaError_t syncStatus = cudaDeviceSynchronize();
  if (syncStatus != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: "
              << cudaGetErrorString(syncStatus) << std::endl;
    cudaFree(M_d);
    cudaFree(N_d);
    return 1;
  }

  // Setup the execution configuration
  dim3 dimBlock(width, width);
  dim3 dimGrid(1, 1);

  // Launch the device computation threads
  MatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

  // Transfer P from the device to the host
  memcpyStatus = cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
  if (memcpyStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(memcpyStatus)
              << std::endl;
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
    return 1;
  }
  syncStatus = cudaDeviceSynchronize();
  if (syncStatus != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize failed: "
              << cudaGetErrorString(syncStatus) << std::endl;
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
    return 1;
  }
  // Free device matrices
  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);

  return 0;
}