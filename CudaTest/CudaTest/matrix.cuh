#ifndef CUDA_TEST_MATRIX_H
#define CUDA_TEST_MATRIX_H
#include <cuda_runtime.h>

#include <iostream>

#include "device_launch_parameters.h"

int MatrixMultiplication(int* M, int* N, int* P, int width);

#endif  // CUDA_TEST_MATRIX_H
